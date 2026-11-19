"""Separación de los jugadores en dos equipos por el color de la camiseta.

Se clasifica el color del torso en cada fotograma, no solo una vez por track:
así se detectan los intercambios de identidad del tracker (cuando dos
jugadores de equipos distintos se cruzan) y se parte el track en ese punto.
"""

from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans

from src.seguimiento import Track

MAX_ENTRENAMIENTO = 4000

# Todas las predicciones de la última clasificación, incluidas las de poco margen
etiquetas_debiles: dict = {}


def _caracteristicas(recorte: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten()
    return hist / (hist.sum() + 1e-6)


def _color_visible(recortes: list[np.ndarray]) -> tuple[tuple[int, int, int], float]:
    """Color representativo para dibujar el equipo y su luminosidad."""
    pix = np.concatenate([cv2.cvtColor(cv2.resize(r, (12, 12)), cv2.COLOR_BGR2HSV).reshape(-1, 3)
                          for r in recortes])
    saturados = pix[(pix[:, 1] > 90) & (pix[:, 2] > 60)]
    brillo = float(np.median(pix[:, 2]))
    if len(saturados) / len(pix) > 0.25:
        tono = int(np.median(saturados[:, 0]))
        bgr = cv2.cvtColor(np.uint8([[[tono, 210, 215]]]), cv2.COLOR_HSV2BGR)[0, 0]
        return tuple(int(v) for v in bgr), brillo
    return ((230, 230, 230) if brillo > 80 else (45, 45, 45)), brillo


def clasificar(torsos: dict[tuple[int, int], np.ndarray], ocluidos: set):
    """Etiqueta cada (track_id, k) con equipo 0/1.

    Devuelve (etiquetas, colores, brillos).
    """
    claves = list(torsos)
    X = np.array([_caracteristicas(torsos[c]) for c in claves])
    limpios = [i for i, c in enumerate(claves) if c not in ocluidos] or list(range(len(claves)))
    rng = np.random.default_rng(0)
    if len(limpios) > MAX_ENTRENAMIENTO:
        limpios = list(rng.choice(limpios, MAX_ENTRENAMIENTO, replace=False))
    if len(limpios) < 2:
        raise RuntimeError("No hay suficientes recortes para separar equipos")

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X[limpios])
    distancias = kmeans.transform(X)                 # (N, 2)
    predicho = distancias.argmin(axis=1)
    d_propia = distancias.min(axis=1)
    d_otra = distancias.max(axis=1)
    margen = (d_otra - d_propia) / (d_otra + d_propia + 1e-9)
    # Solo se guardan etiquetas con margen claro; el resto se considera dudoso
    umbral = np.quantile(margen, 0.25)
    etiquetas = {c: int(e) for c, e, m in zip(claves, predicho, margen) if m >= umbral}
    # Etiquetas dudosas: solo se usan si un track no tiene ninguna clara
    etiquetas_debiles.clear()
    etiquetas_debiles.update({c: int(e) for c, e in zip(claves, predicho)})

    colores, brillos = {}, {}
    for e in (0, 1):
        muestras = [torsos[claves[i]] for i in limpios if predicho[i] == e][:600]
        colores[e], brillos[e] = _color_visible(muestras) if muestras else ((128, 128, 128), 128.0)
    # Si los dos colores salen iguales (p. ej. dos equipaciones oscuras), se fuerza contraste
    if colores[0] == colores[1]:
        colores[1] = (40, 40, 200)
    return etiquetas, colores, brillos


def _moda(valores: list[int], ventana: int) -> list[int]:
    r = []
    for i in range(len(valores)):
        trozo = valores[max(0, i - ventana // 2): i + ventana // 2 + 1]
        r.append(Counter(trozo).most_common(1)[0][0])
    return r


def dividir_por_equipo(tracks: dict[int, Track], etiquetas: dict, ocluidos: set,
                       fps: float, min_tramo_s: float = 0.6):
    """Parte los tracks donde el color de camiseta cambia de forma sostenida.

    Devuelve (tracks_nuevos, equipo_por_track).
    """
    siguiente_id = max(tracks) + 1 if tracks else 0
    nuevos, equipo_por_track = {}, {}
    divisiones = 0

    min_etiquetas = max(3, int(fps * 0.25))

    for tid in sorted(tracks):
        t = tracks[tid]
        # Cuentan las observaciones con etiqueta clara sin oclusión; las ocluidas
        # solo donde no hay ninguna limpia cerca (si no, un cambio de jugador
        # durante un cruce largo pasaría desapercibido)
        limpias = [(i, etiquetas[(tid, o.k)]) for i, o in enumerate(t.obs)
                   if (tid, o.k) in etiquetas and (tid, o.k) not in ocluidos]
        frames_limpios = np.array([t.obs[i].k for i, _ in limpias])
        cerca = int(fps * 0.5)
        fiables = list(limpias)
        for i, o in enumerate(t.obs):
            if (tid, o.k) in etiquetas and (tid, o.k) in ocluidos:
                if len(frames_limpios) == 0 or np.abs(frames_limpios - o.k).min() > cerca:
                    fiables.append((i, etiquetas[(tid, o.k)]))
        fiables.sort()
        if not fiables:
            fiables = [(i, etiquetas_debiles[(tid, o.k)]) for i, o in enumerate(t.obs)
                       if (tid, o.k) in etiquetas_debiles]
        if not fiables:
            continue
        indices = [i for i, _ in fiables]
        suave = _moda([e for _, e in fiables], int(fps * 0.5) | 1)

        # Tramos de etiquetas iguales: [primera_pos, ultima_pos, etiqueta, n_etiquetas]
        tramos = []
        for pos, e in zip(indices, suave):
            if tramos and tramos[-1][2] == e:
                tramos[-1][1] = pos
                tramos[-1][3] += 1
            else:
                tramos.append([pos, pos, e, 1])

        # Un tramo solo cuenta si es largo en tiempo y tiene etiquetas suficientes
        def solido(tr):
            # El primer tramo abarca desde el inicio del track y el último hasta
            # el final (los fotogramas dudosos de los extremos también son suyos)
            ini = 0 if tr is tramos[0] else tr[0]
            fin = len(t.obs) - 1 if tr is tramos[-1] else tr[1]
            duracion = (t.obs[fin].k - t.obs[ini].k) / fps
            return duracion >= min_tramo_s and tr[3] >= min_etiquetas

        fusionados = []
        for tr in tramos:
            if fusionados and (not solido(tr) or tr[2] == fusionados[-1][2]):
                fusionados[-1][1] = tr[1]
                fusionados[-1][3] += tr[3]
            else:
                fusionados.append(list(tr))
        if len(fusionados) > 1 and not solido(fusionados[0]):
            fusionados[1][0] = fusionados[0][0]
            fusionados[1][3] += fusionados[0][3]
            fusionados.pop(0)

        # Cortes a mitad de camino entre tramos
        cortes = [0]
        for a, b in zip(fusionados[:-1], fusionados[1:]):
            cortes.append((a[1] + b[0]) // 2 + 1)
        cortes.append(len(t.obs))

        for n in range(len(fusionados)):
            nuevo_id = tid if n == 0 else siguiente_id
            if n > 0:
                siguiente_id += 1
                divisiones += 1
            a, b = cortes[n], cortes[n + 1]
            votos = Counter(e for i, e in fiables if a <= i < b)
            nuevos[nuevo_id] = Track(nuevo_id, t.obs[a:b])
            equipo_por_track[nuevo_id] = (votos.most_common(1)[0][0] if votos
                                          else fusionados[n][2])

    conteo = Counter(equipo_por_track.values())
    print(f"Equipos por color: grupo 0 = {conteo[0]} tracks, grupo 1 = {conteo[1]} tracks "
          f"({divisiones} tracks partidos por cambio de camiseta)")
    return nuevos, equipo_por_track
