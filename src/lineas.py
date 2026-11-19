"""Refinado de la homografía con las líneas pintadas de la pista.

Los keypoints del modelo dan una homografía aproximada, pero con pocos puntos
en una zona (p. ej. la banda lejana) puede quedar desplazada varios píxeles y
un jugador que corre pegado a la línea aparece fuera de la pista.

Para cada fotograma:
  1. Se proyectan las líneas rectas de la pista (bandas, fondos, medio campo,
     zona y tiros libres) con la homografía actual.
  2. Se detectan en la imagen los segmentos rectos largos y continuos de trazo
     claro (top-hat + Hough), tapando a los jugadores. Las letras de
     publicidad, logos del suelo y reflejos dan trozos cortos y se ignoran.
  3. Cada línea esperada se empareja con el segmento detectado más cercano de
     su misma orientación, y sus puntos se llevan sobre esa línea real.
  4. Se recalcula la homografía con esos puntos más una rejilla de la
     homografía previa (que mantiene el resto del campo donde no se ven líneas).
"""

from pathlib import Path

import cv2
import numpy as np

from src import cancha
from src.recortes import leer_fotogramas

ESCALA = 0.5                 # se trabaja a media resolución
MUESTRAS_POR_LINEA = 40
MIN_PUNTOS_LINEA = 6


def _segmentos_pista() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    L, W = cancha.LARGO, cancha.ANCHO
    ancho_zona = 4.88
    largo_zona = 5.79
    y0 = (W - ancho_zona) / 2
    y1 = y0 + ancho_zona
    segmentos = [
        ((0, 0), (L, 0)), ((0, W), (L, W)),          # bandas
        ((0, 0), (0, W)), ((L, 0), (L, W)),          # fondos
        ((L / 2, 0), (L / 2, W)),                    # medio campo
    ]
    for x0, xz in ((0, largo_zona), (L, L - largo_zona)):
        segmentos += [((x0, y0), (xz, y0)), ((x0, y1), (xz, y1)), ((xz, y0), (xz, y1))]
    return segmentos


SEGMENTOS = _segmentos_pista()


def _respuesta_lineas(frame: np.ndarray) -> np.ndarray:
    gris = cv2.cvtColor(cv2.resize(frame, None, fx=ESCALA, fy=ESCALA,
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(gris, cv2.MORPH_TOPHAT, nucleo)
    return cv2.GaussianBlur(tophat, (3, 3), 0).astype(np.float32)


def _segmentos_detectados(respuesta: np.ndarray, cajas) -> np.ndarray:
    """Segmentos rectos largos y continuos (líneas pintadas). Las letras, logos
    y reflejos se rompen en trozos cortos y no pasan el filtro de longitud."""
    h, w = respuesta.shape
    # Umbral bajo: las líneas lejanas son finas y responden poco; el ruido
    # (letras, reflejos) se descarta luego por no formar rectas largas
    umbral = max(25.0, 0.3 * float(np.percentile(respuesta, 99.5)))
    binaria = (respuesta >= umbral).astype(np.uint8) * 255
    for c in cajas:                                   # los jugadores tapan líneas
        x1, y1, x2, y2 = (int(v * ESCALA) for v in c)
        binaria[max(0, y1):y2, max(0, x1):x2] = 0
    segs = cv2.HoughLinesP(binaria, 1, np.pi / 360, threshold=30,
                           minLineLength=25, maxLineGap=6)
    return np.empty((0, 4)) if segs is None else segs.reshape(-1, 4).astype(np.float32)


def _emparejar(p0, p1, segs, radio, min_longitud):
    """Busca la línea real correspondiente a la esperada p0→p1.

    Los jugadores cortan las líneas en varios trozos, así que se agrupan los
    segmentos con la misma orientación que caen sobre la misma recta y se
    elige el grupo con más longitud total (penalizando un poco la distancia).
    Devuelve (punto, dirección) de la recta real o None."""
    d = p1 - p0
    largo = np.linalg.norm(d)
    u = d / largo
    n = np.array([-u[1], u[0]])
    candidatos = []                 # (desplazamiento, longitud, a, b)
    for x1, y1, x2, y2 in segs:
        a, b = np.array([x1, y1]), np.array([x2, y2])
        s = b - a
        ls = np.linalg.norm(s)
        if ls < 15 or abs(np.dot(s / ls, u)) < np.cos(np.radians(3)):
            continue
        oa, ob = np.dot(a - p0, n), np.dot(b - p0, n)
        if max(abs(oa), abs(ob)) > radio:
            continue
        ta, tb = np.dot(a - p0, u), np.dot(b - p0, u)
        if max(ta, tb) < -0.1 * largo or min(ta, tb) > 1.1 * largo:
            continue
        candidatos.append(((oa + ob) / 2, ls, a, b))
    if not candidatos:
        return None

    mejor, mejor_puntos = None, -np.inf
    for c, _, _, _ in candidatos:
        grupo = [e for e in candidatos if abs(e[0] - c) < 3.0]
        total = sum(e[1] for e in grupo)
        if total < min_longitud:
            continue
        puntos = total - 2.0 * abs(c)
        if puntos > mejor_puntos:
            mejor, mejor_puntos = grupo, puntos
    if mejor is None:
        return None
    extremos = np.array([p for e in mejor for p in (e[2], e[3])], dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(extremos, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _refinar_fotograma(H, respuesta, cajas, ancho, alto, rejilla_img):
    """Una homografía imagen→pista refinada con las líneas, o None."""
    h, w = respuesta.shape
    segs = _segmentos_detectados(respuesta, cajas)
    if len(segs) == 0:
        return None

    H_actual = H
    for radio in (45, 15):                      # búsqueda amplia y luego fina
        Hi = np.linalg.inv(H_actual)
        img_pts, pista_pts = [], []
        for a, b in SEGMENTOS:
            ts = np.linspace(0, 1, MUESTRAS_POR_LINEA)
            pista = np.stack([a[0] + (b[0] - a[0]) * ts, a[1] + (b[1] - a[1]) * ts], axis=1)
            proy = cv2.perspectiveTransform(pista.reshape(-1, 1, 2).astype(np.float32),
                                            Hi).reshape(-1, 2) * ESCALA
            visibles = ((proy[:, 0] >= 0) & (proy[:, 0] < w) &
                        (proy[:, 1] >= 0) & (proy[:, 1] < h))
            if visibles.sum() < MIN_PUNTOS_LINEA:
                continue
            p0, p1 = proy[visibles][0], proy[visibles][-1]
            largo = np.linalg.norm(p1 - p0)
            if largo < 40:
                continue
            emparejado = _emparejar(p0, p1, segs, radio, min_longitud=max(40, 0.2 * largo))
            if emparejado is None:
                continue
            q, u = emparejado
            n = np.array([-u[1], u[0]])
            # Cada punto esperado se lleva en perpendicular sobre la línea real
            for p_img, p_pista in zip(proy[visibles], pista[visibles]):
                p_real = p_img - n * np.dot(p_img - q, n)
                img_pts.append(p_real / ESCALA)
                pista_pts.append(p_pista)

        if len(img_pts) < 12:
            return None if H_actual is H else H_actual
        # La rejilla de la homografía previa sujeta las zonas sin líneas visibles
        prior_pista = cancha.proyectar(H_actual, rejilla_img)
        src = np.vstack([np.array(img_pts, np.float32), rejilla_img])
        dst = np.vstack([np.array(pista_pts, np.float32), prior_pista])
        H_nueva, _ = cv2.findHomography(src, dst, 0)
        if H_nueva is None:
            return None if H_actual is H else H_actual
        H_actual = H_nueva / H_nueva[2, 2]
    return H_actual


def refinar(ruta_video: Path, paso: int, fotogramas: list[dict], homografias: list,
            ancho: int, alto: int, max_cambio_m: float = 2.0) -> list:
    xs = np.linspace(0.1 * ancho, 0.9 * ancho, 5)
    ys = np.linspace(0.45 * alto, 0.95 * alto, 3)
    rejilla = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)

    refinadas, mejoradas = list(homografias), 0
    for k, frame in leer_fotogramas(ruta_video, paso):
        if k >= len(fotogramas) or homografias[k] is None:
            continue
        cajas = [d["caja"] for d in fotogramas[k]["detecciones"]]
        H_ref = _refinar_fotograma(homografias[k], _respuesta_lineas(frame), cajas,
                                   ancho, alto, rejilla)
        if H_ref is None:
            continue
        # Seguridad: no aceptar cambios enormes (serían un mal emparejamiento)
        cambio = np.median(np.linalg.norm(
            cancha.proyectar(H_ref, rejilla) - cancha.proyectar(homografias[k], rejilla), axis=1))
        if cambio <= max_cambio_m:
            refinadas[k] = H_ref
            mejoradas += 1
    print(f"Homografía refinada con las líneas de la pista en {mejoradas}/{len(homografias)} fotogramas")
    return refinadas
