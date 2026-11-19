"""Seguimiento del balón.

1. En la imagen: entre todas las detecciones de "balón" del clip se elige la
   trayectoria más coherente (camino de coste mínimo). El balón no puede pasar
   de un sitio a otro lejano entre dos fotogramas: los desplazamientos se miden
   con el movimiento de cámara compensado y los imposibles se penalizan, así
   que objetos de la grada que el detector confunde con el balón quedan fuera.
   Los huecos cortos se rellenan con flujo óptico (Lucas-Kanade) hacia delante
   y hacia atrás desde las detecciones vecinas.

2. En la pizarra: el balón suele estar en el aire, así que no se proyecta
   directamente (la homografía solo vale para el suelo). Se usa:
     - "posesion": el balón está en la caja de un jugador → posición del jugador.
     - "aro": el balón está en el aro → la canasta correspondiente.
     - "aire": entre dos posiciones conocidas se interpola (pase, tiro, rebote).
"""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src import cancha
from src.identidad import Jugador
from src.recortes import leer_fotogramas

SALTO = 0.5           # coste por fotograma sin balón
REINICIO = 4.0        # coste de empezar un tramo nuevo lejos del anterior
VEL_LIBRE = 35.0      # px por fotograma (cámara compensada) permitidos sin coste
EXCESO_MAX = 160.0    # por encima, el salto se considera imposible


def _centro(caja):
    return np.array([(caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2], dtype=np.float32)


def _transformar(M, punto):
    p = cv2.perspectiveTransform(np.array([[punto]], dtype=np.float32), M)
    return p[0, 0]


def _flujo(previo, actual, centro, radio) -> np.ndarray | None:
    """Desplaza el balón con flujo óptico (ida y vuelta para validar)."""
    x, y = centro
    rejilla = np.array([(x + dx, y + dy) for dx in np.linspace(-radio, radio, 4)
                        for dy in np.linspace(-radio, radio, 4)], dtype=np.float32).reshape(-1, 1, 2)
    p1, e1, _ = cv2.calcOpticalFlowPyrLK(previo, actual, rejilla, None, winSize=(21, 21), maxLevel=3)
    if p1 is None:
        return None
    p0, e0, _ = cv2.calcOpticalFlowPyrLK(actual, previo, p1, None, winSize=(21, 21), maxLevel=3)
    ok = (e1.ravel() == 1) & (e0.ravel() == 1) & (np.linalg.norm(p0 - rejilla, axis=2).ravel() < 1.5)
    if ok.sum() < 4:
        return None
    desplazamiento = np.median((p1 - rejilla).reshape(-1, 2)[ok], axis=0)
    return np.asarray(centro, dtype=np.float32) + desplazamiento


def _cerca_de_jugadores(punto, cajas, factor=1.2) -> bool:
    for c in cajas:
        alto = c[3] - c[1]
        cx = (c[0] + c[2]) / 2
        if abs(punto[0] - cx) <= factor * alto * 0.6 and c[1] - factor * alto <= punto[1] <= c[3]:
            return True
    return False


def _acumulado_camara(movimiento: list) -> list[np.ndarray]:
    """C[k] lleva píxeles del fotograma k al sistema del fotograma 0 (imagen
    "estabilizada"): así las distancias entre fotogramas no incluyen el paneo."""
    C = [np.eye(3)]
    for k in range(1, len(movimiento)):
        M = movimiento[k] if movimiento[k] is not None else np.eye(3)
        Ck = C[-1] @ M
        C.append(Ck / Ck[2, 2])
    return C


def seguir_en_imagen(ruta_video: Path, paso: int, fotogramas: list[dict],
                     jugadores: list[Jugador], movimiento: list,
                     hueco_max: int = 45, hueco_relleno: int = 15) -> list[dict | None]:
    """Elige la trayectoria del balón más coherente de todo el clip.

    Grafo: cada detección de balón es un nodo; se puede ir de una detección a
    otra posterior (saltándose fotogramas). El camino de menor coste premia la
    confianza y la cercanía a los jugadores y penaliza los fotogramas saltados
    y los desplazamientos imposibles (medidos con la cámara compensada).
    Devuelve por fotograma {"pos", "radio", "fuente": "det"|"interp"} o None.
    """
    n = len(fotogramas)
    C = _acumulado_camara(movimiento)

    nodos = []      # (k, pos_imagen, pos_estabilizada, radio, coste_unario)
    for k, f in enumerate(fotogramas):
        cajas_jug = [j.cajas[k] for j in jugadores if k in j.cajas]
        for d in f["detecciones"]:
            if d["clase"].lower() != "ball":
                continue
            p = _centro(d["caja"])
            q = _transformar(C[k], p)
            cerca = _cerca_de_jugadores(p, cajas_jug, factor=1.0)
            algo_cerca = cerca or _cerca_de_jugadores(p, cajas_jug, factor=2.0)
            unario = 0.8 - 1.5 * d["conf"] + (0.0 if cerca else 0.6 if algo_cerca else 1.5)
            radio = max(6.0, (d["caja"][2] - d["caja"][0]) / 2)
            nodos.append((k, p, q, radio, unario))

    coste = [0.0] * len(nodos)
    previo = [None] * len(nodos)
    for i, (k, p, q, r, u) in enumerate(nodos):
        mejor, origen = SALTO * k, None              # empezar aquí
        for h in range(i - 1, -1, -1):
            kh, _, qh, _, _ = nodos[h]
            g = k - kh
            if g <= 0:
                continue
            if g > hueco_max:
                break
            exceso = max(0.0, float(np.linalg.norm(q - qh)) - VEL_LIBRE * g)
            transicion = exceso / 40.0 if exceso <= EXCESO_MAX else REINICIO
            c = coste[h] + transicion + SALTO * (g - 1)
            if c < mejor:
                mejor, origen = c, h
        coste[i], previo[i] = mejor + u, origen

    resultado = [None] * n
    if not nodos:
        print("Balón en imagen: sin detecciones")
        return resultado
    final = min(range(len(nodos)), key=lambda i: coste[i] + SALTO * (n - 1 - nodos[i][0]))
    camino, i = [], final
    while i is not None:
        camino.append(i)
        i = previo[i]
    camino.reverse()

    for i in camino:
        k, p, q, r, _ = nodos[i]
        resultado[k] = {"pos": p, "radio": r, "fuente": "det"}
    # Huecos cortos entre detecciones consecutivas de la trayectoria
    huecos = []
    for a, b in zip(camino[:-1], camino[1:]):
        if 1 < nodos[b][0] - nodos[a][0] <= hueco_relleno:
            huecos.append((nodos[a], nodos[b]))
    necesarios = {k for na, nb in huecos for k in range(na[0], nb[0] + 1)}
    grises = {}
    if necesarios:
        for k, frame in leer_fotogramas(ruta_video, paso):
            if k in necesarios:
                grises[k] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if k > max(necesarios):
                break

    con_flujo = 0
    for (ka, pa, qa, ra, _), (kb, pb, qb, rb, _) in huecos:
        radio = (ra + rb) / 2
        # Flujo óptico hacia delante desde a y hacia atrás desde b
        adelante, p = {}, pa
        for k in range(ka + 1, kb):
            p = _flujo(grises[k - 1], grises[k], p, radio) if p is not None else None
            adelante[k] = p
        atras, p = {}, pb
        for k in range(kb - 1, ka, -1):
            p = _flujo(grises[k + 1], grises[k], p, radio) if p is not None else None
            atras[k] = p
        for k in range(ka + 1, kb):
            t = (k - ka) / (kb - ka)
            # Respaldo: interpolación en la imagen estabilizada (cámara compensada)
            interp = _transformar(np.linalg.inv(C[k]), (1 - t) * qa + t * qb)
            f, b = adelante[k], atras[k]
            if f is not None and b is not None and np.linalg.norm(f - b) < 3 * radio + 10:
                pos, fuente = (1 - t) * f + t * b, "flujo"
                con_flujo += 1
            else:
                pos, fuente = interp, "interp"
            resultado[k] = {"pos": pos, "radio": radio, "fuente": fuente}

    rellenos = sum(1 for r in resultado if r and r["fuente"] != "det")
    print(f"Balón en imagen: {len(camino)} detecciones en la trayectoria, "
          f"{len(nodos) - len(camino)} descartadas, {rellenos} fotogramas rellenados "
          f"({con_flujo} con flujo óptico)")
    return resultado


def _en_aro(punto, radio, aros) -> bool:
    for a in aros:
        w = a[2] - a[0]
        if (a[0] - 0.3 * w - radio <= punto[0] <= a[2] + 0.3 * w + radio
                and a[1] - 1.5 * radio <= punto[1] <= a[3] + 1.5 * radio):
            return True
    return False


def _distancia_zona(punto, caja) -> float:
    """0 si el balón está en la zona de manos/cuerpo del jugador; si no, la
    distancia a esa zona en alturas de jugador."""
    w, h = caja[2] - caja[0], caja[3] - caja[1]
    x0, x1 = caja[0] - 0.25 * w, caja[2] + 0.25 * w
    y0, y1 = caja[1] - 0.10 * h, caja[3] - 0.10 * h
    dx = max(x0 - punto[0], 0.0, punto[0] - x1)
    dy = max(y0 - punto[1], 0.0, punto[1] - y1)
    return max(dx, dy) / max(h, 1.0)


# Costes de cambio de poseedor (cuanto mayor, menos probable)
C_COMPANERO = 3.0        # de mano a mano sin pasar por el aire
C_RIVAL = 7.0            # robo directo
C_SOLTAR = 0.8           # pase / tiro / bote largo: el balón queda suelto
C_RECOGER_PROPIO = 0.8   # lo recoge alguien del equipo que lo soltó
C_RECOGER_RIVAL = 2.0    # rebote o intercepción del rival
C_DESVIO = 3.0           # balón suelto tocado por el otro equipo sin control


def _posesion_viterbi(fotogramas, jugadores, imagen) -> tuple[list, list]:
    """Decide quién tiene el balón en todo el clip a la vez (Viterbi).

    Evita que un rival que pasa entre la cámara y el balón "robe" la posesión:
    el balón está dentro de las cajas de los dos, pero cambiar de poseedor a un
    rival es caro y el poseedor real sigue siendo compatible con la imagen."""
    n = len(fotogramas)
    J = len(jugadores)
    S = J + 2                        # + "suelto tras equipo 0" y "suelto tras equipo 1"
    equipo_estado = [j.equipo for j in jugadores] + [0, 1]
    INF = 1e6
    en_aro = [False] * n

    emision = np.zeros((n, S))
    for k, f in enumerate(fotogramas):
        aros = [d["caja"] for d in f["detecciones"] if d["clase"].lower() == "rim"]
        bola = imagen[k]
        if bola is not None:
            en_aro[k] = _en_aro(bola["pos"], bola["radio"], aros)
        alguno_cerca = False
        for s, j in enumerate(jugadores):
            if np.isnan(j.pos[k, 0]):
                emision[k, s] = INF
                continue
            clase_posesion = k in j.clases and "possession" in j.clases[k].lower()
            if bola is None:
                emision[k, s] = 0.8 - (0.8 if clase_posesion else 0.0)
            elif k not in j.cajas:
                # En pista pero sin caja este fotograma (no detectado): neutro
                emision[k, s] = 0.6
            else:
                d = _distancia_zona(bola["pos"], j.cajas[k])
                alguno_cerca |= d == 0
                # La clase "in-possession" del detector solo desempata un poco:
                # se equivoca cuando un rival tapa el balón
                emision[k, s] = 4.0 * d - (0.05 if clase_posesion else 0.0)
                if en_aro[k]:
                    emision[k, s] += 3.0
        suelto = 0.6 if bola is None else (0.0 if en_aro[k] else 1.2 if alguno_cerca else 0.2)
        emision[k, J:] = suelto

    transicion = np.zeros((S, S))
    for a in range(S):
        for b in range(S):
            if a == b:
                continue
            mismo = equipo_estado[a] == equipo_estado[b]
            if a < J and b < J:
                transicion[a, b] = C_COMPANERO if mismo else C_RIVAL
            elif a < J:                       # jugador → suelto
                transicion[a, b] = C_SOLTAR if mismo else C_RIVAL
            elif b < J:                       # suelto → jugador
                transicion[a, b] = C_RECOGER_PROPIO if mismo else C_RECOGER_RIVAL
            else:                             # suelto → suelto del otro equipo
                transicion[a, b] = C_DESVIO

    coste = emision[0].copy()
    atras = np.zeros((n, S), dtype=int)
    for k in range(1, n):
        total = coste[:, None] + transicion
        atras[k] = total.argmin(axis=0)
        coste = total.min(axis=0) + emision[k]
    estados = [int(coste.argmin())]
    for k in range(n - 1, 0, -1):
        estados.append(int(atras[k, estados[-1]]))
    estados.reverse()

    poseedor = [jugadores[s].id if s < J else None for s in estados]
    return poseedor, en_aro


def seguir(ruta_video: Path, paso: int, fotogramas: list[dict], jugadores: list[Jugador],
           movimiento: list, fps: float) -> dict:
    n = len(fotogramas)
    imagen = seguir_en_imagen(ruta_video, paso, fotogramas, jugadores, movimiento)
    poseedor, en_aro = _posesion_viterbi(fotogramas, jugadores, imagen)
    por_id = {j.id: j for j in jugadores}

    pos = np.full((n, 2), np.nan, dtype=np.float32)
    estado = [None] * n
    for k in range(n):
        if poseedor[k] is not None:
            pos[k] = por_id[poseedor[k]].pos[k]
            estado[k] = "posesion"

    # Canasta: la más cercana a la última posición conocida del balón
    ultima = None
    for k in range(n):
        if estado[k] == "posesion":
            ultima = pos[k]
        elif en_aro[k]:
            ref = ultima if ultima is not None else np.array([cancha.LARGO / 2, cancha.ANCHO / 2])
            aro = min((cancha.ARO_IZQ, cancha.ARO_DER), key=lambda a: np.linalg.norm(a - ref))
            pos[k] = aro
            estado[k] = "aro"

    # Interpolación en el aire entre posiciones conocidas (máx. 3 s)
    conocidos = [k for k in range(n) if estado[k] is not None]
    for a, b in zip(conocidos[:-1], conocidos[1:]):
        if 1 < b - a <= fps * 3:
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                pos[k] = (1 - t) * pos[a] + t * pos[b]
                estado[k] = "aire"

    print(f"Balón en pizarra: posesión {estado.count('posesion')} fotogramas, "
          f"aire {estado.count('aire')}, aro {estado.count('aro')}, "
          f"sin dato {estado.count(None)}")
    return {"pos": pos, "estado": estado, "poseedor": poseedor, "imagen": imagen}
