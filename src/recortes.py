"""Extrae recortes de torso (todas las observaciones) y de dorsal en una sola
lectura del vídeo."""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.seguimiento import Track, _iou


def leer_fotogramas(ruta_video: Path, paso: int):
    """Genera (k, frame) para los fotogramas analizados."""
    video = cv2.VideoCapture(str(ruta_video))
    indice, k = 0, 0
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if indice % paso == 0:
                yield k, frame
                k += 1
            indice += 1
    finally:
        video.release()


def _ocluidos(tracks: dict[int, Track]) -> set[tuple[int, int]]:
    """(track_id, k) donde la caja del jugador se solapa con la de otro."""
    por_frame = defaultdict(list)
    for t in tracks.values():
        for o in t.obs:
            por_frame[o.k].append((t.id, o.caja))
    resultado = set()
    for k, cajas in por_frame.items():
        for i, (ta, ca) in enumerate(cajas):
            for tb, cb in cajas[i + 1:]:
                if _iou(ca, cb) > 0.1:
                    resultado.add((ta, k))
                    resultado.add((tb, k))
    return resultado


def _asignar_numeros(fotogramas: list[dict], tracks: dict[int, Track]):
    """Asocia cada detección "number" a la única caja de jugador que la contiene."""
    cajas_por_frame = defaultdict(list)
    for t in tracks.values():
        for o in t.obs:
            cajas_por_frame[o.k].append((t.id, o.caja))

    por_frame = defaultdict(list)       # k -> [(track_id, caja_numero, peso)]
    for k, f in enumerate(fotogramas):
        for d in f["detecciones"]:
            if "number" not in d["clase"].lower():
                continue
            x1, y1, x2, y2 = d["caja"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dentro = [(tid, c) for tid, c in cajas_por_frame[k]
                      if c[0] <= cx <= c[2] and c[1] <= cy <= c[3]]
            if len(dentro) != 1:
                continue
            tid, c = dentro[0]
            # El dorsal está en el pecho/espalda (parte superior de la caja).
            # Más abajo suele ser el logo del pantalón, que el detector confunde.
            relativo = (cy - c[1]) / max(1.0, c[3] - c[1])
            if 0.08 <= relativo <= 0.50:
                por_frame[k].append((tid, np.array(d["caja"]), d["conf"] * (y2 - y1)))
    return por_frame


def _recortar(frame, caja, margen=0):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = caja
    x1, y1 = max(0, int(x1) - margen), max(0, int(y1) - margen)
    x2, y2 = min(w, int(x2) + margen), min(h, int(y2) + margen)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2].copy()


def _torso(frame, caja):
    x1, y1, x2, y2 = caja
    ancho, alto = x2 - x1, y2 - y1
    return _recortar(frame, (x1 + 0.25 * ancho, y1 + 0.15 * alto,
                             x2 - 0.25 * ancho, y1 + 0.50 * alto))


def recopilar(ruta_video: Path, paso: int, fotogramas: list[dict],
              tracks: dict[int, Track]):
    """Devuelve:
        torsos:   {(track_id, k): recorte}
        ocluidos: {(track_id, k)}
        numeros:  [(track_id, k, peso, recorte)]
    """
    cajas = defaultdict(dict)
    for t in tracks.values():
        for o in t.obs:
            cajas[o.k][t.id] = o.caja
    numeros_por_frame = _asignar_numeros(fotogramas, tracks)

    torsos, numeros = {}, []
    for k, frame in leer_fotogramas(ruta_video, paso):
        for tid, caja in cajas.get(k, {}).items():
            r = _torso(frame, caja)
            if r is not None:
                torsos[(tid, k)] = r
        for tid, caja, peso in numeros_por_frame.get(k, []):
            r = _recortar(frame, caja, margen=3)
            if r is not None:
                numeros.append((tid, k, peso, r))
    return torsos, _ocluidos(tracks), numeros
