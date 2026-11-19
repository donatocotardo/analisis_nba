"""Movimiento de cámara entre fotogramas (flujo óptico + homografía).

Las cámaras de retransmisión giran y hacen zoom desde un punto fijo, así que el
movimiento entre dos fotogramas es una homografía de imagen. Se estima con
puntos del fondo (se excluyen jugadores y gráficos fijos como el marcador) y
sirve para predecir la homografía de la pista cuando los keypoints fallan o son
ambiguos (por ejemplo en el centro de la pista).
"""

from pathlib import Path

import cv2
import numpy as np

from src.recortes import leer_fotogramas

ANCHO_TRABAJO = 640


def _mascara_estatica(grises: list[np.ndarray]) -> np.ndarray:
    """Píxeles que no cambian en todo el clip (marcador, logos) → excluidos."""
    muestra = np.stack(grises[:: max(1, len(grises) // 40)]).astype(np.float32)
    return muestra.std(axis=0) < 6.0


def calcular(ruta_video: Path, paso: int, fotogramas: list[dict], dir_salida: Path,
             reprocesar: bool = False) -> list[np.ndarray | None]:
    """Devuelve M[k]: homografía que lleva píxeles del fotograma k al k-1
    (coordenadas a resolución completa). M[0] es la identidad."""
    ruta_cache = dir_salida / "movimiento_camara.npz"
    if ruta_cache.exists() and not reprocesar:
        datos = np.load(ruta_cache)
        if len(datos["M"]) == len(fotogramas):
            return [None if not v else m for m, v in zip(datos["M"], datos["valido"])]

    grises, escala = [], None
    for k, frame in leer_fotogramas(ruta_video, paso):
        if k >= len(fotogramas):
            break
        escala = ANCHO_TRABAJO / frame.shape[1]
        pequeno = cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        grises.append(cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY))

    estatico = _mascara_estatica(grises)
    S = np.diag([escala, escala, 1.0])
    S_inv = np.linalg.inv(S)

    M = [np.eye(3)]
    for k in range(1, len(grises)):
        previo, actual = grises[k - 1], grises[k]
        mascara = np.where(estatico, 0, 255).astype(np.uint8)
        for d in fotogramas[k - 1]["detecciones"]:
            x1, y1, x2, y2 = (int(v * escala) for v in d["caja"])
            mascara[max(0, y1):y2, max(0, x1):x2] = 0

        p0 = cv2.goodFeaturesToTrack(previo, maxCorners=500, qualityLevel=0.01,
                                     minDistance=8, mask=mascara)
        if p0 is None or len(p0) < 12:
            M.append(None)
            continue
        p1, estado, _ = cv2.calcOpticalFlowPyrLK(previo, actual, p0, None,
                                                 winSize=(21, 21), maxLevel=3)
        ok = estado.ravel() == 1
        if ok.sum() < 12:
            M.append(None)
            continue
        H_small, inliers = cv2.findHomography(p1[ok], p0[ok], cv2.RANSAC, 2.0)
        if H_small is None or inliers.sum() < 10:
            M.append(None)
            continue
        M.append(S_inv @ H_small @ S)

    validos = np.array([m is not None for m in M])
    np.savez(ruta_cache, M=np.array([m if m is not None else np.eye(3) for m in M]),
             valido=validos)
    print(f"Movimiento de cámara estimado en {validos.sum()}/{len(M)} fotogramas")
    return M
