"""Pasada 1: inferencia (jugadores, balón, dorsales, keypoints) con caché en disco."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from src import config
from src.roboflow_api import codificar, inferir

VERSION_CACHE = 1


def _caja(p: dict) -> list[float]:
    return [p["x"] - p["width"] / 2, p["y"] - p["height"] / 2,
            p["x"] + p["width"] / 2, p["y"] + p["height"] / 2]


def _procesar(indice: int, jpg_b64: str) -> dict:
    det = inferir(config.MODELO_JUGADORES, jpg_b64, confianza=config.CONF_JUGADORES)
    cancha = inferir(config.MODELO_CANCHA, jpg_b64)

    detecciones = [
        {"clase": p["class"], "conf": round(p["confidence"], 3),
         "caja": [round(v, 1) for v in _caja(p)]}
        for p in det.get("predictions", [])
    ]
    keypoints = []
    preds = cancha.get("predictions", [])
    if preds:
        mejor = max(preds, key=lambda p: p.get("confidence", 0))
        keypoints = [
            {"class_id": k["class_id"], "x": round(k["x"], 1), "y": round(k["y"], 1),
             "confidence": round(k["confidence"], 3)}
            for k in mejor.get("keypoints", [])
        ]
    return {"indice": indice, "detecciones": detecciones, "keypoints": keypoints}


def ejecutar(ruta_video: Path, dir_salida: Path, fps_analisis: float,
             reprocesar: bool = False, hilos: int = 6) -> dict:
    ruta_cache = dir_salida / "inferencia.json"
    if ruta_cache.exists() and not reprocesar:
        datos = json.loads(ruta_cache.read_text())
        if datos.get("version") == VERSION_CACHE and datos.get("fps_analisis") == fps_analisis:
            print(f"Usando inferencia en caché: {ruta_cache}")
            return datos

    video = cv2.VideoCapture(str(ruta_video))
    if not video.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {ruta_video}")
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("FPS no válidos")
    ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    paso = max(1, round(fps / fps_analisis))

    # Se codifican los fotogramas muestreados en JPEG (poca memoria)
    pendientes = []
    indice = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if indice % paso == 0:
            pendientes.append((indice, codificar(frame)))
        indice += 1
    video.release()

    total = len(pendientes)
    print(f"Vídeo: {indice} fotogramas a {fps:.1f} fps → {total} a analizar "
          f"(1 de cada {paso}). Enviando a Roboflow...")

    resultados = {}
    with ThreadPoolExecutor(max_workers=hilos) as pool:
        futuros = [pool.submit(_procesar, i, b64) for i, b64 in pendientes]
        for n, futuro in enumerate(as_completed(futuros), 1):
            r = futuro.result()
            resultados[r["indice"]] = r
            if n % 25 == 0 or n == total:
                print(f"  {n}/{total} fotogramas inferidos")

    datos = {
        "version": VERSION_CACHE,
        "video": str(ruta_video),
        "fps_video": fps,
        "fps_analisis": fps_analisis,
        "paso": paso,
        "ancho": ancho,
        "alto": alto,
        "fotogramas": [resultados[i] for i, _ in pendientes],
    }
    dir_salida.mkdir(parents=True, exist_ok=True)
    ruta_cache.write_text(json.dumps(datos))
    print(f"Inferencia guardada en {ruta_cache}")
    return datos
