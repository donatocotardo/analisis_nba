"""Paso 1: prueba los modelos de Roboflow sobre un fotograma y muestra qué devuelven.

Uso:
    python -m src.probar_modelos videos/jugada.mp4 [segundo]
"""

import json
import sys
from collections import Counter

import cv2

from src import config
from src.roboflow_api import inferir


def leer_fotograma(ruta: str, segundo: float):
    video = cv2.VideoCapture(ruta)
    if not video.isOpened():
        raise RuntimeError(f"No se pudo abrir {ruta}")
    video.set(cv2.CAP_PROP_POS_MSEC, segundo * 1000)
    ok, frame = video.read()
    video.release()
    if not ok:
        raise RuntimeError("No se pudo leer el fotograma")
    return frame


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else "videos/jugada.mp4"
    segundo = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    frame = leer_fotograma(ruta, segundo)
    salida = config.DIR_OUTPUTS / "prueba_modelos"
    salida.mkdir(parents=True, exist_ok=True)

    # --- Jugadores / balón / árbitros / dorsales ---
    print(f"\n== {config.MODELO_JUGADORES}")
    det = inferir(config.MODELO_JUGADORES, frame, confianza=config.CONF_JUGADORES)
    preds = det.get("predictions", [])
    print("Clases:", dict(Counter(p["class"] for p in preds)))
    (salida / "jugadores.json").write_text(json.dumps(det, indent=2))

    vis = frame.copy()
    for p in preds:
        x1, y1 = int(p["x"] - p["width"] / 2), int(p["y"] - p["height"] / 2)
        x2, y2 = int(p["x"] + p["width"] / 2), int(p["y"] + p["height"] / 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f'{p["class"]} {p["confidence"]:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- Keypoints de cancha ---
    print(f"\n== {config.MODELO_CANCHA}")
    cancha = inferir(config.MODELO_CANCHA, frame)
    (salida / "cancha.json").write_text(json.dumps(cancha, indent=2))
    for p in cancha.get("predictions", []):
        kps = p.get("keypoints", [])
        buenos = [k for k in kps if k["confidence"] >= config.CONF_KEYPOINT]
        print(f"Keypoints: {len(kps)} totales, {len(buenos)} con conf >= {config.CONF_KEYPOINT}")
        for k in buenos:
            centro = (int(k["x"]), int(k["y"]))
            cv2.circle(vis, centro, 6, (0, 0, 255), -1)
            cv2.putText(vis, str(k.get("class", k.get("class_id"))), centro,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- OCR de dorsales sobre el primer recorte "number" ---
    numeros = [p for p in preds if "number" in p["class"].lower()]
    if numeros:
        p = max(numeros, key=lambda d: d["confidence"])
        x1, y1 = max(0, int(p["x"] - p["width"] / 2)), max(0, int(p["y"] - p["height"] / 2))
        x2, y2 = int(p["x"] + p["width"] / 2), int(p["y"] + p["height"] / 2)
        recorte = frame[y1:y2, x1:x2]
        print(f"\n== {config.MODELO_DORSALES}")
        try:
            ocr = inferir(config.MODELO_DORSALES, recorte)
            print(json.dumps(ocr, indent=2)[:800])
        except RuntimeError as error:
            print("OCR falló:", error)
    else:
        print("\nNo se detectó ningún dorsal en este fotograma")

    cv2.imwrite(str(salida / "prueba.jpg"), vis)
    print(f"\nResultado visual: {salida / 'prueba.jpg'}")


if __name__ == "__main__":
    main()
