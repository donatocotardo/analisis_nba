
from ultralytics import YOLO
import cv2
from pathlib import Path
import time

start_time = time.perf_counter()
VIDEO_PATH = Path("videos/jugada.mp4")
OUTPUT_PATH = Path("outputs/deteccion.mp4")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Modelo preentrenado de detección
model = YOLO("yolo11m.pt")

video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise RuntimeError("No se pudo abrir el clip")

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    video.release()
    raise RuntimeError("FPS no válidos")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)

if not output.isOpened():
    video.release()
    raise RuntimeError("No se pudo crear el vídeo de salida")

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Detectamos únicamente personas (clase 0 de COCO)
    results = model.predict(
        frame,
        classes=[0],
        conf=0.20,
        imgsz=960,
        iou=0.6,
        device=0,
        verbose=False
    )

    annotated_frame = results[0].plot()

    output.write(annotated_frame)

video.release()
output.release()

print(f"Vídeo procesado: {OUTPUT_PATH}")
print(f"Tiempo total de ejecución: {time.perf_counter() - start_time:.2f} segundos")
