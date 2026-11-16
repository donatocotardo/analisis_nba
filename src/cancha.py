
from ultralytics import YOLO
import cv2
from pathlib import Path
import time

start_time = time.perf_counter()

VIDEO_PATH = Path("videos/jugada.mp4")
MODEL_PATH = Path("models/court_keypoint_detector.pt")
OUTPUT_PATH = Path("outputs/cancha_detectada.mp4")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Configuración
DEVICE = 0
IMAGE_SIZE = 960
CONFIDENCE = 0.25

# Para la primera prueba solo procesaremos 5 segundos
MAX_SECONDS = 5

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encuentra el modelo: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print(f"Modelo cargado: {MODEL_PATH}")
print(f"Tipo de modelo: {model.task}")

video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise RuntimeError("No se pudo abrir el vídeo")

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    video.release()
    raise RuntimeError("FPS no válidos")

max_frames = int(MAX_SECONDS * fps)

output = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not output.isOpened():
    video.release()
    raise RuntimeError("No se pudo crear el vídeo de salida")


# detectar puntos de la cancha

frame_index = 0

try:
    while frame_index < max_frames:

        ret, frame = video.read()

        if not ret:
            break

        result = model.predict(
            frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )[0]

        # Visualización de las predicciones del modelo
        annotated_frame = result.plot()

        # Diagnóstico de keypoints
        keypoint_count = 0

        if result.keypoints is not None:
            keypoints = result.keypoints.xy

            if len(keypoints) > 0:
                keypoint_count = len(keypoints[0])

        cv2.putText(
            annotated_frame,
            f"Keypoints: {keypoint_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        output.write(annotated_frame)

        frame_index += 1

        if frame_index % 30 == 0:
            print(f"Fotogramas procesados: {frame_index}")

finally:
    video.release()
    output.release()
    cv2.destroyAllWindows()

print(f"Vídeo procesado: {OUTPUT_PATH}")
print(
    f"Tiempo total: "
    f"{time.perf_counter() - start_time:.2f} segundos"
)
