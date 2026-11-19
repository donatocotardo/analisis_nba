
import cv2
from pathlib import Path

VIDEO_PATH = Path("videos/Philadelphia 76ers vs Orlando Magic Full Game Highlights - November 18, 2026 _ NBA Season.mp4")
OUTPUT_PATH = Path("videos/jugada_76ers.mp4")

# Segundos del vídeo original
START_TIME = 125
END_TIME = 131

video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise RuntimeError("No se pudo abrir el vídeo")

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    video.release()
    raise RuntimeError("FPS no válidos")

start_frame = round(START_TIME * fps)
end_frame = min(round(END_TIME * fps), total_frames)

if start_frame < 0 or end_frame <= start_frame:
    video.release()
    raise ValueError("Intervalo de tiempo no válido")

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

video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

for _ in range(start_frame, end_frame):
    ret, frame = video.read()

    if not ret:
        break

    output.write(frame)

video.release()
output.release()

print(f"Clip guardado en: {OUTPUT_PATH}")
print(f"Intervalo: {START_TIME}s - {END_TIME}s")
