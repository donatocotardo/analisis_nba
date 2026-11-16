
import cv2
import time

#video = cv2.VideoCapture("videos/jugada.mp4")
video = cv2.VideoCapture("outputs/deteccion.mp4")


if not video.isOpened():
    raise RuntimeError("No se pudo abrir el vídeo")

fps = video.get(cv2.CAP_PROP_FPS)
frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

print(f"FPS: {fps}")
print(f"Fotogramas: {int(frames)}")

if fps <= 0:
    video.release()
    raise RuntimeError("No se pudo obtener una frecuencia de fotogramas válida")

print(f"Duración: {frames / fps:.2f} segundos")

window_name = "NBA Play Analyzer"
playback_started = time.perf_counter()
frame_index = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    cv2.imshow(window_name, frame)
    frame_index += 1

    next_frame_time = playback_started + frame_index / fps
    remaining_ms = max(1, round((next_frame_time - time.perf_counter()) * 1000))
    key = cv2.waitKey(remaining_ms) & 0xFF

    if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

video.release()
cv2.destroyAllWindows()
