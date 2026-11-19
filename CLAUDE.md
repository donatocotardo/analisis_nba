# Analizador de jugadas NBA

Analiza clips cortos de NBA: sigue jugadores en pista y balón, separa equipos, identifica dorsal + nombre y genera una pizarra 2D de cancha completa (vídeo lado a lado).

## Uso
```
.venv\Scripts\python.exe main.py videos/jugada.mp4 --local SAS --visitante GSW
```
Salidas en `outputs/<clip>/`: `pizarra.mp4`, `datos.json` (posiciones en metros), `correcciones.yaml` (editable), cachés `inferencia.json` y `movimiento_camara.npz`. `--reprocesar` repite la inferencia.

## Entorno
- Windows, Python 3.12 en `.venv` (usar siempre `.venv\Scripts\python.exe`).
- GPU RTX 3050 Ti (4 GB), torch 2.11+cu128. **No instalar paquetes que reemplacen torch/opencv/numpy** (`inference-gpu` e `inference-sdk` lo hacen). Hacer `pip install --dry-run` antes de añadir dependencias. EasyOCR va con `--no-deps` (ver requirements.txt).
- Roboflow por HTTP directo (`src/roboflow_api.py`, API alojada `serverless.roboflow.com`). Clave en `.env` como `API_KEY_ROBOFLOW` (no leer ni imprimir `.env`).
- El modelo `basketball-jersey-numbers-ocr` no funciona en la API alojada ("Service misconfiguration"); los dorsales se leen con EasyOCR local.

## Pipeline (`src/pipeline.py`)
1. `inferencia.py` — jugadores/balón/árbitros/dorsales (`basketball-player-detection-3-ycjdo/4`) y 33 keypoints de pista (`basketball-court-detection-2/14`), cacheado.
2. `camara.py` — movimiento de cámara entre fotogramas (flujo óptico, excluye jugadores y gráficos fijos).
3. `seguimiento.py` — homografía fusionando keypoints + predicción por movimiento de cámara (los keypoints confunden las mitades en el centro de la pista) y suavizada en el tiempo compensando la cámara; NMS entre clases de jugador; ByteTrack (`trackers`); sin pie si la caja toca el borde; descarte de árbitros y de quien está fuera de pista.
   `lineas.py` — refina esa homografía con las líneas pintadas reales (top-hat + Hough, agrupando trozos colineales que cortan los jugadores; umbral bajo porque las líneas lejanas son finas). Corrige el desplazamiento de la banda lejana. Luego se vuelve a suavizar.
4. `equipo.py` — color del torso por fotograma (KMeans k=2) y partición de tracks cuando el tracker intercambia jugadores de distinto equipo.
5. `dorsales.py` (EasyOCR; solo números en la mitad superior de la caja — abajo el detector confunde el logo del pantalón; votación con la plantilla del equipo), `plantillas.py` (nba_api, caché editable en `cache/plantillas/`).
6. `identidad.py` — nombra equipos, aplica correcciones, une tracks rotos, dorsales únicos, suavizado (límite 9.5 m/s, Savitzky-Golay 0.8 s, dentro de pista).
7. `balon.py` — trayectoria del balón en imagen por camino de coste mínimo sobre todas las detecciones (cámara compensada; descarta falsos de la grada), huecos con flujo óptico; posesión por geometría (la clase "in-possession" solo si no se ve el balón) / aire / aro.
8. `pizarra.py` — render; `cancha.py` — geometría NBA (class_id k del modelo = vértice k de roboflow/sports rama feat/basketball).

## Convenciones
- Código, comentarios y mensajes en español.
- Coordenadas de pista en metros: x a lo largo (0–28.65), y a lo ancho (0–15.24).
- Diagnóstico de modelos: `python -m src.probar_modelos videos/<clip>.mp4 <segundo>`.
- Plan de trabajo: `.claude/plans/`.
