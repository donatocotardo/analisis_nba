# Analizador de jugadas NBA → pizarra 2D

## Contexto
Se quiere analizar jugadas cortas (p. ej. `videos/jugada.mp4`: 10 s, 1920x1080, 60 fps, rebote Spurs → contraataque → canasta). El objetivo es seguir a los 10 jugadores en pista y al balón, ignorar público/árbitros/banquillo, separar los dos equipos, identificar a cada jugador (dorsal + nombre) y plasmarlo todo en una pizarra 2D de cancha completa, sincronizada con el vídeo. Debe funcionar con cualquier clip. Se parte de cero (el código anterior se borró a propósito); se mantiene solo `src/extraer_clip.py`.

Decisiones del usuario: modelos preentrenados de Roboflow (API key gratuita), OCR de dorsal + plantilla NBA (nba_api) con corrección manual opcional, salida = vídeo lado a lado.

Entorno: RTX 3050 Ti 4 GB, torch 2.11+cu128, ultralytics, opencv, numpy, matplotlib ya instalados.

## Dependencias nuevas
`inference-gpu` (o `inference`), `supervision`, `nba_api`, `scikit-learn`, `python-dotenv`, `pyyaml`. Crear `requirements.txt`. API key en `.env` (`ROBOFLOW_API_KEY`), añadir `.env` y `cache/` a `.gitignore`.

## Modelos (Roboflow)
- Detección: `basketball-player-detection-3` → clases player (+ variantes en posesión/tiro), ball, referee, number, rim.
- Cancha: `basketball-court-detection-2` → keypoints de la pista.
- Dorsales: modelo OCR de dorsales de Roboflow (`basketball-jersey-numbers-ocr`) sobre el recorte de la clase `number`.
Versiones exactas e IDs de keypoints se fijan en `src/config.py` tras verificarlos en el paso 1.

## Arquitectura (`src/`)
- `config.py` — rutas, IDs de modelos, umbrales, dimensiones NBA (28,65 × 15,24 m), colores.
- `cancha.py` — geometría de la pista NBA en metros (líneas, zona, triple, aros) y coordenadas reales de cada keypoint del modelo. Cálculo de homografía por fotograma (`cv2.findHomography` RANSAC con keypoints conf > 0.5, mínimo 4), suavizado temporal y reutilización de la última válida si falla.
- `deteccion.py` — llamada a los modelos por fotograma; separa jugadores, balón, árbitros, dorsales.
- `seguimiento.py` — ByteTrack (`supervision`) sobre jugadores; filtro “en pista”: se proyecta el punto de pies (centro inferior de la caja) y se descarta si cae fuera del polígono de la pista (+ margen) o si es árbitro. Fusión de tracks rotos por proximidad 2D + equipo + dorsal; límite de 5 por equipo. Suavizado de trayectorias (media móvil / Savitzky-Golay) e interpolación de huecos cortos.
- `balon.py` — selección de la detección de balón más probable, interpolación de huecos; posición en pizarra = jugador en posesión si existe clase “in-possession”/solape, si no la proyección del balón (marcado como aproximado cuando está en el aire).
- `equipo.py` — recorte del torso de cada track, histograma HSV/Lab → KMeans k=2 sobre todo el clip, voto mayoritario por track. Se nombra cada cluster cruzando los dorsales leídos con las plantillas.
- `dorsales.py` — OCR por recorte `number` asociado a la caja del jugador; voto ponderado por confianza a lo largo del track.
- `plantillas.py` — `nba_api` (`commonteamroster`) para los dos equipos y temporada, cacheado en `cache/plantillas/*.json`; mapeo dorsal → nombre.
- `pizarra.py` — dibuja la cancha completa (OpenCV), jugadores como círculos con color de equipo + dorsal, nombre corto debajo, balón, estela de los últimos ~1,5 s. Compone vídeo original anotado (cajas, ID, dorsal) + pizarra lado a lado.
- `pipeline.py` — dos pasadas: (1) inferencia + tracking + homografías, guardado en `outputs/<jugada>/datos.json` (cachea para no repetir inferencia); (2) equipos, dorsales, nombres, suavizado y render.
- `main.py` (raíz) — CLI:
  `python main.py videos/jugada.mp4 --local SAS --visitante GSW [--temporada 2026-27] [--reprocesar]`
- Correcciones opcionales: `outputs/<jugada>/correcciones.yaml` (track_id → dorsal/equipo, o descartar) que se aplica en la pasada 2 sin volver a inferir.

## Salidas
`outputs/<jugada>/pizarra.mp4` (lado a lado, mismos fps) y `outputs/<jugada>/datos.json` (posiciones por fotograma en metros, equipo, dorsal, nombre — base para el análisis posterior de "qué fue clave").

## Orden de implementación
1. requirements, `.env`, `config.py`; script de prueba que carga los modelos de Roboflow y muestra clases/keypoints en un fotograma de `jugada.mp4`.
2. `cancha.py` + verificación de homografía (dibujar la pista proyectada sobre el vídeo).
3. Detección + tracking + filtro en pista.
4. Balón.
5. Equipos y dorsales + plantillas.
6. Pizarra y vídeo lado a lado; `main.py`.

## Verificación
- Paso 2: vídeo de depuración con las líneas de la pista superpuestas — deben coincidir con las reales durante el paneo de cámara.
- `python main.py videos/jugada.mp4 --local SAS --visitante GSW` → revisar `pizarra.mp4`: 10 jugadores máx., 5 por color, sin árbitros ni público, contraataque de los Spurs visible de una canasta a otra, nombres coherentes con el vídeo.
- Probar con otro clip extraído con `src/extraer_clip.py` para confirmar que no hay nada específico de `jugada.mp4`.
- Limitaciones a comunicar: jugadores fuera de plano no aparecen (la cámara solo ve parte de la pista); balón en el aire es aproximado; con 4 GB de VRAM la inferencia irá a pocos fps (por eso se cachea).
