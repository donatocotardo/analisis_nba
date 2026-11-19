# Análisis de jugadas NBA → pizarra 2D

Convierte un clip corto de una retransmisión de la NBA en una **pizarra táctica 2D de la pista completa**. Sigue a los 10 jugadores y al balón, separa los dos equipos, identifica a cada jugador por su dorsal y su nombre, y genera un vídeo con la imagen original y la pizarra sincronizadas, una al lado de la otra.

## Ejemplo

<video src="ejemplo.mp4" controls width="100%"></video>

Si el vídeo no se reproduce arriba, ábrelo directamente: [ejemplo.mp4](ejemplo.mp4)

En el vídeo:
- **Izquierda:** el vídeo original. Cada jugador tiene una caja del color de su equipo con su dorsal y su nombre. El balón va marcado con un círculo naranja.
- **Derecha:** la pista completa vista desde arriba. Cada jugador es un círculo con su dorsal y una estela de su último segundo y medio de movimiento.
- **Arriba:** el tiempo y quién tiene el balón (o si está en el aire o en el aro).

## Qué hace

1. **Detecta** jugadores, balón, árbitros, dorsales y puntos de referencia de la pista con modelos de [Roboflow](https://roboflow.com).
2. **Sitúa cada jugador en la pista (homografía):**
   - Combina los puntos de referencia con el movimiento de la cámara, estimado con flujo óptico.
   - Afina el resultado con las líneas reales pintadas en el suelo.
   - Suaviza en el tiempo para que los paneos no hagan temblar las posiciones.
3. **Sigue a los jugadores:**
   - Descarta árbitros, banquillo y público.
   - Separa los equipos por el color de la camiseta y detecta cuándo el seguimiento cambia de jugador en un cruce.
   - Mantiene la misma identidad durante todo el clip, con 5 jugadores por equipo.
4. **Lee los dorsales** con EasyOCR y elige el resultado más probable comparándolo con la plantilla oficial del equipo (vía `nba_api`).
5. **Sigue el balón:**
   - Elige la trayectoria más coherente entre todas las detecciones del clip, de forma que algo de la grada no se confunda con el balón.
   - Rellena los huecos cortos con flujo óptico.
   - Decide quién tiene la posesión considerando toda la jugada a la vez, para que un rival que pasa entre la cámara y el balón no "se lo quede".
6. **Genera** el vídeo con la pizarra y un JSON con todas las posiciones en metros.

## Requisitos

- Windows o Linux con Python 3.12.
- GPU NVIDIA recomendada (EasyOCR la usa; sin GPU funciona, pero más lento).
- Cuenta gratuita de Roboflow y su API key.
- Conexión a internet: hace falta para Roboflow y para descargar las plantillas de la NBA.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate

# torch con CUDA (ajusta la versión de CUDA a tu equipo)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

# EasyOCR sin dependencias, para que no instale opencv-python-headless
pip install scikit-image python-bidi shapely pyclipper ninja
pip install --no-deps easyocr
```

> No instales `inference`, `inference-gpu` ni `inference-sdk`: sustituyen torch por una versión sin CUDA y bajan las versiones de numpy y opencv. El proyecto llama a la API de Roboflow directamente por HTTP.

Crea un fichero `.env` en la raíz con tu clave:

```
API_KEY_ROBOFLOW=tu_clave
```

## Uso

### 1. (Opcional) Recortar la jugada de un partido completo

Edita las rutas y los segundos de inicio y fin en [src/extraer_clip.py](src/extraer_clip.py) y ejecuta:

```bash
python src/extraer_clip.py
```

Funciona mejor con jugadas de 5 a 15 segundos grabadas con la cámara principal de la retransmisión (plano general lateral).

### 2. Analizar

```bash
python main.py videos/mi_jugada.mp4 --local SAS --visitante GSW
```

| Opción | Descripción |
|---|---|
| `--local`, `--visitante` | Abreviaturas NBA de los equipos (SAS, GSW, LAL, BOS, PHI, ORL...) |
| `--temporada 2025-26` | Temporada de las plantillas (por defecto, la actual) |
| `--fps 30` | Fotogramas por segundo que se analizan |
| `--reprocesar` | Repite la inferencia de Roboflow aunque haya caché |

La primera ejecución envía los fotogramas a Roboflow (unas 600 llamadas para un clip de 10 s). Después se usa la caché y cada ejecución tarda menos de un minuto.

### 3. Resultados

Todo se guarda en `outputs/<nombre_del_clip>/`:

| Fichero | Contenido |
|---|---|
| `pizarra.mp4` | Vídeo original y pizarra lado a lado |
| `datos.json` | Posiciones de jugadores y balón por fotograma (en metros), equipos, dorsales, nombres y posesión |
| `correcciones.yaml` | Correcciones manuales (ver abajo) |
| `inferencia.json`, `movimiento_camara.npz` | Cachés |

Coordenadas en `datos.json`: `x` va a lo largo de la pista (0–28,65 m) e `y` a lo ancho (0–15,24 m).

### 4. Corregir errores a mano

Si algún jugador sale mal identificado, edita `outputs/<clip>/correcciones.yaml` y vuelve a ejecutar el mismo comando. No se repite la inferencia, así que no gasta créditos de Roboflow. Al final del fichero está la lista de tracks detectados, con sus IDs y tiempos.

```yaml
tracks:
  9: {dorsal: 23}                                  # dorsal correcto
  5: {equipo: PHI, dorsal: 11, nombre: "Justin Edwards"}
  15: {descartar: true}                            # no es un jugador en pista
  3: {cortar: 2.4}                                 # el track cambia de jugador en el segundo 2.4
equipos:                                           # si los equipos salen al revés
  0: PHI
  1: ORL
```

Las plantillas descargadas se guardan en `cache/plantillas/` y se pueden editar si algún jugador ha cambiado de dorsal.

## Estructura

```
main.py                 CLI
src/
  pipeline.py           orquesta todo el análisis
  inferencia.py         llamadas a Roboflow (con caché)
  roboflow_api.py       cliente HTTP de Roboflow
  camara.py             movimiento de cámara (flujo óptico)
  cancha.py             geometría NBA, homografía y dibujo de la pista
  lineas.py             refinado de la homografía con las líneas del suelo
  seguimiento.py        homografías por fotograma y seguimiento (ByteTrack)
  recortes.py           recortes de torso y dorsal
  equipo.py             separación de equipos por color
  dorsales.py           OCR de dorsales y votación con la plantilla
  plantillas.py         plantillas NBA (nba_api)
  identidad.py          identidades finales, unión de fragmentos y suavizado
  balon.py              trayectoria del balón y posesión
  pizarra.py            render del vídeo final
  extraer_clip.py       recorta una jugada de un vídeo largo
  probar_modelos.py     diagnóstico de los modelos sobre un fotograma
```

## Limitaciones

- Un jugador que no sale en plano no aparece en la pizarra mientras no se le ve. Si está tapado por otro jugador, se estima su posición siguiendo al que lo tapa durante un máximo de 3 s.
- Si un jugador salta haciendo un tapón o tirando, la pizarra refleja que se mueve en el eje vertical, cuando en realidad en un plano 2D no debería moverse.
- La posición del balón en el aire (pases y tiros) es una aproximación entre quién lo suelta y quién lo recibe o el aro.
- No funciona bien con repeticiones, primeros planos ni la cámara detrás de la canasta.
- Los dorsales se leen a partir de píxeles pequeños. Cuando un jugador sale de espaldas o lejos de la cámara puede quedar sin identificar y hay que corregirlo a mano.
