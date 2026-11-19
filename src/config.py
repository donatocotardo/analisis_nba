"""Configuración global del analizador de jugadas."""

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

# --- Rutas ---
DIR_VIDEOS = RAIZ / "videos"
DIR_OUTPUTS = RAIZ / "outputs"
DIR_CACHE = RAIZ / "cache"

# --- Roboflow (API alojada) ---
ROBOFLOW_API_KEY = os.getenv("API_KEY_ROBOFLOW", "")
ROBOFLOW_URL = "https://serverless.roboflow.com"

# Modelos "id/version". Se verifican con src/probar_modelos.py.
MODELO_JUGADORES = "basketball-player-detection-3-ycjdo/4"
MODELO_CANCHA = "basketball-court-detection-2/14"
MODELO_DORSALES = "basketball-jersey-numbers-ocr/3"

# --- Inferencia ---
FPS_ANALISIS = 30          # se submuestrea el vídeo a estos fps
CONF_JUGADORES = 0.30
CONF_KEYPOINT = 0.50
MIN_KEYPOINTS = 4

# --- Cancha NBA (metros) ---
LARGO_CANCHA = 28.65
ANCHO_CANCHA = 15.24


def comprobar_api_key() -> str:
    if not ROBOFLOW_API_KEY:
        raise RuntimeError(
            "Falta API_KEY_ROBOFLOW en el fichero .env de la raíz del proyecto"
        )
    return ROBOFLOW_API_KEY
