"""Cliente mínimo de la API alojada de Roboflow (sin inference-sdk)."""

import base64
import time

import cv2
import numpy as np
import requests

from src import config

_sesion = requests.Session()


def _codificar(imagen: np.ndarray, calidad: int = 90) -> str:
    ok, buffer = cv2.imencode(".jpg", imagen, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def codificar(imagen: np.ndarray) -> str:
    return _codificar(imagen)


def inferir(modelo: str, imagen: np.ndarray | str, confianza: float | None = None,
            reintentos: int = 4, timeout: float = 60) -> dict:
    """Envía una imagen (array o JPEG en base64) a un modelo "id/version"."""
    params = {"api_key": config.comprobar_api_key()}
    if confianza is not None:
        params["confidence"] = confianza

    cuerpo = imagen if isinstance(imagen, str) else _codificar(imagen)
    url = f"{config.ROBOFLOW_URL}/{modelo}"

    for intento in range(1, reintentos + 1):
        try:
            respuesta = _sesion.post(
                url,
                params=params,
                data=cuerpo,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=timeout,
            )
        except requests.RequestException as error:
            if intento == reintentos:
                raise RuntimeError(f"Error de red con {modelo}: {error}") from error
            time.sleep(2 * intento)
            continue

        if respuesta.status_code == 200:
            return respuesta.json()
        if respuesta.status_code in (429, 500, 502, 503, 504) and intento < reintentos:
            time.sleep(2 * intento)
            continue
        # No incluir la URL (lleva la API key) en el mensaje
        raise RuntimeError(
            f"Roboflow respondió {respuesta.status_code} para {modelo}: "
            f"{respuesta.text[:300]}"
        )

    raise RuntimeError(f"Sin respuesta de {modelo}")
