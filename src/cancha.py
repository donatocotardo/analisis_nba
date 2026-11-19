"""Geometría de la cancha NBA, homografía vídeo → pista y dibujo de la pizarra.

Sistema de coordenadas de la pista (metros):
    x: a lo largo (0 = línea de fondo izquierda, 28.65 = línea de fondo derecha)
    y: a lo ancho (0 = banda superior, 15.24 = banda inferior)

Los 33 keypoints del modelo `basketball-court-detection-2` (class_id 0..32) se
corresponden, en este orden, con los vértices de la configuración NBA del
repositorio roboflow/sports (rama feat/basketball, sports/basketball/config.py).
"""

import cv2
import numpy as np

from src import config

# --- Medidas NBA en centímetros (preset de roboflow/sports) ---
_ANCHO = 1524
_LARGO = 2865
_RADIO_TRIPLE = 724
_TRIPLE_RECTO = 424          # longitud del tramo recto del triple desde el fondo
_BANDA_A_TRIPLE = 91
_ANCHO_ZONA = 488
_LARGO_ZONA = 579
_RADIO_CIRCULO = 183
_RADIO_RESTRINGIDA = 122
_FONDO_A_ARO = 160
_FONDO_A_SAQUE = 835


def _vertices_cm():
    zona = (_ANCHO - _ANCHO_ZONA) // 2
    medio = _ANCHO // 2
    L, W = _LARGO, _ANCHO
    return [
        (0, 0), (0, _BANDA_A_TRIPLE), (0, zona), (0, zona + _ANCHO_ZONA),
        (0, W - _BANDA_A_TRIPLE), (0, W),
        (_FONDO_A_ARO, medio),
        (_TRIPLE_RECTO, _BANDA_A_TRIPLE), (_TRIPLE_RECTO, W - _BANDA_A_TRIPLE),
        (_LARGO_ZONA, zona), (_LARGO_ZONA, zona + _ANCHO_ZONA // 2),
        (_LARGO_ZONA, zona + _ANCHO_ZONA),
        (_FONDO_A_SAQUE, 0), (_FONDO_A_ARO + _RADIO_TRIPLE, medio), (_FONDO_A_SAQUE, W),
        (L // 2, 0), (L // 2, medio), (L // 2, W),
        (L - _FONDO_A_SAQUE, 0), (L - _FONDO_A_ARO - _RADIO_TRIPLE, medio),
        (L - _FONDO_A_SAQUE, W),
        (L - _LARGO_ZONA, zona), (L - _LARGO_ZONA, zona + _ANCHO_ZONA // 2),
        (L - _LARGO_ZONA, zona + _ANCHO_ZONA),
        (L - _TRIPLE_RECTO, _BANDA_A_TRIPLE), (L - _TRIPLE_RECTO, W - _BANDA_A_TRIPLE),
        (L - _FONDO_A_ARO, medio),
        (L, 0), (L, _BANDA_A_TRIPLE), (L, zona), (L, zona + _ANCHO_ZONA),
        (L, W - _BANDA_A_TRIPLE), (L, W),
    ]


VERTICES = np.array(_vertices_cm(), dtype=np.float32) / 100.0   # metros
LARGO = _LARGO / 100.0
ANCHO = _ANCHO / 100.0
ARO_IZQ = VERTICES[6]
ARO_DER = VERTICES[26]


# ---------------------------------------------------------------------------
# Homografía
# ---------------------------------------------------------------------------

def homografia(keypoints: list[dict], conf_min: float = config.CONF_KEYPOINT,
               min_puntos: int = config.MIN_KEYPOINTS):
    """Calcula la homografía imagen → pista (metros) a partir de los keypoints.

    Devuelve (H, n_inliers) o (None, 0) si no hay puntos suficientes o son
    degenerados (por ejemplo, todos sobre la misma línea).
    """
    img, pista = [], []
    for k in keypoints:
        idx = int(k["class_id"])
        if k["confidence"] >= conf_min and 0 <= idx < len(VERTICES):
            img.append((k["x"], k["y"]))
            pista.append(VERTICES[idx])
    if len(img) < min_puntos:
        return None, 0

    img = np.array(img, dtype=np.float32)
    pista = np.array(pista, dtype=np.float32)

    # Evitar configuraciones casi colineales: área de la envolvente en la pista
    if cv2.contourArea(cv2.convexHull(pista)) < 4.0:
        return None, 0

    # Se estima pista → imagen para que el umbral RANSAC esté en píxeles
    H_pi, mascara = cv2.findHomography(pista, img, cv2.RANSAC, 12.0)
    if H_pi is None or mascara is None:
        return None, 0
    inliers = int(mascara.sum())
    if inliers < min_puntos:
        return None, 0
    try:
        H = np.linalg.inv(H_pi)
    except np.linalg.LinAlgError:
        return None, 0
    return H / H[2, 2], inliers


def proyectar(H: np.ndarray, puntos: np.ndarray) -> np.ndarray:
    """Proyecta puntos (N, 2) de la imagen a la pista (metros)."""
    if len(puntos) == 0:
        return np.empty((0, 2), dtype=np.float32)
    p = np.asarray(puntos, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def dentro_de_pista(punto, margen: float = 0.5) -> bool:
    x, y = punto
    return -margen <= x <= LARGO + margen and -margen <= y <= ANCHO + margen


# ---------------------------------------------------------------------------
# Dibujo de la pizarra
# ---------------------------------------------------------------------------

class Pizarra:
    """Convierte metros a píxeles y dibuja la pista completa."""

    COLOR_PARQUET = (96, 140, 196)     # BGR madera
    COLOR_ZONA = (70, 110, 170)
    COLOR_LINEA = (245, 245, 245)

    def __init__(self, escala: float = 40.0, margen: int = 40):
        self.escala = escala
        self.margen = margen
        self.ancho = int(round(LARGO * escala)) + 2 * margen
        self.alto = int(round(ANCHO * escala)) + 2 * margen
        self._fondo = self._dibujar_fondo()

    def a_px(self, punto) -> tuple[int, int]:
        # Se limita al margen visible para que nada se dibuje fuera del lienzo
        limite = (self.margen - 6) / self.escala
        x = min(max(float(punto[0]), -limite), LARGO + limite)
        y = min(max(float(punto[1]), -limite), ANCHO + limite)
        return (int(round(self.margen + x * self.escala)),
                int(round(self.margen + y * self.escala)))

    def fondo(self) -> np.ndarray:
        return self._fondo.copy()

    def _polilinea(self, img, puntos, grosor=2, cerrada=False):
        pts = np.array([self.a_px(p) for p in puntos], dtype=np.int32)
        cv2.polylines(img, [pts], cerrada, self.COLOR_LINEA, grosor, cv2.LINE_AA)

    def _arco(self, centro, radio, a0, a1, n=60):
        angulos = np.linspace(np.radians(a0), np.radians(a1), n)
        return [(centro[0] + radio * np.cos(a), centro[1] + radio * np.sin(a))
                for a in angulos]

    def _dibujar_fondo(self) -> np.ndarray:
        img = np.full((self.alto, self.ancho, 3), (40, 40, 40), dtype=np.uint8)
        cv2.rectangle(img, self.a_px((0, 0)), self.a_px((LARGO, ANCHO)),
                      self.COLOR_PARQUET, -1)

        zona_y0 = (_ANCHO - _ANCHO_ZONA) / 200.0
        zona_y1 = zona_y0 + _ANCHO_ZONA / 100.0
        largo_zona = _LARGO_ZONA / 100.0
        medio = ANCHO / 2

        for lado in (0, 1):
            # x espejo para el lado derecho
            def X(x):
                return x if lado == 0 else LARGO - x

            # Zona pintada
            cv2.rectangle(img, self.a_px((min(X(0), X(largo_zona)), zona_y0)),
                          self.a_px((max(X(0), X(largo_zona)), zona_y1)),
                          self.COLOR_ZONA, -1)
            self._polilinea(img, [(X(0), zona_y0), (X(largo_zona), zona_y0),
                                  (X(largo_zona), zona_y1), (X(0), zona_y1)])

            # Círculo de tiros libres
            cx = X(largo_zona)
            self._polilinea(img, self._arco((cx, medio), _RADIO_CIRCULO / 100, 0, 360), 1)

            # Triple: tramos rectos + arco
            aro = (X(_FONDO_A_ARO / 100), medio)
            r = _RADIO_TRIPLE / 100
            y_esq0 = _BANDA_A_TRIPLE / 100
            y_esq1 = ANCHO - y_esq0
            ang = np.degrees(np.arcsin((medio - y_esq0) / r))
            if lado == 0:
                arco = self._arco(aro, r, -ang, ang)
            else:
                arco = self._arco(aro, r, 180 - ang, 180 + ang)
            x_union = arco[0][0]
            self._polilinea(img, [(X(0), y_esq0), (x_union, y_esq0)])
            self._polilinea(img, [(X(0), y_esq1), (arco[-1][0], y_esq1)])
            self._polilinea(img, arco)

            # Zona restringida, tablero y aro
            if lado == 0:
                restringida = self._arco(aro, _RADIO_RESTRINGIDA / 100, -90, 90)
            else:
                restringida = self._arco(aro, _RADIO_RESTRINGIDA / 100, 90, 270)
            self._polilinea(img, restringida, 1)
            self._polilinea(img, [(X(1.22), medio - 0.91), (X(1.22), medio + 0.91)], 3)
            cv2.circle(img, self.a_px(aro), max(3, int(0.23 * self.escala)),
                       (0, 110, 255), 2, cv2.LINE_AA)

        # Límites, medio campo y círculo central
        self._polilinea(img, [(0, 0), (LARGO, 0), (LARGO, ANCHO), (0, ANCHO)], 3, True)
        self._polilinea(img, [(LARGO / 2, 0), (LARGO / 2, ANCHO)])
        self._polilinea(img, self._arco((LARGO / 2, medio), _RADIO_CIRCULO / 100, 0, 360))
        return img
