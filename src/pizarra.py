"""Render del vídeo final: vídeo original anotado + pizarra 2D sincronizada."""

from pathlib import Path

import cv2
import numpy as np

from src.cancha import Pizarra
from src.identidad import Jugador, visibles_por_equipo
from src.recortes import leer_fotogramas

ALTO_CABECERA = 56
COLOR_BALON = (0, 140, 255)
SEGUNDOS_ESTELA = 1.5
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def _color_texto(color) -> tuple[int, int, int]:
    gris = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2GRAY)[0, 0]
    return (20, 20, 20) if gris > 140 else (245, 245, 245)


def _texto_centrado(img, texto, centro, escala, color, grosor=1, fondo=None):
    (w, h), _ = cv2.getTextSize(texto, FUENTE, escala, grosor)
    x, y = int(centro[0] - w / 2), int(centro[1] + h / 2)
    if fondo is not None:
        cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + 4), fondo, -1)
    cv2.putText(img, texto, (x, y), FUENTE, escala, color, grosor, cv2.LINE_AA)


def _estela(img, pizarra, puntos, color, grosor):
    validos = [pizarra.a_px(p) for p in puntos if not np.isnan(p[0])]
    if len(validos) >= 2:
        cv2.polylines(img, [np.array(validos, dtype=np.int32)], False, color, grosor, cv2.LINE_AA)


def dibujar_pizarra(pizarra: Pizarra, k: int, jugadores: list[Jugador], balon: dict,
                    colores: dict, fps: float) -> np.ndarray:
    img = pizarra.fondo()
    largo = int(SEGUNDOS_ESTELA * fps)
    visibles = visibles_por_equipo(jugadores, k)

    for j in visibles:
        _estela(img, pizarra, j.pos[max(0, k - largo):k + 1], colores[j.equipo], 2)

    _estela(img, pizarra, balon["pos"][max(0, k - largo):k + 1], COLOR_BALON, 1)

    poseedor = balon["poseedor"][k]
    for j in visibles:
        centro = pizarra.a_px(j.pos[k])
        color = colores[j.equipo]
        if j.id == poseedor:
            cv2.circle(img, centro, 21, COLOR_BALON, 3, cv2.LINE_AA)
        cv2.circle(img, centro, 15, color, -1, cv2.LINE_AA)
        cv2.circle(img, centro, 15, (25, 25, 25), 2, cv2.LINE_AA)
        _texto_centrado(img, j.dorsal or "?", centro, 0.5, _color_texto(color), 2)
        nombre = j.etiqueta.split(" ", 1)[1] if j.dorsal and j.nombre else (
            "" if j.dorsal else j.etiqueta)
        if nombre:
            _texto_centrado(img, nombre, (centro[0], centro[1] + 28), 0.42,
                            (255, 255, 255), 1, fondo=(30, 30, 30))

    if not np.isnan(balon["pos"][k, 0]):
        c = pizarra.a_px(balon["pos"][k])
        if balon["estado"][k] == "aire":
            cv2.circle(img, c, 8, COLOR_BALON, 2, cv2.LINE_AA)
        else:
            cv2.circle(img, (c[0] + 12, c[1] - 12), 7, COLOR_BALON, -1, cv2.LINE_AA)
            cv2.circle(img, (c[0] + 12, c[1] - 12), 7, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def anotar_video(frame: np.ndarray, k: int, jugadores: list[Jugador], colores: dict,
                 balon_imagen: dict | None) -> np.ndarray:
    img = frame.copy()
    for j in jugadores:
        if k not in j.cajas:
            continue
        x1, y1, x2, y2 = (int(v) for v in j.cajas[k])
        color = colores[j.equipo]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        etiqueta = j.etiqueta
        (w, h), _ = cv2.getTextSize(etiqueta, FUENTE, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w + 8, y1), color, -1)
        cv2.putText(img, etiqueta, (x1 + 4, y1 - 6), FUENTE, 0.7, _color_texto(color), 2, cv2.LINE_AA)
    if balon_imagen is not None:
        centro = tuple(int(v) for v in balon_imagen["pos"])
        r = int(balon_imagen["radio"]) + 5
        # Círculo continuo = detectado; discontinuo = seguido por flujo óptico
        if balon_imagen["fuente"] == "det":
            cv2.circle(img, centro, r, COLOR_BALON, 3, cv2.LINE_AA)
        else:
            for a in range(0, 360, 40):
                cv2.ellipse(img, centro, (r, r), 0, a, a + 20, COLOR_BALON, 3, cv2.LINE_AA)
    return img


def _cabecera(ancho, k, fps, nombres_equipo, colores, balon, jugadores) -> np.ndarray:
    img = np.full((ALTO_CABECERA, ancho, 3), (28, 28, 28), dtype=np.uint8)
    cv2.putText(img, f"t = {k / fps:5.2f} s", (20, 38), FUENTE, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    x = 230
    for e in (0, 1):
        cv2.circle(img, (x, 28), 12, colores[e], -1, cv2.LINE_AA)
        cv2.circle(img, (x, 28), 12, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, nombres_equipo[e], (x + 20, 38), FUENTE, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        x += 140
    estado = balon["estado"][k]
    if estado == "posesion":
        j = next((j for j in jugadores if j.id == balon["poseedor"][k]), None)
        texto = f"Balon: {j.etiqueta} ({nombres_equipo[j.equipo]})" if j else "Balon: posesion"
    elif estado == "aire":
        texto = "Balon: en el aire"
    elif estado == "aro":
        texto = "Balon: en el aro"
    else:
        texto = "Balon: sin dato"
    cv2.putText(img, texto, (x + 30, 38), FUENTE, 0.8, COLOR_BALON, 2, cv2.LINE_AA)
    return img


def renderizar(ruta_video: Path, ruta_salida: Path, paso: int, fps: float,
               fotogramas: list[dict], jugadores: list[Jugador], balon: dict,
               colores: dict, nombres_equipo: dict) -> None:
    pizarra = Pizarra(escala=40, margen=40)
    alto_panel = pizarra.alto
    escritor = None

    for k, frame in leer_fotogramas(ruta_video, paso):
        if k >= len(fotogramas):
            break
        video = anotar_video(frame, k, jugadores, colores, balon["imagen"][k])
        ancho_video = int(round(video.shape[1] * alto_panel / video.shape[0]))
        video = cv2.resize(video, (ancho_video, alto_panel), interpolation=cv2.INTER_AREA)
        panel = np.hstack([video, dibujar_pizarra(pizarra, k, jugadores, balon, colores, fps)])
        cabecera = _cabecera(panel.shape[1], k, fps, nombres_equipo, colores, balon, jugadores)
        salida = np.vstack([cabecera, panel])
        # Dimensiones pares para el códec
        h, w = salida.shape[:2]
        salida = salida[: h - h % 2, : w - w % 2]

        if escritor is None:
            ruta_salida.parent.mkdir(parents=True, exist_ok=True)
            escritor = cv2.VideoWriter(str(ruta_salida), cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps, (salida.shape[1], salida.shape[0]))
            if not escritor.isOpened():
                raise RuntimeError("No se pudo crear el vídeo de salida")
        escritor.write(salida)

    if escritor is not None:
        escritor.release()
    print(f"Vídeo de la pizarra: {ruta_salida}")
