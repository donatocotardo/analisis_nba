"""Homografías por fotograma, seguimiento de jugadores y filtro "en pista"."""

from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from src import cancha

# Si un fotograma no tiene homografía válida se reutiliza la última durante
# como mucho este número de fotogramas analizados.
MAX_FOTOGRAMAS_SIN_H = 15


@dataclass
class Observacion:
    k: int                       # índice de fotograma analizado (0..N-1)
    caja: np.ndarray             # xyxy en píxeles
    conf: float
    clase: str
    pie: np.ndarray | None       # posición en la pista (m) o None


@dataclass
class Track:
    id: int
    obs: list[Observacion] = field(default_factory=list)

    @property
    def inicio(self) -> int:
        return self.obs[0].k

    @property
    def fin(self) -> int:
        return self.obs[-1].k


def _rejilla(ancho: int, alto: int) -> np.ndarray:
    """Puntos de la imagen donde suele verse el suelo (para comparar homografías)."""
    xs = np.linspace(0.1 * ancho, 0.9 * ancho, 7)
    ys = np.linspace(0.40 * alto, 0.95 * alto, 4)
    return np.array([(x, y) for y in ys for x in xs], dtype=np.float32)


def calcular_homografias(fotogramas: list[dict], movimiento: list, ancho: int, alto: int,
                         max_desacuerdo_m: float = 1.5,
                         reinicio_tras: int = 8) -> list[np.ndarray | None]:
    """Fusiona la homografía de los keypoints con la predicha por el movimiento
    de cámara. Si los keypoints contradicen a la predicción (p. ej. confunden
    las dos mitades de la pista) se usa la predicción; si lo hacen de forma
    sostenida se asume que la predicción ha derivado y se reinicia."""
    rejilla = _rejilla(ancho, alto)
    homografias = []
    H_prev, desacuerdos, sin_dato = None, 0, 0
    usadas = {"keypoints": 0, "fusion": 0, "prediccion": 0, "reinicio": 0}

    for k, f in enumerate(fotogramas):
        H_kp, _ = cancha.homografia(f["keypoints"])
        H_pred = None
        if H_prev is not None and movimiento[k] is not None:
            H_pred = H_prev @ movimiento[k]
            H_pred /= H_pred[2, 2]

        if H_kp is None and H_pred is None:
            sin_dato += 1
            H = H_prev if sin_dato <= MAX_FOTOGRAMAS_SIN_H else None
        elif H_kp is None:
            H, sin_dato = H_pred, 0
            usadas["prediccion"] += 1
        elif H_pred is None:
            H, sin_dato = H_kp, 0
            usadas["keypoints"] += 1
        else:
            sin_dato = 0
            gk = cancha.proyectar(H_kp, rejilla)
            gp = cancha.proyectar(H_pred, rejilla)
            diferencia = float(np.median(np.linalg.norm(gk - gp, axis=1)))
            if diferencia < max_desacuerdo_m:
                H, _ = cv2.findHomography(rejilla, 0.5 * (gk + gp))
                desacuerdos = 0
                usadas["fusion"] += 1
            else:
                desacuerdos += 1
                if desacuerdos >= reinicio_tras:
                    H, desacuerdos = H_kp, 0
                    usadas["reinicio"] += 1
                else:
                    H = H_pred
                    usadas["prediccion"] += 1
        homografias.append(H)
        if H is not None:
            H_prev = H

    validas = sum(h is not None for h in homografias)
    print(f"Homografía disponible en {validas}/{len(homografias)} fotogramas {usadas}")
    return suavizar_homografias(homografias, movimiento, rejilla)


def suavizar(homografias: list, movimiento: list, ancho: int, alto: int) -> list:
    return suavizar_homografias(homografias, movimiento, _rejilla(ancho, alto))


def suavizar_homografias(homografias: list, movimiento: list, rejilla: np.ndarray,
                         radio: int = 8) -> list:
    """Promedia cada homografía con las de los fotogramas vecinos llevadas al
    fotograma actual mediante el movimiento de cámara. Elimina el temblor sin
    retrasar los paneos (el movimiento de cámara se compensa, no se promedia)."""
    n = len(homografias)
    sigma = radio / 2
    suavizadas = []
    for k in range(n):
        if homografias[k] is None:
            suavizadas.append(None)
            continue
        puntos, pesos = [cancha.proyectar(homografias[k], rejilla)], [1.0]
        # Hacia atrás: T lleva píxeles de k a j
        T = np.eye(3)
        for j in range(k - 1, max(-1, k - radio - 1), -1):
            if movimiento[j + 1] is None:
                break
            T = movimiento[j + 1] @ T
            if homografias[j] is not None:
                puntos.append(cancha.proyectar(homografias[j] @ T, rejilla))
                pesos.append(np.exp(-((k - j) ** 2) / (2 * sigma ** 2)))
        # Hacia delante
        T = np.eye(3)
        for j in range(k + 1, min(n, k + radio + 1)):
            if movimiento[j] is None:
                break
            T = np.linalg.inv(movimiento[j]) @ T
            if homografias[j] is not None:
                puntos.append(cancha.proyectar(homografias[j] @ T, rejilla))
                pesos.append(np.exp(-((k - j) ** 2) / (2 * sigma ** 2)))
        pesos = np.array(pesos)[:, None, None]
        media = (np.array(puntos) * pesos).sum(axis=0) / pesos.sum()
        H, _ = cv2.findHomography(rejilla, media)
        suavizadas.append(H if H is not None else homografias[k])
    return suavizadas


def nms_jugadores(detecciones: list[dict], iou: float = 0.6) -> list[dict]:
    """Supresión de no máximos entre todas las clases de jugador (el detector
    puede dar 'player' y 'player-jump-shot' para la misma persona)."""
    jugadores = sorted((d for d in detecciones if es_jugador(d["clase"])),
                       key=lambda d: d["conf"], reverse=True)
    elegidos = []
    for d in jugadores:
        if all(_iou(d["caja"], e["caja"]) < iou for e in elegidos):
            elegidos.append(d)
    return elegidos


def es_jugador(clase: str) -> bool:
    return clase.lower().startswith("player")


def seguir_jugadores(fotogramas: list[dict], homografias: list, fps: float,
                     ancho: int, alto: int) -> dict[int, Track]:
    tracker = ByteTrackTracker(
        lost_track_buffer=int(fps * 1.5),
        frame_rate=fps,
        track_activation_threshold=0.4,
        minimum_consecutive_frames=2,
        minimum_iou_threshold=0.1,
        high_conf_det_threshold=0.5,
    )
    tracks: dict[int, Track] = {}

    for k, f in enumerate(fotogramas):
        jugadores = nms_jugadores(f["detecciones"])
        if jugadores:
            xyxy = np.array([d["caja"] for d in jugadores], dtype=np.float32)
            conf = np.array([d["conf"] for d in jugadores], dtype=np.float32)
            clases = np.arange(len(jugadores))       # índice para recuperar la detección
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            conf = np.empty((0,), dtype=np.float32)
            clases = np.empty((0,), dtype=int)
        dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=clases)
        salida = tracker.update(dets)
        if salida.tracker_id is None or len(salida) == 0:
            continue

        H = homografias[k]
        for i in range(len(salida)):
            tid = int(salida.tracker_id[i])
            if tid < 0:
                continue
            caja = salida.xyxy[i]
            original = _deteccion_original(jugadores, salida, i, caja)
            pie = None
            # Caja cortada por el borde de la imagen → los pies no se ven
            cortada = caja[0] < 4 or caja[2] > ancho - 4 or caja[3] > alto - 4
            if H is not None and not cortada:
                punto = np.array([[(caja[0] + caja[2]) / 2, caja[3]]], dtype=np.float32)
                pie = cancha.proyectar(H, punto)[0]
            tracks.setdefault(tid, Track(tid)).obs.append(Observacion(
                k=k, caja=caja.copy(),
                conf=float(original["conf"]) if original else 0.0,
                clase=original["clase"] if original else "player",
                pie=pie,
            ))
    return tracks


def _deteccion_original(jugadores, salida, i, caja):
    if salida.class_id is not None and i < len(salida.class_id):
        idx = int(salida.class_id[i])
        if 0 <= idx < len(jugadores):
            return jugadores[idx]
    # Respaldo: la detección con mayor IoU
    mejor, mejor_iou = None, 0.0
    for d in jugadores:
        iou = _iou(caja, d["caja"])
        if iou > mejor_iou:
            mejor, mejor_iou = d, iou
    return mejor


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def descartar_arbitros(tracks: dict[int, Track], fotogramas: list[dict],
                       umbral: float = 0.25) -> dict[int, Track]:
    """Quita los tracks que coinciden a menudo con una detección "referee"
    (el detector a veces etiqueta a un árbitro como jugador)."""
    resultado = {}
    for tid, t in tracks.items():
        coincidencias = 0
        for o in t.obs:
            arbitros = [d["caja"] for d in fotogramas[o.k]["detecciones"]
                        if d["clase"].lower() == "referee"]
            if any(_iou(o.caja, a) > 0.5 for a in arbitros):
                coincidencias += 1
        if coincidencias / len(t.obs) < umbral:
            resultado[tid] = t
    if len(resultado) < len(tracks):
        print(f"Descartados {len(tracks) - len(resultado)} tracks de árbitros")
    return resultado


def filtrar_en_pista(tracks: dict[int, Track], fps: float,
                     min_fraccion: float = 0.6, min_duracion_s: float = 0.4) -> dict[int, Track]:
    """Descarta tracks que casi siempre están fuera de la pista (banquillo, público)
    o que duran demasiado poco para ser fiables."""
    resultado = {}
    for tid, t in tracks.items():
        con_pie = [o for o in t.obs if o.pie is not None]
        if len(t.obs) < min_duracion_s * fps or not con_pie:
            continue
        # Margen amplio: jugadores esperando en la esquina o sacando de banda
        # pisan la línea; el banquillo y el público están bastante más lejos
        dentro = sum(cancha.dentro_de_pista(o.pie, 1.2) for o in con_pie)
        if dentro / len(con_pie) >= min_fraccion:
            resultado[tid] = t
    print(f"Tracks de jugadores: {len(tracks)} totales, {len(resultado)} en pista")
    return resultado
