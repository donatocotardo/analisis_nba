"""Lectura de dorsales con EasyOCR (GPU local) y votación a lo largo del track.

La votación usa la plantilla del equipo como información previa: una lectura
completa ("23") vota de lleno a ese dorsal; una lectura parcial ("2", muy
común cuando un dígito queda tapado o borroso) solo da un voto débil a los
dorsales de la plantilla que contienen ese dígito.
"""

from collections import defaultdict

import cv2
import numpy as np

MAX_POR_TRACK = 60
PESO_PARCIAL = 0.3

_lector = None


def _obtener_lector():
    global _lector
    if _lector is None:
        import easyocr
        import torch
        _lector = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return _lector


def _ocr(img) -> tuple[str, float] | None:
    resultados = _obtener_lector().readtext(img, allowlist="0123456789", detail=1)
    if not resultados:
        return None
    resultados.sort(key=lambda r: min(p[0] for p in r[0]))      # izquierda → derecha
    texto = "".join(r[1] for r in resultados)
    if not texto or len(texto) > 2:
        return None
    return texto, float(np.mean([r[2] for r in resultados]))


def _leer(recorte: np.ndarray) -> tuple[str, float] | None:
    escala = max(1.0, 80.0 / recorte.shape[0])
    img = cv2.resize(recorte, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    lectura = _ocr(img)
    if lectura is None:
        # Segundo intento: gris con contraste local (números claros sobre tela oscura y al revés)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gris = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gris)
        lectura = _ocr(gris)
    return lectura


def _repartidos(lista: list, n: int) -> list:
    """Elige n elementos repartidos en el tiempo (la lista viene ordenada por k)."""
    if len(lista) <= n:
        return lista
    idx = np.linspace(0, len(lista) - 1, n).astype(int)
    return [lista[i] for i in idx]


def _votar(lecturas: list[tuple[str, float]], roster: dict) -> dict[str, float]:
    votos = defaultdict(float)
    for texto, conf in lecturas:
        if not roster:
            votos[texto] += conf
            continue
        if texto in roster:
            votos[texto] += conf
        if len(texto) == 1:
            for dorsal in roster:
                if len(dorsal) == 2 and texto in dorsal:
                    votos[dorsal] += PESO_PARCIAL * conf
    return votos


def leer(numeros: list[tuple]) -> dict[int, list[tuple[str, float]]]:
    """numeros: [(track_id, k, peso, recorte)] → {track_id: [(texto, confianza)]}."""
    por_track = defaultdict(list)
    for tid, k, peso, recorte in numeros:
        if recorte.shape[0] >= 12:
            por_track[tid].append((k, recorte))
    resultado = {}
    for tid, lista in por_track.items():
        lista.sort(key=lambda e: e[0])
        resultado[tid] = [l for _, r in _repartidos(lista, MAX_POR_TRACK) if (l := _leer(r))]
    return resultado


def decidir(lecturas_por_track: dict[int, list], roster_por_track: dict[int, dict] | None = None,
            min_votos: float = 1.0, ventaja: float = 1.5, verbose: bool = True) -> dict[int, dict]:
    """Devuelve {track_id: {"dorsal": "23", "confianza": 0.8}} para los fiables.
    roster_por_track: {track_id: {dorsal: nombre}} (opcional)."""
    roster_por_track = roster_por_track or {}
    resultado = {}
    for tid, lecturas in lecturas_por_track.items():
        votos = _votar(lecturas, roster_por_track.get(tid, {}))
        if not votos:
            continue
        orden = sorted(votos.items(), key=lambda kv: kv[1], reverse=True)
        dorsal, puntos = orden[0]
        segundo = orden[1][1] if len(orden) > 1 else 0.0
        if puntos >= min_votos and puntos >= ventaja * segundo:
            resultado[tid] = {"dorsal": dorsal,
                              "confianza": round(puntos / sum(votos.values()), 2)}
    if verbose:
        print(f"Dorsales decididos en {len(resultado)}/{len(lecturas_por_track)} tracks con recortes")
    return resultado
