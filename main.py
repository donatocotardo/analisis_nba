"""Analizador de jugadas NBA → pizarra 2D.

Uso:
    python main.py videos/jugada.mp4 --local SAS --visitante GSW
    python main.py videos/otra.mp4 --local LAL --visitante BOS --temporada 2026-27
    python main.py videos/jugada.mp4 --local SAS --visitante GSW --reprocesar
"""

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

from src import config  # noqa: E402
from src.pipeline import analizar  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Analiza una jugada NBA y genera una pizarra 2D")
    parser.add_argument("video", type=Path, help="Ruta del clip (mp4)")
    parser.add_argument("--local", required=True, help="Abreviatura del equipo local (p. ej. SAS)")
    parser.add_argument("--visitante", required=True, help="Abreviatura del visitante (p. ej. GSW)")
    parser.add_argument("--temporada", help="Temporada NBA, p. ej. 2026-27 (por defecto la actual)")
    parser.add_argument("--fps", type=float, default=config.FPS_ANALISIS,
                        help=f"Fotogramas por segundo a analizar (por defecto {config.FPS_ANALISIS})")
    parser.add_argument("--reprocesar", action="store_true",
                        help="Repite la inferencia aunque exista en caché")
    args = parser.parse_args()

    if not args.video.exists():
        parser.error(f"No existe el vídeo: {args.video}")

    inicio = time.perf_counter()
    salida = analizar(args.video, args.local, args.visitante, args.temporada,
                      args.fps, args.reprocesar)
    print(f"\nListo en {time.perf_counter() - inicio:.1f} s → {salida}")


if __name__ == "__main__":
    main()
