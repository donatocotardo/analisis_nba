"""Plantillas NBA (dorsal → nombre) con nba_api y caché local editable."""

import json
from datetime import date

from src import config

DIR_PLANTILLAS = config.DIR_CACHE / "plantillas"


def temporada_actual(hoy: date | None = None) -> str:
    hoy = hoy or date.today()
    inicio = hoy.year if hoy.month >= 10 else hoy.year - 1
    return f"{inicio}-{(inicio + 1) % 100:02d}"


def obtener(abreviatura: str, temporada: str) -> dict[str, str]:
    """Devuelve {"24": "Nombre Apellido", ...}. Si falla la descarga, {}.

    El JSON queda en cache/plantillas/ y se puede editar a mano (fichajes,
    dorsales cambiados, etc.).
    """
    abreviatura = abreviatura.upper()
    DIR_PLANTILLAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_PLANTILLAS / f"{abreviatura}_{temporada}.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))

    try:
        from nba_api.stats.endpoints import commonteamroster
        from nba_api.stats.static import teams

        equipo = teams.find_team_by_abbreviation(abreviatura)
        if not equipo:
            print(f"Aviso: equipo desconocido '{abreviatura}'")
            return {}
        df = commonteamroster.CommonTeamRoster(
            team_id=equipo["id"], season=temporada, timeout=30
        ).get_data_frames()[0]
        plantilla = {str(fila["NUM"]).strip(): fila["PLAYER"]
                     for _, fila in df.iterrows() if str(fila["NUM"]).strip()}
    except Exception as error:   # red, API de la NBA caída, etc.
        print(f"Aviso: no se pudo descargar la plantilla de {abreviatura}: {error}")
        return {}

    ruta.write_text(json.dumps(plantilla, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Plantilla {abreviatura} {temporada}: {len(plantilla)} jugadores → {ruta}")
    return plantilla


def nombre_corto(nombre: str) -> str:
    """'Victor Wembanyama' → 'V. Wembanyama'."""
    partes = nombre.split()
    if len(partes) < 2:
        return nombre
    return f"{partes[0][0]}. {' '.join(partes[1:])}"
