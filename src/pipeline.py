"""Orquesta el análisis completo de una jugada."""

import json
from pathlib import Path

import numpy as np
import yaml

from src import balon as mod_balon
from src import camara, config, dorsales, lineas, equipo, identidad, inferencia, plantillas, recortes, seguimiento
from src.pizarra import renderizar


def _cargar_correcciones(ruta: Path) -> dict:
    if not ruta.exists():
        return {}
    return yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}


ID_CORTES = 1000      # los tramos creados con "cortar" reciben ids 1000, 1001...


def _aplicar_cortes(tracks, correcciones, fps):
    """`tracks: {3: {cortar: 2.4}}` parte el track 3 en el segundo 2.4: desde ahí
    sigue siendo el 3 y lo anterior pasa a un track nuevo (1000, 1001...).
    Sirve cuando el tracker cambia de jugador en un cruce y no se detecta solo.
    Admite varios cortes: {cortar: [1.5, 4.0]}."""
    from src.seguimiento import Track
    nuevo_id = ID_CORTES
    for tid, c in sorted((correcciones.get("tracks") or {}).items()):
        if not c or "cortar" not in c or int(tid) not in tracks:
            continue
        cortes = c["cortar"] if isinstance(c["cortar"], list) else [c["cortar"]]
        t = tracks[int(tid)]
        limites = sorted(int(round(s * fps)) for s in cortes)
        tramos, inicio = [], 0
        for lim in limites:
            fin = next((i for i, o in enumerate(t.obs) if o.k >= lim), len(t.obs))
            tramos.append(t.obs[inicio:fin])
            inicio = fin
        tramos.append(t.obs[inicio:])
        tracks[int(tid)] = Track(int(tid), tramos[-1])
        for obs in tramos[:-1]:
            if obs:
                tracks[nuevo_id] = Track(nuevo_id, obs)
                print(f"Corte manual: track {tid} → tramo {nuevo_id} "
                      f"({obs[0].k / fps:.1f}-{obs[-1].k / fps:.1f}s)")
                nuevo_id += 1
    return tracks


def _reasignar_numeros(numeros, tracks_previos, tracks_nuevos):
    """Traslada los recortes de dorsal del track original al tramo en que quedó
    tras partir por cambio de camiseta."""
    nuevo_de_obs = {id(o): t.id for t in tracks_nuevos.values() for o in t.obs}
    obs_previa = {(t.id, o.k): o for t in tracks_previos.values() for o in t.obs}
    resultado = []
    for tid, k, peso, recorte in numeros:
        o = obs_previa.get((tid, k))
        nuevo = nuevo_de_obs.get(id(o)) if o is not None else None
        if nuevo is not None:
            resultado.append((nuevo, k, peso, recorte))
    return resultado


def _etiquetas_por_track(etiquetas, tracks_previos, tracks_nuevos) -> dict:
    """Traslada las etiquetas de color (track original, k) al tramo nuevo."""
    nuevo_de_obs = {id(o): t.id for t in tracks_nuevos.values() for o in t.obs}
    resultado = {}
    for t in tracks_previos.values():
        for o in t.obs:
            if (t.id, o.k) in etiquetas and id(o) in nuevo_de_obs:
                resultado[(nuevo_de_obs[id(o)], o.k)] = etiquetas[(t.id, o.k)]
    return resultado


def _plantilla_correcciones(ruta: Path, tracks, equipo_por_track, nombres_equipo, lecturas, fps):
    """Crea correcciones.yaml (todo comentado) para que el usuario lo edite."""
    lineas = [
        "# Correcciones manuales. Se aplican al volver a ejecutar main.py (sin repetir la inferencia).",
        "# Los IDs son los que aparecen como 'ID n' en el vídeo (o el primer track de cada jugador).",
        "#",
        "# equipos:            # fuerza qué grupo de color es cada equipo",
        f"#   0: {nombres_equipo[0]}",
        f"#   1: {nombres_equipo[1]}",
        "#",
        "# tracks:",
        "#   12: {dorsal: 24}                 # dorsal correcto",
        "#   15: {descartar: true}            # no es un jugador en pista",
        "#   20: {equipo: GSW, dorsal: 30}    # equipo y dorsal",
        "#   31: {nombre: \"Nombre Apellido\"}  # nombre a mano",
        "#   3: {cortar: 2.4}                 # el track cambia de jugador en el segundo 2.4",
        "#                                    # (lo anterior pasa a un track nuevo 1000, 1001...)",
        "#",
        "# Tracks detectados:",
    ]
    for tid, t in sorted(tracks.items()):
        e = equipo_por_track.get(tid)
        lineas.append(
            f"#   {tid}: equipo={nombres_equipo.get(e, '?')} "
            f"dorsal_leido={lecturas.get(tid, {}).get('dorsal', '-')} "
            f"t={t.inicio / fps:.1f}-{t.fin / fps:.1f}s"
        )
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def _guardar_datos(ruta: Path, fps, nombres_equipo, colores, jugadores, balon):
    def lista(pos):
        return [None if np.isnan(p[0]) else [round(float(p[0]), 2), round(float(p[1]), 2)]
                for p in pos]

    datos = {
        "fps": fps,
        "unidades": "metros; x a lo largo (0-28.65), y a lo ancho (0-15.24)",
        "equipos": {str(e): {"nombre": nombres_equipo[e], "color_bgr": colores[e]} for e in (0, 1)},
        "jugadores": [
            {"id": j.id, "tracks": j.tracks, "equipo": nombres_equipo[j.equipo],
             "dorsal": j.dorsal, "nombre": j.nombre, "posiciones": lista(j.pos)}
            for j in jugadores
        ],
        "balon": {"posiciones": lista(balon["pos"]), "estado": balon["estado"],
                  "poseedor": balon["poseedor"]},
    }
    ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")


def analizar(ruta_video: Path, local: str, visitante: str, temporada: str | None = None,
             fps_analisis: float = config.FPS_ANALISIS, reprocesar: bool = False) -> Path:
    temporada = temporada or plantillas.temporada_actual()
    dir_salida = config.DIR_OUTPUTS / ruta_video.stem

    # 1. Inferencia (cacheada)
    datos = inferencia.ejecutar(ruta_video, dir_salida, fps_analisis, reprocesar)
    fotogramas = datos["fotogramas"]
    fps = datos["fps_video"] / datos["paso"]
    n = len(fotogramas)

    # 2. Homografías y seguimiento
    movimiento = camara.calcular(ruta_video, datos["paso"], fotogramas, dir_salida, reprocesar)
    homografias = seguimiento.calcular_homografias(fotogramas, movimiento,
                                                   datos["ancho"], datos["alto"])
    homografias = lineas.refinar(ruta_video, datos["paso"], fotogramas, homografias,
                                 datos["ancho"], datos["alto"])
    homografias = seguimiento.suavizar(homografias, movimiento, datos["ancho"], datos["alto"])
    tracks = seguimiento.seguir_jugadores(fotogramas, homografias, fps,
                                          datos["ancho"], datos["alto"])
    tracks = seguimiento.descartar_arbitros(tracks, fotogramas)
    tracks = seguimiento.filtrar_en_pista(tracks, fps)
    ruta_corr = dir_salida / "correcciones.yaml"
    correcciones = _cargar_correcciones(ruta_corr)
    tracks = _aplicar_cortes(tracks, correcciones, fps)

    # 3. Equipos (color por fotograma, partiendo tracks con intercambio de identidad)
    print("Extrayendo recortes de camisetas y dorsales...")
    torsos, ocluidos, numeros = recortes.recopilar(ruta_video, datos["paso"], fotogramas, tracks)
    etiquetas, colores, brillos = equipo.clasificar(torsos, ocluidos)
    tracks_previos = tracks
    tracks, equipo_por_track = equipo.dividir_por_equipo(tracks, etiquetas, ocluidos, fps)
    tracks = seguimiento.filtrar_en_pista(tracks, fps)

    # 4. Dorsales sobre los tracks ya partidos
    print("Leyendo dorsales (EasyOCR)...")
    lecturas_ocr = dorsales.leer(_reasignar_numeros(numeros, tracks_previos, tracks))

    # 4. Plantillas y nombres de equipo
    rosters = {local.upper(): plantillas.obtener(local, temporada),
               visitante.upper(): plantillas.obtener(visitante, temporada)}
    nombres_equipo = identidad.nombrar_equipos(
        equipo_por_track, dorsales.decidir(lecturas_ocr, verbose=False),
        rosters[local.upper()], rosters[visitante.upper()],
        local, visitante, brillos, correcciones.get("equipos"))
    # Con el equipo conocido, se vuelve a votar usando su plantilla
    lecturas = dorsales.decidir(lecturas_ocr, {
        tid: rosters.get(nombres_equipo[e], {}) for tid, e in equipo_por_track.items()})
    if not ruta_corr.exists():
        _plantilla_correcciones(ruta_corr, tracks, equipo_por_track, nombres_equipo, lecturas, fps)

    # 5. Jugadores finales y balón
    jugadores = identidad.construir_jugadores(
        tracks, n, fps, equipo_por_track, lecturas, nombres_equipo, rosters, correcciones,
        etiquetas=_etiquetas_por_track(etiquetas, tracks_previos, tracks),
        etiquetas_debiles=_etiquetas_por_track(equipo.etiquetas_debiles, tracks_previos, tracks))
    balon = mod_balon.seguir(ruta_video, datos["paso"], fotogramas, jugadores, movimiento, fps)

    for j in sorted(jugadores, key=lambda j: (j.equipo, j.id)):
        presencia = np.count_nonzero(~np.isnan(j.pos[:, 0])) / fps
        print(f"  {nombres_equipo[j.equipo]:>4}  {j.etiqueta:<28} tracks={j.tracks} "
              f"en pizarra {presencia:.1f}s")

    # 6. Salidas
    _guardar_datos(dir_salida / "datos.json", fps, nombres_equipo, colores, jugadores, balon)
    ruta_video_salida = dir_salida / "pizarra.mp4"
    renderizar(ruta_video, ruta_video_salida, datos["paso"], fps, fotogramas,
               jugadores, balon, colores, nombres_equipo)
    print(f"Correcciones opcionales: {ruta_corr}")
    return ruta_video_salida
