"""Construye jugadores finales: nombra equipos, aplica correcciones, une tracks
rotos y suaviza las trayectorias en la pista."""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from src import cancha, plantillas
from src.seguimiento import Track


@dataclass
class Jugador:
    id: int
    tracks: list[int]
    equipo: int                     # 0 / 1 (grupo de color)
    dorsal: str | None
    nombre: str | None
    pos: np.ndarray                 # (N, 2) metros, NaN si no hay dato
    cajas: dict[int, np.ndarray] = field(default_factory=dict)   # k -> xyxy
    clases: dict[int, str] = field(default_factory=dict)         # k -> clase detectada

    @property
    def etiqueta(self) -> str:
        if self.dorsal and self.nombre:
            return f"#{self.dorsal} {plantillas.nombre_corto(self.nombre)}"
        if self.dorsal:
            return f"#{self.dorsal}"
        return f"ID {self.id}"


# ---------------------------------------------------------------------------
# Equipos
# ---------------------------------------------------------------------------

def nombrar_equipos(equipo_por_track, dorsales, roster_local, roster_visit,
                    local, visitante, brillos, forzado=None) -> dict[int, str]:
    """Decide qué grupo de color es el equipo local y cuál el visitante."""
    if forzado:
        mapa = {int(k): str(v).upper() for k, v in forzado.items()}
        print(f"Equipos fijados por correcciones.yaml: {mapa}")
        return mapa

    puntos = 0     # > 0 → grupo 0 es el local
    for tid, info in dorsales.items():
        d = info["dorsal"]
        en_local, en_visit = d in roster_local, d in roster_visit
        if en_local == en_visit:
            continue
        signo = 1 if en_local else -1
        puntos += signo if equipo_por_track.get(tid) == 0 else -signo

    if puntos != 0:
        grupo_local = 0 if puntos > 0 else 1
        motivo = f"dorsales coincidentes con las plantillas (puntuación {puntos:+d})"
    else:
        grupo_local = 0 if brillos[0] >= brillos[1] else 1
        motivo = "camiseta más clara = local (sin dorsales concluyentes)"
    print(f"Equipo local {local} = grupo {grupo_local} ({motivo})")
    return {grupo_local: local.upper(), 1 - grupo_local: visitante.upper()}


# ---------------------------------------------------------------------------
# Series de posiciones
# ---------------------------------------------------------------------------

def _serie(track: Track, n: int) -> np.ndarray:
    pos = np.full((n, 2), np.nan, dtype=np.float32)
    for o in track.obs:
        if o.pie is not None:
            pos[o.k] = o.pie
    return pos


def _primera_ultima(pos: np.ndarray):
    validos = np.where(~np.isnan(pos[:, 0]))[0]
    if len(validos) == 0:
        return None, None
    return validos[0], validos[-1]


def _quitar_saltos(pos: np.ndarray, fps: float, vmax: float = 11.0) -> np.ndarray:
    """Elimina puntos que implican velocidades imposibles (fallos de homografía)."""
    pos = pos.copy()
    validos = np.where(~np.isnan(pos[:, 0]))[0]
    if len(validos) < 3:
        return pos
    # Mediana deslizante como referencia robusta
    ref = pos.copy()
    for i in validos:
        ventana = validos[np.abs(validos - i) <= 3]
        ref[i] = np.median(pos[ventana], axis=0)
    for i in validos:
        if np.linalg.norm(pos[i] - ref[i]) > vmax / fps * 3:
            pos[i] = np.nan
    return pos


def _interpolar(pos: np.ndarray, max_hueco: int) -> np.ndarray:
    pos = pos.copy()
    validos = np.where(~np.isnan(pos[:, 0]))[0]
    for a, b in zip(validos[:-1], validos[1:]):
        if 1 < b - a <= max_hueco:
            for d in (0, 1):
                pos[a:b + 1, d] = np.linspace(pos[a, d], pos[b, d], b - a + 1)
    return pos


def _limitar_velocidad(pos: np.ndarray, fps: float, vmax: float = 9.5) -> np.ndarray:
    """Ningún jugador supera ~9.5 m/s: los saltos mayores se recortan."""
    pos = pos.copy()
    paso_max = vmax / fps
    anterior = None
    for i in range(len(pos)):
        if np.isnan(pos[i, 0]):
            anterior = None
            continue
        if anterior is not None:
            delta = pos[i] - pos[anterior]
            d = float(np.linalg.norm(delta))
            if d > paso_max:
                pos[i] = pos[anterior] + delta * (paso_max / d)
        anterior = i
    return pos


def _dentro_de_pista(pos: np.ndarray, margen: float = 0.4) -> np.ndarray:
    pos = pos.copy()
    pos[:, 0] = np.clip(pos[:, 0], -margen, cancha.LARGO + margen)
    pos[:, 1] = np.clip(pos[:, 1], -margen, cancha.ANCHO + margen)
    return pos


def _suavizar(pos: np.ndarray, ventana: int) -> np.ndarray:
    pos = pos.copy()
    valido = ~np.isnan(pos[:, 0])
    i = 0
    while i < len(pos):
        if not valido[i]:
            i += 1
            continue
        j = i
        while j < len(pos) and valido[j]:
            j += 1
        largo = j - i
        if largo >= 5:
            w = min(ventana, largo if largo % 2 else largo - 1)
            if w >= 5:
                pos[i:j] = savgol_filter(pos[i:j], w, 2, axis=0)
        i = j
    return pos


# ---------------------------------------------------------------------------
# Unión de tracks
# ---------------------------------------------------------------------------

def _unir(candidatos: list[dict], fps: float) -> list[list[int]]:
    """Encadena tracks del mismo equipo que no se solapan en el tiempo."""
    enlaces = []
    for a in candidatos:
        for b in candidatos:
            if a is b or a["equipo"] != b["equipo"]:
                continue
            if a["fin"] >= b["inicio"]:
                continue
            if a["dorsal"] and b["dorsal"] and a["dorsal"] != b["dorsal"]:
                continue
            hueco = (b["inicio"] - a["fin"]) / fps
            mismo_dorsal = a["dorsal"] is not None and a["dorsal"] == b["dorsal"]
            if hueco > (4.0 if mismo_dorsal else 1.0):
                continue
            pa = a["pos"][a["ultimo"]]
            pb = b["pos"][b["primero"]]
            distancia = float(np.linalg.norm(pa - pb))
            limite = (2.0 + 7.0 * hueco) if mismo_dorsal else (1.0 + 6.0 * hueco)
            if distancia <= limite:
                coste = distancia + 2.0 * hueco - (5.0 if mismo_dorsal else 0.0)
                enlaces.append((coste, a["id"], b["id"]))

    enlaces.sort()
    siguiente, anterior = {}, {}
    for _, a, b in enlaces:
        if a in siguiente or b in anterior:
            continue
        # evitar ciclos
        x = a
        while x in anterior:
            x = anterior[x]
        if x == b:
            continue
        siguiente[a], anterior[b] = b, a

    cadenas = []
    for c in candidatos:
        if c["id"] in anterior:
            continue
        cadena, x = [c["id"]], c["id"]
        while x in siguiente:
            x = siguiente[x]
            cadena.append(x)
        cadenas.append(cadena)
    return cadenas


def construir_jugadores(tracks: dict[int, Track], n: int, fps: float,
                        equipo_por_track: dict[int, int], dorsales: dict[int, dict],
                        nombres_equipo: dict[int, str], rosters: dict[str, dict],
                        correcciones: dict, etiquetas: dict | None = None,
                        etiquetas_debiles: dict | None = None) -> list[Jugador]:
    inverso = {v: k for k, v in nombres_equipo.items()}
    corr_tracks = {int(k): v or {} for k, v in (correcciones.get("tracks") or {}).items()}

    candidatos = []
    for tid, t in tracks.items():
        c = corr_tracks.get(tid, {})
        if c.get("descartar"):
            continue
        equipo = equipo_por_track.get(tid)
        if "equipo" in c and str(c["equipo"]).upper() in inverso:
            equipo = inverso[str(c["equipo"]).upper()]
        if equipo is None:
            continue
        if "dorsal" in c:
            dorsal = str(c["dorsal"])
        else:
            dorsal = dorsales.get(tid, {}).get("dorsal")
            roster = rosters.get(nombres_equipo[equipo], {})
            if dorsal and roster and dorsal not in roster:
                dorsal = None            # lectura imposible para este equipo
        pos = _quitar_saltos(_serie(t, n), fps)
        primero, ultimo = _primera_ultima(pos)
        if primero is None:
            continue
        candidatos.append({"id": tid, "equipo": equipo, "dorsal": dorsal, "pos": pos,
                           "inicio": t.inicio, "fin": t.fin,
                           "primero": primero, "ultimo": ultimo, "track": t})

    por_id = {c["id"]: c for c in candidatos}
    descartes: set[tuple[int, int]] = set()
    cadenas = _completar_cinco(_unir(candidatos, fps), por_id, fps, etiquetas, descartes,
                               etiquetas_debiles=etiquetas_debiles)

    jugadores = []
    for cadena in cadenas:
        partes = [por_id[i] for i in cadena]
        pos = np.full((n, 2), np.nan, dtype=np.float32)
        cajas, clases = {}, {}
        for p in partes:
            for k in np.where(~np.isnan(p["pos"][:, 0]))[0]:
                if (p["id"], int(k)) not in descartes and np.isnan(pos[k, 0]):
                    pos[k] = p["pos"][k]
            for o in p["track"].obs:
                if (p["id"], o.k) not in descartes:
                    cajas.setdefault(o.k, o.caja)
                    clases.setdefault(o.k, o.clase)
        dorsal = next((p["dorsal"] for p in partes if p["dorsal"]), None)
        equipo = partes[0]["equipo"]
        roster = rosters.get(nombres_equipo[equipo], {})
        nombre = roster.get(dorsal) if dorsal else None
        if "nombre" in corr_tracks.get(cadena[0], {}):
            nombre = corr_tracks[cadena[0]]["nombre"]

        pos = _quitar_saltos(pos, fps)        # saltos en las uniones entre tracks
        pos = _interpolar(pos, max_hueco=int(fps * 0.8))
        pos = _limitar_velocidad(pos, fps)
        pos = _suavizar(pos, ventana=int(fps * 0.8) | 1)
        pos = _dentro_de_pista(pos)
        jugadores.append(Jugador(id=cadena[0], tracks=cadena, equipo=equipo,
                                 dorsal=dorsal, nombre=nombre, pos=pos,
                                 cajas=cajas, clases=clases))

    _dorsales_unicos(jugadores, set(corr_tracks))
    _rellenar_ocultos(jugadores, fps)

    print(f"Jugadores tras unir tracks: {len(jugadores)} "
          f"({sum(j.equipo == 0 for j in jugadores)} grupo 0, "
          f"{sum(j.equipo == 1 for j in jugadores)} grupo 1)")
    return jugadores


def _completar_cinco(cadenas: list[list[int]], por_id: dict, fps: float,
                     etiquetas: dict | None = None, descartes: set | None = None,
                     max_por_equipo: int = 5, solape_max_s: float = 1.0,
                     etiquetas_debiles: dict | None = None) -> list[list[int]]:
    """En pista hay 5 jugadores por equipo. Si un equipo tiene más identidades,
    son fragmentos del mismo jugador (sale de plano y vuelve, el tracker lo
    pierde o lo cambia por otro en un cruce). Se unen los pares compatibles
    (sin dorsales distintos y con un desplazamiento posible) empezando por el
    más verosímil, hasta quedar en 5.

    Pueden solaparse hasta `solape_max_s`: en esos fotogramas se queda el
    fragmento cuyo color de camiseta encaja con el equipo y el otro se recorta
    (se añade a `descartes` como (track_id, k))."""
    etiquetas = etiquetas or {}
    etiquetas_debiles = etiquetas_debiles or {}
    descartes = descartes if descartes is not None else set()
    solape_max = int(solape_max_s * fps)

    def info(cadena):
        frames, pos = {}, {}
        dorsal = None
        for i in cadena:
            p = por_id[i]
            dorsal = dorsal or p["dorsal"]
            for o in p["track"].obs:
                if (i, o.k) not in descartes:
                    frames.setdefault(o.k, i)
            for k in np.where(~np.isnan(p["pos"][:, 0]))[0]:
                if (i, int(k)) not in descartes:
                    pos.setdefault(int(k), p["pos"][k])
        return {"frames": frames, "pos": pos, "dorsal": dorsal,
                "equipo": por_id[cadena[0]]["equipo"]}

    def acuerdo(inf, frames):
        """Fotogramas del solape donde el color coincide con su equipo."""
        return sum(1 for k in frames
                   if etiquetas.get((inf["frames"][k], k)) == inf["equipo"])

    def evidencia_propia(inf, frames):
        """+ si en esos fotogramas lleva la camiseta de su equipo, - si la del
        rival. Etiquetas claras valen 1 y las dudosas 0.5."""
        s = 0.0
        for k in frames:
            clave = (inf["frames"][k], k)
            if clave in etiquetas:
                s += 1.0 if etiquetas[clave] == inf["equipo"] else -1.0
            elif clave in etiquetas_debiles:
                s += 0.5 if etiquetas_debiles[clave] == inf["equipo"] else -0.5
        return s

    def recorte(a, b):
        """Devuelve (perdedor 'a'|'b', fotogramas del solape) o None si no hay."""
        solape = set(a["frames"]) & set(b["frames"])
        if not solape:
            return None
        sa, sb = acuerdo(a, solape), acuerdo(b, solape)
        if sa == sb:
            perdedor = "a" if len(a["frames"]) < len(b["frames"]) else "b"
        else:
            perdedor = "a" if sa < sb else "b"
        return perdedor, solape

    def coste_union(a, b):
        if a["equipo"] != b["equipo"]:
            return None
        if a["dorsal"] and b["dorsal"] and a["dorsal"] != b["dorsal"]:
            return None
        solape = set(a["frames"]) & set(b["frames"])
        if len(solape) > solape_max:
            return None
        r = recorte(a, b)
        pos_a, pos_b = dict(a["pos"]), dict(b["pos"])
        penal_solape = 0.0
        if r:
            quitar = pos_a if r[0] == "a" else pos_b
            for k in r[1]:
                quitar.pop(k, None)
            # Si el fragmento recortado llevaba en el solape la camiseta de su
            # propio equipo, había dos jugadores de ese equipo a la vez: unión
            # muy improbable. Si llevaba la del rival, era un intercambio del
            # tracker y el recorte es correcto.
            perdedor = a if r[0] == "a" else b
            penal_solape = 20.0 * max(0.0, evidencia_propia(perdedor, r[1])) / fps
        a, b = dict(a, pos=pos_a), dict(b, pos=pos_b)
        # Recorrer la unión en el tiempo y medir cada cambio de fragmento
        marcas = sorted([(k, "a") for k in a["pos"]] + [(k, "b") for k in b["pos"]])
        if not marcas:
            return None
        peor = 0.0
        for (k1, o1), (k2, o2) in zip(marcas[:-1], marcas[1:]):
            if o1 == o2:
                continue
            p1 = (a if o1 == "a" else b)["pos"][k1]
            p2 = (a if o2 == "a" else b)["pos"][k2]
            hueco = max(k2 - k1, 1) / fps
            distancia = float(np.linalg.norm(p1 - p2))
            if distancia > 2.5 + 9.0 * hueco:      # más rápido que un jugador
                return None
            peor = max(peor, distancia / (hueco + 0.5))
        bonus = -1.0 if (a["dorsal"] and b["dorsal"]) else 0.0
        return peor + bonus + penal_solape

    cadenas = [list(c) for c in cadenas]
    uniones = 0
    while True:
        infos = [info(c) for c in cadenas]
        hecho = False
        for equipo in (0, 1):
            indices = [i for i, inf in enumerate(infos) if inf["equipo"] == equipo]
            if len(indices) <= max_por_equipo:
                continue
            mejor = None
            for x in indices:
                for y in indices:
                    if x >= y:
                        continue
                    c = coste_union(infos[x], infos[y])
                    if c is not None and (mejor is None or c < mejor[0]):
                        mejor = (c, x, y)
            if mejor:
                _, x, y = mejor
                r = recorte(infos[x], infos[y])
                if r:
                    perdedor = infos[x] if r[0] == "a" else infos[y]
                    for k in r[1]:
                        descartes.add((perdedor["frames"][k], k))
                # El que tiene dorsal (o más duración) da el id al jugador unido
                if (infos[y]["dorsal"] and not infos[x]["dorsal"]) or (
                        bool(infos[y]["dorsal"]) == bool(infos[x]["dorsal"])
                        and len(infos[y]["frames"]) > len(infos[x]["frames"])):
                    x, y = y, x
                cadenas[x] = cadenas[x] + cadenas[y]
                cadenas.pop(y)
                uniones += 1
                hecho = True
                break
        if not hecho:
            break
    if uniones:
        print(f"Unidos {uniones} fragmentos para dejar como máximo 5 jugadores por equipo")
    return cadenas


def _rellenar_ocultos(jugadores: list[Jugador], fps: float, max_hueco_s: float = 3.0,
                      cerca_m: float = 1.8) -> None:
    """Rellena huecos de un jugador tapado por otro (p. ej. un defensor pegado
    al atacante: el detector solo ve una caja). Si al desaparecer y al
    reaparecer está junto al mismo jugador, durante el hueco sigue a ese
    jugador con un desfase que pasa del inicial al final."""
    rellenos = 0
    for j in jugadores:
        validos = np.where(~np.isnan(j.pos[:, 0]))[0]
        for a, b in zip(validos[:-1], validos[1:]):
            if b - a <= 1 or (b - a) / fps > max_hueco_s:
                continue
            for otro in jugadores:
                if otro is j or np.isnan(otro.pos[a:b + 1, 0]).any():
                    continue
                off_a = j.pos[a] - otro.pos[a]
                off_b = j.pos[b] - otro.pos[b]
                if np.linalg.norm(off_a) <= cerca_m and np.linalg.norm(off_b) <= cerca_m:
                    for k in range(a + 1, b):
                        t = (k - a) / (b - a)
                        j.pos[k] = otro.pos[k] + (1 - t) * off_a + t * off_b
                    rellenos += 1
                    break
    if rellenos:
        print(f"Rellenados {rellenos} huecos de jugadores tapados por otro jugador")


def _dorsales_unicos(jugadores: list[Jugador], corregidos: set[int]) -> None:
    """Dos jugadores del mismo equipo presentes a la vez no pueden llevar el
    mismo dorsal: se mantiene en el que tiene más presencia (o corrección manual)."""
    def presencia(j):
        return np.count_nonzero(~np.isnan(j.pos[:, 0]))

    for a in jugadores:
        for b in jugadores:
            if a is b or a.equipo != b.equipo or not a.dorsal or a.dorsal != b.dorsal:
                continue
            solapan = np.any(~np.isnan(a.pos[:, 0]) & ~np.isnan(b.pos[:, 0]))
            if not solapan:
                continue
            a_manual = bool(set(a.tracks) & corregidos)
            b_manual = bool(set(b.tracks) & corregidos)
            perdedor = b if (a_manual, presencia(a)) >= (b_manual, presencia(b)) else a
            perdedor.dorsal, perdedor.nombre = None, None


def visibles_por_equipo(jugadores: list[Jugador], k: int, maximo: int = 5) -> list[Jugador]:
    """Jugadores con posición en el fotograma k, como mucho 5 por equipo
    (prioriza los de trayectoria más larga, que son los más fiables)."""
    resultado = []
    for e in (0, 1):
        presentes = [j for j in jugadores if j.equipo == e and not np.isnan(j.pos[k, 0])]
        presentes.sort(key=lambda j: -np.count_nonzero(~np.isnan(j.pos[:, 0])))
        resultado.extend(presentes[:maximo])
    return resultado
