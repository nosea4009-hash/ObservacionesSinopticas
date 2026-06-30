#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot de estaciones de observacion METAR / observacion sinoptica de superficie.

Usa MetPy (StationPlot) para el modelo de estacion y Cartopy para el mapa.
Datos de entrada (repo ObservacionesSinopticas):
  - estaciones_smn.txt    : catalogo de estaciones SMN (nombre, lat/lon, altura, WMO, OACI)
  - estado_tiempo*.txt    : observaciones (estado del cielo, T, Td, HR, viento, presion)

Salida: PNG con el modelo de estacion sobre el norte de Argentina y sur de Brasil.

El recuadro tiene cuadricula lat/lon, fondo blanco, fronteras provinciales finas
y fronteras nacionales gruesas. Justo arriba del recuadro se escribe, en fuente
Arial (Liberation Sans, metricamente identica):

    YYYYMMDDHH / METAR & Observacion Sinoptica de Superficie / TRP Meteorologia
"""

import argparse
import glob
import os
import re
import unicodedata

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from metpy.calc import wind_components, dewpoint_from_relative_humidity
from metpy.plots import StationPlot, current_weather, sky_cover
from metpy.units import units
from metpy.calc import reduce_point_density
from metpy.interpolate import interpolate_to_grid, remove_nan_observations


# URL oficial del SMN: "Tiempo Presente" (observaciones de superficie actuales).
# Devuelve un ZIP con un archivo estado_tiempo<AAAAMMDD>.txt
SMN_TIEMPO_PRESENTE_URL = "https://ssl.smn.gob.ar/dpd/zipopendata.php?dato=tiepre"


# --------------------------------------------------------------------------- #
# Lista de Regiones  (extent = LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
# Usar con:  python plot_observaciones.py --region nea
# --------------------------------------------------------------------------- #
REGIONS = {
    "region_nea":          (-63.5, -53.0, -31.5, -21.5),  # NEA: Misiones, Corrientes, Chaco, Formosa
    "region_noa":          (-68.5, -61.0, -32.0, -21.0),  # NOA: Jujuy, Salta, Tucuman, Catamarca, La Rioja, Sgo. del Estero
    "region_cuyo":         (-71.0, -64.5, -37.5, -28.0),  # Cuyo: Mendoza, San Juan, San Luis
    "region_pampas":       (-66.0, -56.0, -41.0, -29.0),  # Region Pampeana: Bs. As., La Pampa, Cordoba, Santa Fe, Entre Rios
    "region_centro":       (-66.0, -57.5, -35.5, -28.0),  # Centro: Cordoba, Santa Fe, Entre Rios
    "region_litoral":      (-63.0, -53.0, -34.5, -22.0),  # Litoral fluvial: Entre Rios, Corrientes, Misiones, Chaco, Formosa, Santa Fe
    "region_patagonia":    (-74.0, -62.0, -55.5, -38.0),  # Patagonia: Neuquen, Rio Negro, Chubut, Santa Cruz, Tierra del Fuego
    "region_buenos_aires": (-63.5, -56.5, -41.5, -33.0),  # Provincia de Buenos Aires
    "region_argentina":    (-74.0, -52.0, -56.0, -21.0),  # Todo el pais
    "region_norte_brasil": (-71.0, -47.0, -36.0, -19.0),  # Norte de Argentina + sur de Brasil (vista por defecto)
}
DEFAULT_REGION = "region_norte_brasil"


# --------------------------------------------------------------------------- #
# Fuente Arial (Liberation Sans es metricamente identica y libre)
# --------------------------------------------------------------------------- #
def setup_font():
    for path in glob.glob("/usr/share/fonts/liberation-sans/LiberationSans-*.ttf"):
        try:
            font_manager.fontManager.addfont(path)
        except Exception:
            pass
    plt.rcParams["font.family"] = "sans-serif"
    # Si el sistema tuviera Arial real la usaria; si no, Liberation Sans.
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]


# --------------------------------------------------------------------------- #
# Utilidades de normalizacion / parseo numerico
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "aero", "aeropuerto", "airport", "observatorio", "obs", "ba", "b", "a",
    "un", "universidad", "nacional", "pcia", "esc", "del", "de", "la", "el",
    "los", "las", "ex",
}


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def token_set(s: str) -> set:
    return {t for t in normalize(s).split() if t not in _STOPWORDS}


def to_float(s: str) -> float:
    s = (s or "").strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- #
# Parseo del catalogo de estaciones (fixed-width, latin-1)
# --------------------------------------------------------------------------- #
def parse_stations(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="latin-1") as f:
        lines = f.readlines()

    for line in lines[2:]:  # saltea las 2 lineas de encabezado
        if not line.strip():
            continue
        name = line[:30].strip()
        rest = line[30:]
        tokens = rest.split()

        # primer token entero (negativo) = grados de latitud
        idx = None
        for i, t in enumerate(tokens):
            if re.fullmatch(r"-?\d+", t):
                idx = i
                break
        if idx is None or idx + 6 > len(tokens):
            continue  # linea de continuacion / sin datos

        provincia = " ".join(tokens[:idx])
        lat_gr, lat_min, lon_gr, lon_min, alt, nro = (int(x) for x in tokens[idx:idx + 6])
        oaci = tokens[idx + 6] if len(tokens) > idx + 6 else ""

        lat = -(abs(lat_gr) + lat_min / 60.0) if lat_gr < 0 else (lat_gr + lat_min / 60.0)
        lon = -(abs(lon_gr) + lon_min / 60.0) if lon_gr < 0 else (lon_gr + lon_min / 60.0)

        rows.append(dict(nombre=name, provincia=provincia, lat=lat, lon=lon,
                         alt=alt, nro=nro, oaci=oaci))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Parseo de las observaciones (separado por ;)
# --------------------------------------------------------------------------- #
_DIRS = {
    "norte": 0.0, "nornoreste": 22.5, "noreste": 45.0, "estenoreste": 67.5,
    "este": 90.0, "estesudeste": 112.5, "sudeste": 135.0, "sudsudeste": 157.5,
    "sur": 180.0, "sudsudoeste": 202.5, "sudoeste": 225.0, "oestesudoeste": 247.5,
    "oeste": 270.0, "oestenoroeste": 292.5, "noroeste": 315.0, "nornoroeste": 337.5,
}


def parse_wind(s: str):
    """Devuelve (direccion_grados_FROM, velocidad_kmh).

    Calma -> (0, 0)   (circulo de calma)
    Direcciones variables / desconocida -> (nan, velocidad)
    """
    s = (s or "").strip()
    low = strip_accents(s).lower()
    if low.startswith("calma") or low == "":
        return 0.0, 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*$", s)
    spd = float(m.group(1)) if m else 0.0
    if "variable" in low:
        return np.nan, spd
    dirword = re.sub(r"[\d\.\s]+$", "", strip_accents(s)).strip().lower().replace(" ", "")
    return _DIRS.get(dirword, np.nan), spd


def sky_to_oktas(desc: str) -> int:
    d = strip_accents(desc).lower()
    if "cubierto" in d:
        return 8
    if "parcialmente nublado" in d:
        return 4
    if "algo nublado" in d:
        return 2
    if "nublado" in d:
        return 6
    if "despejado" in d:
        return 0
    return 0


def present_weather(desc: str) -> int:
    """Mapea el estado del cielo a un codigo WMO ww aproximado (0 = sin fenomeno)."""
    d = strip_accents(desc).lower()
    if "tormenta" in d:
        return 95
    if "lloviz" in d:
        return 51
    if "lluvia" in d or "precipitaci" in d:
        return 61
    if "nevada" in d or "nieve" in d:
        return 71
    if "niebla" in d:
        return 45
    if "neblina" in d or "bruma" in d:
        return 10
    if "humo" in d:
        return 4
    if "polvo" in d:
        return 6
    return 0


def parse_weather(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 10:
                continue
            nombre, fecha, hora, cielo, vis, temp, dew, hum, viento, pres = parts[:10]
            pres = pres.replace("/", "").strip()
            wdir, wspd = parse_wind(viento)
            rows.append(dict(
                nombre=nombre, fecha=fecha, hora=hora, cielo=cielo,
                temp=to_float(temp), dew=to_float(dew), hum=to_float(hum),
                wdir=wdir, wspd_kmh=wspd, pres=to_float(pres),
                oktas=sky_to_oktas(cielo), ww=present_weather(cielo),
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cruce observaciones <-> catalogo (fuzzy por tokens)
# --------------------------------------------------------------------------- #
_ALIASES = {  # nombre normalizado de obs -> nombre normalizado de catalogo
    "aeroparque buenos aires": "aeroparque aero",
    "buenos aires": "buenos aires observatorio",
    "esc aviacion militar": "escuela de aviacion militar aero",
    "pcia roque saenz pena": "presidencia roque saenz pena a",
    "jujuy universidad nacional": "jujuy u n",
    "san martin mza": "san martin mza",
    "base carlini": "base carlini ex jubany",
    "santa fe": "sauce viejo aero",
}


def best_match(obs_name: str, stations: pd.DataFrame):
    obs_norm = normalize(obs_name)
    obs_tokens = token_set(obs_name)

    # alias explicito
    target = _ALIASES.get(obs_norm)
    if target:
        cand = stations[stations["nombre"].apply(normalize) == target]
        if len(cand):
            return cand.iloc[0], 1.0

    best, best_score = None, 0.0
    for _, st in stations.iterrows():
        st_tokens = token_set(st["nombre"])
        if not st_tokens or not obs_tokens:
            continue
        inter = len(obs_tokens & st_tokens)
        union = len(obs_tokens | st_tokens)
        jaccard = inter / union if union else 0.0
        # bonus si todos los tokens de la obs estan contenidos
        contained = inter / len(obs_tokens)
        score = max(jaccard, 0.85 * contained)
        if score > best_score:
            best, best_score = st, score
    return best, best_score


def build_dataframe(weather: pd.DataFrame, stations: pd.DataFrame,
                    threshold: float = 0.5) -> pd.DataFrame:
    records, unmatched = [], []
    for _, obs in weather.iterrows():
        st, score = best_match(obs["nombre"], stations)
        if st is None or score < threshold:
            unmatched.append(obs["nombre"])
            continue

        temp, dew, hum = obs["temp"], obs["dew"], obs["hum"]
        # Td desde HR si no esta calculado
        if np.isnan(dew) and not np.isnan(temp) and not np.isnan(hum):
            dew = float(dewpoint_from_relative_humidity(
                temp * units.degC, hum * units.percent).to("degC").m)

        # presion de estacion -> reducida a nivel del mar
        mslp = reduce_to_msl(obs["pres"], st["alt"], temp)

        # viento km/h -> nudos -> componentes u, v
        spd_kt = obs["wspd_kmh"] / 1.852
        if np.isnan(obs["wdir"]):
            u = v = np.nan
        else:
            u_q, v_q = wind_components(spd_kt * units.knots, obs["wdir"] * units.deg)
            u, v = float(u_q.m), float(v_q.m)

        records.append(dict(
            nombre=st["nombre"], lat=st["lat"], lon=st["lon"], alt=st["alt"],
            tair=temp, dewp=dew, mslp=mslp, u=u, v=v,
            oktas=int(obs["oktas"]), ww=int(obs["ww"]),
        ))

    if unmatched:
        print(f"[aviso] {len(unmatched)} estaciones sin cruzar: "
              + ", ".join(unmatched))
    return pd.DataFrame(records)


def reduce_to_msl(p_station: float, elev_m: float, temp_c: float) -> float:
    """Reduce presion de estacion (hPa) a nivel medio del mar (formula barometrica)."""
    if np.isnan(p_station):
        return np.nan
    t = temp_c if not np.isnan(temp_c) else 15.0
    return p_station * (1 + (0.0065 * elev_m) / (t + 0.0065 * elev_m + 273.15)) ** 5.257


# --------------------------------------------------------------------------- #
# Formatters seguros (ignoran NaN)
# --------------------------------------------------------------------------- #
def f_int(v):
    return "" if (v is None or np.isnan(v)) else format(int(round(v)), "d")


def f_mslp(v):
    if v is None or np.isnan(v):
        return ""
    return format(int(round(v * 10)) % 1000, "03d")


# --------------------------------------------------------------------------- #
# Campo de presion: grilla, isobaras y centros de Alta/Baja
# --------------------------------------------------------------------------- #
def compute_mslp_grid(df, extent, hres: float = 0.25, max_alt: float = 1200.0,
                      search_radius: float = 4.0):
    """Interpola la MSLP de las estaciones a una grilla que cubre el recuadro.

    Usa interpolacion de Cressman con radio de busqueda: el campo se extiende
    varios grados mas alla de las estaciones (por eso las isobaras CRUZAN la
    frontera de Argentina hacia Brasil/Uruguay/oceano), pero queda acotado a
    valores fisicos (promedio ponderado de los datos), sin la extrapolacion
    descontrolada de RBF. Mas alla del radio el campo es NaN (no se dibuja).

    Descarta estaciones de gran altura (> max_alt m), cuya reduccion a nivel
    del mar es poco confiable y ensucia el campo de presion.
    """
    sub = df.dropna(subset=["mslp"])
    if "alt" in sub:
        sub = sub[sub["alt"] <= max_alt]
    if len(sub) < 4:
        print("[aviso] muy pocas estaciones con presion: no hay campo de presion")
        return None

    lon, lat, mslp = remove_nan_observations(
        sub["lon"].to_numpy(), sub["lat"].to_numpy(), sub["mslp"].to_numpy())

    margin = 1.0  # para que las isobaras lleguen a los bordes del recuadro
    bounds = {"west": extent[0] - margin, "east": extent[1] + margin,
              "south": extent[2] - margin, "north": extent[3] + margin}
    try:
        gx, gy, gz = interpolate_to_grid(
            lon, lat, mslp, interp_type="cressman", hres=hres,
            search_radius=search_radius, minimum_neighbors=1,
            boundary_coords=bounds)
    except Exception as exc:  # pragma: no cover
        print(f"[aviso] no se pudo interpolar el campo de presion: {exc}")
        return None
    return gx, gy, gz


def add_isobars(ax, gx, gy, gz, transform, step: float = 2.0, color="#1f4ed8"):
    lo = np.floor(np.nanmin(gz) / step) * step
    hi = np.ceil(np.nanmax(gz) / step) * step
    levels = np.arange(lo, hi + step, step)
    cs = ax.contour(gx, gy, gz, levels=levels, colors=color,
                    linewidths=0.9, alpha=0.85, transform=transform, zorder=1)
    ax.clabel(cs, inline=True, fmt="%d", fontsize=7)
    return cs


def mark_pressure_centers(ax, gx, gy, gz, transform,
                          neighborhood_deg: float = 5.0, hres: float = 0.25):
    """Marca centros de Alta (A, azul) y Baja (B, rojo) presion.

    Detecta maximos y minimos locales del campo con filtros de vecindad
    (scipy.ndimage), rotulandolos con la letra y el valor en hPa. Para evitar
    falsos centros: (1) ignora celdas NaN y las pegadas al borde de la zona con
    datos (su entorno cercano debe tener datos), y (2) agrupa extremos muy
    cercanos en uno solo.
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    size = max(5, int(round(neighborhood_deg / hres)))
    valid = ~np.isnan(gz)
    if valid.sum() < size * size:
        return
    filled = np.where(valid, gz, np.nanmean(gz))

    # el entorno cercano (~mitad de la vecindad) debe tener datos reales
    core_size = max(3, size // 2)
    core = minimum_filter(valid.astype(np.int8), size=core_size,
                          mode="constant", cval=0) == 1
    gz_max = maximum_filter(filled, size=size, mode="nearest")
    gz_min = minimum_filter(filled, size=size, mode="nearest")

    min_sep = neighborhood_deg * 0.6

    def _draw(mask, letra, color):
        placed = []
        for i, j in zip(*np.where(mask & core)):
            x, y, val = gx[i, j], gy[i, j], gz[i, j]
            if any(abs(x - px) < min_sep and abs(y - py) < min_sep
                   for px, py in placed):
                continue
            placed.append((x, y))
            ax.text(x, y, letra, transform=transform, color=color, fontsize=20,
                    fontweight="bold", ha="center", va="center", zorder=6,
                    clip_on=True)
            ax.text(x, y - 0.5, f"{int(round(val))}", transform=transform,
                    color=color, fontsize=8, ha="center", va="center", zorder=6,
                    clip_on=True)

    _draw(filled == gz_max, "A", "blue")
    _draw(filled == gz_min, "B", "red")


# --------------------------------------------------------------------------- #
# Descarga de datos actuales del SMN (Tiempo Presente)
# --------------------------------------------------------------------------- #
def download_latest_smn(dest_dir: str = ".") -> str:
    """Descarga el archivo de observaciones mas reciente del SMN y lo guarda.

    Devuelve la ruta al .txt extraido (estado_tiempo<AAAAMMDD>.txt).
    """
    import io
    import urllib.request
    import zipfile

    print(f"[info] descargando datos actuales del SMN: {SMN_TIEMPO_PRESENTE_URL}")
    req = urllib.request.Request(
        SMN_TIEMPO_PRESENTE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        blob = resp.read()

    zf = zipfile.ZipFile(io.BytesIO(blob))
    name = zf.namelist()[0]
    out_path = os.path.join(dest_dir, os.path.basename(name))
    with open(out_path, "wb") as fh:
        fh.write(zf.read(name))
    print(f"[ok] datos guardados en {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Ploteo
# --------------------------------------------------------------------------- #
def make_plot(df: pd.DataFrame, extent, datetime_str: str, out_path: str,
              density_deg: float = 0.5, isobaras: bool = False,
              iso_step: float = 2.0, centros: bool = False):
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=proj)
    ax.set_facecolor("white")

    # --- fronteras ---
    provincias = cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "50m")
    paises = cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", "50m")

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="black", linewidth=0.6, zorder=2)
    ax.add_feature(provincias, edgecolor="black", facecolor="none",
                   linewidth=0.4, zorder=2)            # provincias: finas
    ax.add_feature(paises, edgecolor="black", facecolor="none",
                   linewidth=1.6, zorder=3)            # nacionales: gruesas

    # --- cuadricula lat/lon dentro del recuadro ---
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    # recuadro negro del plot
    ax.spines["geo"].set_edgecolor("black")
    ax.spines["geo"].set_linewidth(1.0)

    # --- campo de presion: isobaras y/o centros A/B (opcional) ---
    if isobaras or centros:
        hres = 0.25
        grid = compute_mslp_grid(df, extent, hres=hres)
        if grid is not None:
            gx, gy, gz = grid
            if isobaras:
                add_isobars(ax, gx, gy, gz, proj, step=iso_step)
            if centros:
                mark_pressure_centers(ax, gx, gy, gz, proj, hres=hres)

    # --- declutter: reduce densidad de puntos ---
    if len(df):
        pts = np.c_[df["lon"].to_numpy(), df["lat"].to_numpy()]
        mask = reduce_point_density(pts, density_deg)
        d = df[mask].reset_index(drop=True)
    else:
        d = df

    # --- modelo de estacion (MetPy) ---
    sp = StationPlot(ax, d["lon"].to_numpy(), d["lat"].to_numpy(),
                     transform=proj, fontsize=8, clip_on=True)
    sp.plot_barb(d["u"].to_numpy(), d["v"].to_numpy(), zorder=5)
    sp.plot_parameter("NW", d["tair"].to_numpy(), color="red",
                      formatter=f_int, zorder=5)
    sp.plot_parameter("SW", d["dewp"].to_numpy(), color="darkgreen",
                      formatter=f_int, zorder=5)
    sp.plot_parameter("NE", d["mslp"].to_numpy(), color="black",
                      formatter=f_mslp, zorder=5)
    sp.plot_symbol("C", d["oktas"].to_numpy(), sky_cover, zorder=5)
    sp.plot_symbol("W", d["ww"].to_numpy(), current_weather,
                   color="purple", zorder=5)

    # --- titulo justo arriba del recuadro (Arial) ---
    title = (f"{datetime_str} / METAR & Observacion Sinoptica de Superficie "
             f"/ TRP Meteorologia")
    ax.set_title(title, fontsize=12, fontfamily="sans-serif", pad=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {len(d)} estaciones ploteadas -> {out_path}")


# --------------------------------------------------------------------------- #
# Fecha YYYYMMDDHH a partir del nombre de archivo + hora dominante
# --------------------------------------------------------------------------- #
def derive_datetime(weather_path: str, weather: pd.DataFrame) -> str:
    m = re.search(r"(\d{8})", os.path.basename(weather_path))
    ymd = m.group(1) if m else "00000000"
    horas = weather["hora"].dropna().astype(str).str.slice(0, 2)
    hh = horas.mode().iloc[0] if len(horas) else "00"
    return f"{ymd}{hh}"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Regiones disponibles (--region):\n  "
               + "\n  ".join(f"{name:22s} {ext}" for name, ext in REGIONS.items()),
    )
    ap.add_argument("--stations", default="estaciones_smn.txt")
    ap.add_argument("--weather", default=None,
                    help="archivo estado_tiempo*.txt (por defecto el mas reciente)")
    ap.add_argument("--actualizar", action="store_true",
                    help="descarga del SMN el archivo de observaciones mas reciente "
                         "(Tiempo Presente) antes de plotear")
    ap.add_argument("--isobaras", action=argparse.BooleanOptionalAction, default=False,
                    help="dibuja isobaras de presion al nivel del mar "
                         "(usar --isobaras para activarlas, --no-isobaras para no)")
    ap.add_argument("--paso-isobaras", dest="paso_isobaras", type=float, default=2.0,
                    help="intervalo entre isobaras en hPa (por defecto 2)")
    ap.add_argument("--centros", action=argparse.BooleanOptionalAction, default=False,
                    help="marca los centros de Alta (A) y Baja (B) presion "
                         "(usar --centros para activarlos, --no-centros para no)")
    ap.add_argument("--region", choices=list(REGIONS), default=None,
                    help="region predefinida (ver lista abajo). Tiene prioridad sobre --extent")
    ap.add_argument("--extent", nargs=4, type=float, default=None,
                    metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                    help="region manual; si no se indica region ni extent se usa "
                         + DEFAULT_REGION)
    ap.add_argument("--datetime", default=None, help="YYYYMMDDHH para el titulo")
    ap.add_argument("--density", type=float, default=None,
                    help="radio (grados) para reducir solape; por defecto se ajusta "
                         "automaticamente al tamano de la region")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_font()

    # --- resolver region ---
    if args.region:
        extent = REGIONS[args.region]
        region_name = args.region
    elif args.extent:
        extent = tuple(args.extent)
        region_name = "custom"
    else:
        extent = REGIONS[DEFAULT_REGION]
        region_name = DEFAULT_REGION

    # --- densidad automatica segun el ancho de la region ---
    if args.density is not None:
        density = args.density
    else:
        lon_span = abs(extent[1] - extent[0])
        density = max(0.08, round(lon_span / 48.0, 2))

    weather_path = args.weather
    if args.actualizar:
        weather_path = download_latest_smn(".")
    elif weather_path is None:
        cands = sorted(glob.glob("estado_tiempo*.txt"))
        if not cands:
            raise SystemExit("No se encontro ningun archivo estado_tiempo*.txt. "
                             "Usa --actualizar para descargarlo del SMN.")
        weather_path = cands[-1]

    stations = parse_stations(args.stations)
    weather = parse_weather(weather_path)
    print(f"[info] {len(stations)} estaciones en catalogo, "
          f"{len(weather)} observaciones en {weather_path}")
    print(f"[info] region={region_name}  extent={extent}  density={density}  "
          f"isobaras={args.isobaras}  centros={args.centros}")

    df = build_dataframe(weather, stations)

    dt = args.datetime or derive_datetime(weather_path, weather)
    out = args.out or f"observaciones_{region_name}_{dt}.png"
    make_plot(df, tuple(extent), dt, out, density_deg=density,
              isobaras=args.isobaras, iso_step=args.paso_isobaras,
              centros=args.centros)


if __name__ == "__main__":
    main()
