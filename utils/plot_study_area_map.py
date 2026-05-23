import argparse
import json
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


DATA_DIR = PROJECT_ROOT / "data" / "ST_FishNet_Features"
OUTPUT_DIR = PROJECT_ROOT / "model_outcomes" / "study_area"
ETOPO_CACHE_DIR = PROJECT_ROOT / "data" / "external" / "etopo2022"
NATURAL_EARTH_CACHE_DIR = PROJECT_ROOT / "data" / "external" / "natural_earth"

STUDY_LON_MIN, STUDY_LON_MAX = 118.0, 126.0
STUDY_LAT_MIN, STUDY_LAT_MAX = 23.0, 28.0
MAIN_EXTENT = (117.35, 126.65, 22.35, 28.65)
FIGURE_WIDTH_IN = 7.60
FIGURE_HEIGHT_IN = 5.35
MIN_BITMAP_DPI = 500
MIN_BITMAP_WIDTH_PX = 3740
EXPORT_PAD_IN = 0.015

ETOPO_BASE_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.csv"
NATURAL_EARTH_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
NATURAL_EARTH_FALLBACK_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
AIS_FILE = "ais_monthly_daymean_grid.npy"
LAND_MASK_FILE = "land_mask_0.1deg.npy"
LATS_FILE = "target_lats.npy"
LONS_FILE = "target_lons.npy"
TIME_FILE = "ais_time_all.npy"
ISOBATH_LEVELS = [50.0, 100.0, 200.0]
STUDY_RED = "#c0392b"
OCEAN_BLUE = "#eaf2f8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-ready study-area map with ETOPO 2022 "
            "50/100/200 m isobaths and 2012-2024 AIS fishing-effort climatology."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=ETOPO_CACHE_DIR)
    parser.add_argument("--natural-earth-cache-dir", type=Path, default=NATURAL_EARTH_CACHE_DIR)
    parser.add_argument("--main-stride", type=int, default=4, help="ETOPO 15s stride for main map.")
    parser.add_argument("--dpi", type=int, default=600, help="Bitmap export resolution; must be >= 500 dpi.")
    parser.add_argument("--vmax-quantile", type=float, default=99.0)
    parser.add_argument("--force-download", action="store_true", help="Refresh cached ETOPO subsets.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use cached ETOPO/Natural Earth subsets only; fail if any cache is missing.",
    )
    return parser.parse_args()


def fmt_token(value: float) -> str:
    text = f"{value:.2f}".replace("-", "m").replace(".", "p")
    return text.rstrip("0").rstrip("p")


def cache_path(cache_dir: Path, extent: Tuple[float, float, float, float], stride: int, tag: str) -> Path:
    lon_min, lon_max, lat_min, lat_max = extent
    name = (
        f"etopo2022_15s_{tag}_"
        f"lon{fmt_token(lon_min)}_{fmt_token(lon_max)}_"
        f"lat{fmt_token(lat_min)}_{fmt_token(lat_max)}_stride{stride}.npz"
    )
    return cache_dir / name


def build_etopo_url(extent: Tuple[float, float, float, float], stride: int) -> str:
    lon_min, lon_max, lat_min, lat_max = extent
    query = f"z[({lat_min}):{stride}:({lat_max})][({lon_min}):{stride}:({lon_max})]"
    return f"{ETOPO_BASE_URL}?{query}"


def download_etopo_subset(
    extent: Tuple[float, float, float, float],
    stride: int,
    cache_file: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    url = build_etopo_url(extent, stride)
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), skiprows=[1])
    required = {"latitude", "longitude", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected ETOPO response columns: {df.columns.tolist()}")

    lats = np.sort(df["latitude"].unique().astype(np.float64))
    lons = np.sort(df["longitude"].unique().astype(np.float64))
    grid = (
        df.pivot(index="latitude", columns="longitude", values="z")
        .reindex(index=lats, columns=lons)
        .to_numpy(dtype=np.float32)
    )
    if grid.shape != (len(lats), len(lons)):
        raise ValueError(f"ETOPO grid shape mismatch: {grid.shape} vs {(len(lats), len(lons))}")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, lats=lats, lons=lons, elevation_m=grid, source_url=url)
    return lats, lons, grid, url


def load_etopo_subset(
    cache_dir: Path,
    extent: Tuple[float, float, float, float],
    stride: int,
    tag: str,
    force_download: bool,
    no_download: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Path]:
    path = cache_path(cache_dir, extent, stride, tag)
    if path.exists() and not force_download:
        data = np.load(path, allow_pickle=False)
        source_url = str(data["source_url"]) if "source_url" in data.files else build_etopo_url(extent, stride)
        return data["lats"], data["lons"], data["elevation_m"], source_url, path

    if no_download:
        raise FileNotFoundError(
            f"ETOPO cache is missing and --no-download was supplied: {path}"
        )
    lats, lons, elev, source_url = download_etopo_subset(extent, stride, path)
    return lats, lons, elev, source_url, path


def natural_earth_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "ne_10m_admin_0_countries.geojson"


def load_natural_earth_geojson(
    cache_dir: Path,
    force_download: bool,
    no_download: bool,
) -> Tuple[Dict, str, Path]:
    path = natural_earth_cache_path(cache_dir)
    if path.exists() and not force_download:
        return json.loads(path.read_text(encoding="utf-8")), NATURAL_EARTH_URL, path

    fallback_path = cache_dir / "ne_50m_admin_0_countries.geojson"
    if no_download:
        if fallback_path.exists():
            return json.loads(fallback_path.read_text(encoding="utf-8")), NATURAL_EARTH_FALLBACK_URL, fallback_path
        raise FileNotFoundError(f"Natural Earth cache is missing and --no-download was supplied: {path}")

    try:
        response = requests.get(NATURAL_EARTH_URL, timeout=120)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        return response.json(), NATURAL_EARTH_URL, path
    except requests.RequestException:
        if fallback_path.exists():
            return json.loads(fallback_path.read_text(encoding="utf-8")), NATURAL_EARTH_FALLBACK_URL, fallback_path
        raise


def iter_geojson_rings(geojson: Dict) -> Iterable[Sequence[Sequence[float]]]:
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates", [])
        if geom_type == "Polygon":
            for ring in coords[:1]:
                yield ring
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon[:1]:
                    yield ring


def ring_intersects_extent(
    ring: Sequence[Sequence[float]],
    extent: Tuple[float, float, float, float],
    margin: float = 0.25,
) -> bool:
    lon_min, lon_max, lat_min, lat_max = extent
    arr = np.asarray(ring, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return False
    xs = arr[:, 0]
    ys = arr[:, 1]
    return (
        np.nanmax(xs) >= lon_min - margin
        and np.nanmin(xs) <= lon_max + margin
        and np.nanmax(ys) >= lat_min - margin
        and np.nanmin(ys) <= lat_max + margin
    )


def draw_natural_earth_land(
    ax,
    geojson: Dict,
    extent: Tuple[float, float, float, float],
    facecolor: str = "#d9d6ce",
    edgecolor: str = "#6f6b63",
    linewidth: float = 0.55,
    zorder: int = 6,
) -> None:
    for ring in iter_geojson_rings(geojson):
        if not ring_intersects_extent(ring, extent):
            continue
        arr = np.asarray(ring, dtype=np.float64)
        ax.fill(
            arr[:, 0],
            arr[:, 1],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )


def get_effort_colormap():
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fishres_deep_red",
        [
            (0.00, "#fceee6"),
            (0.12, "#fddbc7"),
            (0.30, "#fcae91"),
            (0.48, "#fb6a4a"),
            (0.64, "#de2d26"),
            (0.80, "#a50f15"),
            (1.00, "#67000d"),
        ],
        N=256,
    )
    cmap_name = "fishres_deep_red"
    cmap.set_bad((1, 1, 1, 0))
    return cmap, cmap_name


def load_ais_climatology(data_dir: Path) -> Dict[str, np.ndarray]:
    ais_path = data_dir / AIS_FILE
    mask_path = data_dir / LAND_MASK_FILE
    lats_path = data_dir / LATS_FILE
    lons_path = data_dir / LONS_FILE
    time_path = data_dir / TIME_FILE

    for path in [ais_path, mask_path, lats_path, lons_path, time_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required AIS/grid file is missing: {path}")

    ais = np.load(ais_path).astype(np.float32)
    land_mask = np.load(mask_path).astype(bool)
    lats = np.load(lats_path).astype(np.float64)
    lons = np.load(lons_path).astype(np.float64)
    times = np.load(time_path).astype("datetime64[M]")

    if ais.ndim != 3:
        raise ValueError(f"Expected AIS array [T,H,W], got {ais.shape}")
    if ais.shape[1:] != land_mask.shape:
        raise ValueError(f"AIS spatial shape {ais.shape[1:]} does not match mask {land_mask.shape}")
    if ais.shape[1] != len(lats) or ais.shape[2] != len(lons):
        raise ValueError(f"AIS shape {ais.shape} does not match lat/lon lengths {(len(lats), len(lons))}")

    time_mask = (times >= np.datetime64("2012-01")) & (times <= np.datetime64("2024-12"))
    if not np.any(time_mask):
        raise ValueError("No AIS months found within 2012-01 to 2024-12")

    climatology = np.nanmean(ais[time_mask], axis=0)
    climatology = np.where(land_mask, climatology, np.nan)
    climatology = np.ma.masked_invalid(climatology)

    return {
        "lats": lats,
        "lons": lons,
        "land_mask": land_mask,
        "climatology": climatology,
        "times": times[time_mask],
    }


def edges_from_regular_centers(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Coordinate array must be one-dimensional with at least two values")
    step = float(np.median(np.diff(values)))
    return np.concatenate([[values[0] - step / 2.0], values[:-1] + step / 2.0, [values[-1] + step / 2.0]])


def setup_geo_axes(
    ax,
    extent: Tuple[float, float, float, float],
    tick_step_lon: float,
    tick_step_lat: float,
    tick_label_size: float = 8.0,
    tick_pad: float = 2.0,
) -> None:
    lon_min, lon_max, lat_min, lat_max = extent
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    mean_lat = (lat_min + lat_max) / 2.0
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    ax.set_xticks(np.arange(np.ceil(lon_min / tick_step_lon) * tick_step_lon, lon_max + 0.01, tick_step_lon))
    ax.set_yticks(np.arange(np.ceil(lat_min / tick_step_lat) * tick_step_lat, lat_max + 0.01, tick_step_lat))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):.0f}$^\\circ$E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{abs(y):.0f}$^\\circ$N"))
    ax.tick_params(axis="both", labelsize=tick_label_size, length=3.5, width=0.8, pad=tick_pad)
    ax.grid(color="#9aa7b2", linewidth=0.45, linestyle=":", alpha=0.75, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def draw_etopo_land(ax, lons: np.ndarray, lats: np.ndarray, elev: np.ndarray, zorder: int = 3) -> None:
    max_elev = float(np.nanmax(elev))
    ax.contourf(
        lons,
        lats,
        elev,
        levels=[0.0, max(1.0, max_elev)],
        colors=["#d9d6ce"],
        zorder=zorder,
    )
    ax.contour(lons, lats, elev, levels=[0.0], colors="#6f6b63", linewidths=0.55, zorder=zorder + 1)


def segment_length_km(lons: np.ndarray, lats: np.ndarray) -> float:
    if len(lons) < 2 or len(lats) < 2:
        return 0.0
    mean_lat = np.nanmean(lats)
    dx = np.diff(lons) * 111.32 * np.cos(np.deg2rad(mean_lat))
    dy = np.diff(lats) * 111.32
    return float(np.nansum(np.hypot(dx, dy)))


def draw_filtered_isobaths(
    ax,
    lons: np.ndarray,
    lats: np.ndarray,
    elev: np.ndarray,
    levels: Sequence[float] = ISOBATH_LEVELS,
    min_length_km: float = 30.0,
    smooth_sigma: float = 1.0,
) -> Dict[float, int]:
    lon_mask = (lons >= STUDY_LON_MIN) & (lons <= STUDY_LON_MAX)
    lat_mask = (lats >= STUDY_LAT_MIN) & (lats <= STUDY_LAT_MAX)
    lons_sub = lons[lon_mask]
    lats_sub = lats[lat_mask]
    elev_sub = elev[np.ix_(lat_mask, lon_mask)]
    if elev_sub.size == 0:
        return {float(level): 0 for level in levels}

    depth = np.maximum(-elev_sub.astype(np.float64), 0.0)
    if smooth_sigma > 0:
        depth = gaussian_filter(depth, sigma=smooth_sigma, mode="nearest")

    style_by_level = {
        50.0: ("#2c7fb8", "--", 1.0),
        100.0: ("#045a8d", "-.", 1.05),
        200.0: ("#08306b", "-", 1.25),
    }
    drawn_counts: Dict[float, int] = {}
    row_index = np.arange(len(lats_sub), dtype=np.float64)
    col_index = np.arange(len(lons_sub), dtype=np.float64)

    for level in levels:
        color, linestyle, linewidth = style_by_level.get(float(level), ("#045a8d", "-", 1.0))
        count = 0
        for contour in find_contours(depth, level=float(level), fully_connected="high"):
            if contour.shape[0] < 2:
                continue
            contour_lats = np.interp(contour[:, 0], row_index, lats_sub)
            contour_lons = np.interp(contour[:, 1], col_index, lons_sub)
            if segment_length_km(contour_lons, contour_lats) < min_length_km:
                continue
            ax.plot(
                contour_lons,
                contour_lats,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=5,
            )
            count += 1
        drawn_counts[float(level)] = count
    return drawn_counts


def draw_study_rectangle(ax, linewidth: float = 1.45, zorder: int = 8, label: str = None) -> Rectangle:
    rect = Rectangle(
        (STUDY_LON_MIN, STUDY_LAT_MIN),
        STUDY_LON_MAX - STUDY_LON_MIN,
        STUDY_LAT_MAX - STUDY_LAT_MIN,
        fill=False,
        edgecolor=STUDY_RED,
        linewidth=linewidth,
        zorder=zorder,
        label=label,
    )
    ax.add_patch(rect)
    return rect


def draw_scale_bar(ax) -> None:
    lat = 23.45
    center_lon = 125.0
    km = 200.0
    deg_lon = km / (111.32 * np.cos(np.deg2rad(lat)))
    lon0 = center_lon - deg_lon / 2.0
    lon1 = lon0 + deg_lon
    ax.add_patch(
        Rectangle(
            (lon0 - 0.14, lat - 0.12),
            deg_lon + 0.28,
            0.43,
            facecolor="white",
            edgecolor="none",
            linewidth=0,
            alpha=0.72,
            zorder=8.8,
        )
    )
    ax.plot([lon0, lon1], [lat, lat], color="#1f2937", linewidth=1.8, zorder=9)
    ax.plot([lon0, lon0], [lat - 0.035, lat + 0.035], color="#1f2937", linewidth=1.4, zorder=9)
    ax.plot([lon1, lon1], [lat - 0.035, lat + 0.035], color="#1f2937", linewidth=1.4, zorder=9)
    ax.text((lon0 + lon1) / 2, lat + 0.09, "200 km", ha="center", va="bottom", fontsize=8.1, color="#1f2937", zorder=9)


def draw_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(126.32, 27.92),
        xytext=(126.32, 27.40),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#1f2937",
        arrowprops=dict(arrowstyle="-|>", lw=1.15, color="#1f2937", shrinkA=0, shrinkB=0),
        zorder=9,
    )


def get_tight_export_bbox(fig, requested_dpi: int) -> Tuple[Bbox, int, int, int]:
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(EXPORT_PAD_IN)
    export_dpi = int(requested_dpi)
    bitmap_width_px = int(round(bbox.width * export_dpi))
    if bitmap_width_px < MIN_BITMAP_WIDTH_PX:
        export_dpi = int(np.ceil(MIN_BITMAP_WIDTH_PX / bbox.width))
        bitmap_width_px = int(round(bbox.width * export_dpi))
        print(
            f"Increased bitmap export DPI from {requested_dpi} to {export_dpi} "
            f"to keep tight-cropped width >= {MIN_BITMAP_WIDTH_PX}px."
        )
    bitmap_height_px = int(round(bbox.height * export_dpi))
    if bitmap_width_px < MIN_BITMAP_WIDTH_PX:
        raise ValueError(
            f"Tight-cropped bitmap export would be {bitmap_width_px}px wide; "
            f"expected at least {MIN_BITMAP_WIDTH_PX}px"
        )
    return bbox, export_dpi, bitmap_width_px, bitmap_height_px


def create_figure(
    ais: Dict[str, np.ndarray],
    main_etopo: Tuple[np.ndarray, np.ndarray, np.ndarray],
    natural_earth: Dict,
    data_dir: Path,
    output_dir: Path,
    dpi: int,
    vmax_quantile: float,
    metadata: Dict,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    main_lats, main_lons, main_elev = main_etopo

    clim = ais["climatology"]
    valid_values = np.asarray(clim.compressed(), dtype=np.float32)
    valid_values = valid_values[valid_values > 0]
    if valid_values.size == 0:
        raise ValueError("AIS climatology contains no positive ocean cells")
    vmax = float(np.nanpercentile(valid_values, vmax_quantile))
    vmax = max(vmax, 1.0)
    log_clim = np.ma.log10(np.ma.clip(clim, 0.0, vmax) + 1.0)
    log_vmax = float(np.log10(vmax + 1.0))

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(OCEAN_BLUE)

    setup_geo_axes(ax, MAIN_EXTENT, tick_step_lon=2.0, tick_step_lat=1.0, tick_label_size=7.8)
    ax.set_xlabel("")
    ax.set_ylabel("")

    lon_edges = edges_from_regular_centers(ais["lons"])
    lat_edges = edges_from_regular_centers(ais["lats"])
    cmap, cmap_name = get_effort_colormap()
    norm = mcolors.Normalize(vmin=0.0, vmax=log_vmax)
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        log_clim,
        cmap=cmap,
        norm=norm,
        shading="flat",
        zorder=2,
    )

    isobath_counts = draw_filtered_isobaths(ax, main_lons, main_lats, main_elev)

    draw_etopo_land(ax, main_lons, main_lats, main_elev, zorder=6)
    draw_natural_earth_land(ax, natural_earth, MAIN_EXTENT, linewidth=0.55, zorder=7)
    draw_study_rectangle(ax, linewidth=1.55, zorder=9)
    draw_scale_bar(ax)
    draw_north_arrow(ax)

    ax.text(121.0, 23.85, "Taiwan", fontsize=9.2, color="#3f3a33", rotation=72, fontstyle="italic", zorder=10)
    ax.text(123.95, 26.55, "East China Sea", fontsize=9.2, color="#1f425a", fontstyle="italic", zorder=10)
    legend_items = [
        Line2D([0], [0], color=STUDY_RED, lw=1.55, label="Study boundary"),
        Line2D([0], [0], color="#2c7fb8", lw=1.05, linestyle="--", label="50 m isobath"),
        Line2D([0], [0], color="#045a8d", lw=1.05, linestyle="-.", label="100 m isobath"),
        Line2D([0], [0], color="#08306b", lw=1.25, linestyle="-", label="200 m isobath"),
    ]
    leg = ax.legend(
        handles=legend_items,
        loc="lower left",
        bbox_to_anchor=(0.075, 0.12),
        ncol=1,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#cbd5e1",
        fontsize=5.4,
        handlelength=2.35,
        borderpad=0.45,
        labelspacing=0.33,
    )
    leg.get_frame().set_linewidth(0.6)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.052, pad=0.062, extend="neither")
    cbar.set_label(
        "Mean AIS apparent fishing effort (h d$^{-1}$ grid$^{-1}$), 2012–2024, on a log$_{10}$(x + 1) colour scale",
        fontsize=7.7,
    )
    physical_ticks = np.array([0, 1, 5, 10], dtype=float)
    physical_ticks = physical_ticks[physical_ticks <= vmax]
    if physical_ticks[-1] < vmax and vmax < 40:
        physical_ticks = np.append(physical_ticks, round(vmax, 1))
    cbar.set_ticks(np.log10(physical_ticks + 1.0))
    cbar_labels = [f"{tick:g}" for tick in physical_ticks]
    cbar_labels[-1] = f"{physical_ticks[-1]:.1f} h d$^{{-1}}$"
    cbar.set_ticklabels(cbar_labels)
    cbar.ax.tick_params(labelsize=7.5, length=3)
    cbar.outline.set_linewidth(0.7)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.135)

    stem = "Fig1_study_area_climatology"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    tif_path = output_dir / f"{stem}.tif"
    export_bbox, export_dpi, bitmap_width_px, bitmap_height_px = get_tight_export_bbox(fig, dpi)
    fig.savefig(png_path, dpi=export_dpi, bbox_inches=export_bbox)
    fig.savefig(pdf_path, bbox_inches=export_bbox)
    fig.savefig(tif_path, dpi=export_dpi, bbox_inches=export_bbox)
    plt.close(fig)

    metadata.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "figure_files": {
                "png": str(png_path),
                "pdf": str(pdf_path),
                "tif": str(tif_path),
            },
            "export": {
                "pdf": "vector PDF",
                "requested_bitmap_dpi": dpi,
                "bitmap_dpi": export_dpi,
                "figure_width_in": FIGURE_WIDTH_IN,
                "figure_height_in": FIGURE_HEIGHT_IN,
                "export_padding_in": EXPORT_PAD_IN,
                "tight_crop": True,
                "bitmap_width_px": bitmap_width_px,
                "bitmap_height_px": bitmap_height_px,
                "minimum_bitmap_dpi": MIN_BITMAP_DPI,
                "minimum_bitmap_width_px": MIN_BITMAP_WIDTH_PX,
                "location_overview_map": "removed",
            },
            "study_extent": {
                "lon_min": STUDY_LON_MIN,
                "lon_max": STUDY_LON_MAX,
                "lat_min": STUDY_LAT_MIN,
                "lat_max": STUDY_LAT_MAX,
            },
            "main_extent": {
                "lon_min": MAIN_EXTENT[0],
                "lon_max": MAIN_EXTENT[1],
                "lat_min": MAIN_EXTENT[2],
                "lat_max": MAIN_EXTENT[3],
            },
            "ais_climatology": {
                "source_file": str(data_dir / AIS_FILE),
                "mask_file": str(data_dir / LAND_MASK_FILE),
                "time_range": "2012-01 to 2024-12",
                "quantity": "monthly mean of AIS-derived apparent fishing effort, expressed as hours per day per 0.1-degree grid",
                "vmax_quantile": vmax_quantile,
                "vmax_value": vmax,
                "colour_transform": "log10(x + 1), with x clipped at the selected climatology quantile for display",
                "colormap": cmap_name,
                "months_averaged": int(len(ais["times"])),
                "white_cells": "Cells excluded by the land/common-ocean mask are transparent over the map background.",
                "isobath_segment_filter_min_length_km": 30.0,
                "isobath_segments_drawn": {str(k): int(v) for k, v in isobath_counts.items()},
            },
        }
    )
    metadata_path = output_dir / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "tif": str(tif_path),
        "metadata": str(metadata_path),
    }


def write_caption(output_dir: Path, natural_earth_resolution: str) -> Path:
    caption = (
        "Fig. 1. Study area in the East China Sea and Taiwan Strait. "
        "The red box denotes the modelling domain (23.0–28.0°N, 118.0–126.0°E). "
        "Shading shows the climatological mean AIS-derived apparent fishing effort "
        "(h d⁻¹ grid⁻¹), averaged over 156 months from January 2012 to December 2024. "
        "The colour scale uses log10(x + 1) scaling and is clipped at the 99th percentile "
        "for display. Dashed, dash-dotted and solid blue contours indicate the 50 m, 100 m "
        "and 200 m isobaths from NOAA/NCEI ETOPO 2022, respectively. Pale/white cells "
        "indicate zero to very low apparent fishing effort or cells excluded by the common "
        "land/ocean mask, rather than interpolated values. Local linear high-value patches "
        "reflect recurrent AIS-classified fishing tracks retained in the apparent fishing "
        "effort field; in particular, the local north-south high-value band at approximately "
        "121°–122°E corresponds to stable routes used by fleets to and from the main fishing "
        "grounds and was retained as a real activity signal. Coastline and country boundaries "
        f"use Natural Earth {natural_earth_resolution} data."
    )
    path = output_dir / "Fig1_study_area_climatology_caption.txt"
    path.write_text(caption, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.dpi < MIN_BITMAP_DPI:
        raise ValueError(f"--dpi must be at least {MIN_BITMAP_DPI} for publication bitmap export")
    bitmap_width_px = int(round(FIGURE_WIDTH_IN * args.dpi))
    if bitmap_width_px < MIN_BITMAP_WIDTH_PX:
        raise ValueError(
            f"Bitmap export would be {bitmap_width_px}px wide; expected at least {MIN_BITMAP_WIDTH_PX}px"
        )

    ais = load_ais_climatology(args.data_dir)
    main_lats, main_lons, main_elev, main_url, main_cache = load_etopo_subset(
        args.cache_dir,
        MAIN_EXTENT,
        args.main_stride,
        "main",
        force_download=args.force_download,
        no_download=args.no_download,
    )
    natural_earth, natural_earth_url, natural_earth_cache = load_natural_earth_geojson(
        args.natural_earth_cache_dir,
        force_download=args.force_download,
        no_download=args.no_download,
    )

    natural_earth_resolution = "1:10m" if "10m" in natural_earth_cache.name else "1:50m"
    metadata = {
        "etopo": {
            "dataset": "NOAA/NCEI ETOPO 2022, 15 arc-second global relief model via ERDDAP",
            "main_url": main_url,
            "main_cache": str(main_cache),
            "main_stride": args.main_stride,
            "elevation_units": "m; negative values are bathymetric depth below mean sea level",
        },
        "natural_earth": {
            "dataset": f"Natural Earth {natural_earth_resolution} Admin 0 countries",
            "url": natural_earth_url,
            "cache": str(natural_earth_cache),
        }
    }

    paths = create_figure(
        ais=ais,
        main_etopo=(main_lats, main_lons, main_elev),
        natural_earth=natural_earth,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        vmax_quantile=args.vmax_quantile,
        metadata=metadata,
    )
    caption_path = write_caption(args.output_dir, natural_earth_resolution)

    print("Created study-area figure:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    print(f"  caption: {caption_path}")


if __name__ == "__main__":
    main()
