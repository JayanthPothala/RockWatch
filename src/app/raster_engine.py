"""
RockWatch Web Application - Geospatial Raster & GeoTIFF Engine
Processes raster datasets (DEM, Slope, Aspect, Rainfall, LULC), extracts metadata,
performs safe downsampled visualization, WGS84 reprojection, and point queries.
"""

from typing import Dict, Any, Optional, Tuple, List
import io
import base64
from pathlib import Path
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds, transform
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


HIMACHAL_BOUNDS_WGS84 = {
    "west": 74.5,
    "east": 81.0,
    "south": 28.5,
    "north": 34.5,
}


def ensure_sample_dem() -> Path:
    """
    Ensure a realistic sample GeoTIFF DEM for Himachal Pradesh exists in data/samples/.
    Returns the Path to the sample file.
    """
    sample_dir = Path("data/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "himachal_dem_sample.tif"

    if sample_path.exists() and sample_path.stat().st_size > 1000:
        return sample_path

    # Generate synthetic Himalayan DEM grid (150x120 pixels, ~2.5km resolution)
    ny, nx = 120, 150
    lon_min, lat_min, lon_max, lat_max = 75.5, 30.2, 79.2, 33.3
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    
    # Topography gradient: Shivalik foothills (600m) rising to Greater Himalaya (6,000m)
    elev = 600 + 4800 * (0.6 * xx + 0.4 * yy) + 450 * np.sin(xx * 12) * np.cos(yy * 10)
    elev = elev.astype(np.float32)

    transform_aff = from_bounds(lon_min, lat_min, lon_max, lat_max, nx, ny)

    with rasterio.open(
        str(sample_path),
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype=elev.dtype,
        crs="EPSG:4326",
        transform=transform_aff,
        nodata=-9999.0,
    ) as dst:
        dst.write(elev, 1)

    return sample_path


def open_raster_source(file_source):
    """
    Open a raster dataset from a file path, BytesIO, or Streamlit UploadedFile.
    """
    if hasattr(file_source, "read"):
        # UploadedFile or file-like object
        file_bytes = file_source.read()
        # Reset pointer for subsequent operations
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        return MemoryFile(file_bytes).open()
    elif isinstance(file_source, (str, Path)):
        return rasterio.open(str(file_source))
    elif isinstance(file_source, bytes):
        return MemoryFile(file_source).open()
    else:
        raise ValueError(f"Unsupported raster file source type: {type(file_source)}")


def extract_raster_metadata(file_source) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extract comprehensive metadata, coordinate bounds, CRS, and pixel statistics from a GeoTIFF.
    """
    try:
        with open_raster_source(file_source) as src:
            width = src.width
            height = src.height
            count = src.count
            dtypes = [str(d) for d in src.dtypes]
            nodata = src.nodata
            res = src.res

            crs = src.crs
            crs_str = crs.to_string() if crs else "Unspecified / Local Coordinates"
            epsg = crs.to_epsg() if crs else None

            native_bounds = {
                "left": float(src.bounds.left),
                "bottom": float(src.bounds.bottom),
                "right": float(src.bounds.right),
                "top": float(src.bounds.top),
            }

            # Reproject bounds to WGS84 (EPSG:4326) if possible
            wgs84_bounds = None
            is_himachal_intersect = False

            if crs:
                try:
                    w_bounds = transform_bounds(crs, CRS.from_epsg(4326), *src.bounds)
                    wgs84_bounds = {
                        "west": float(w_bounds[0]),
                        "south": float(w_bounds[1]),
                        "east": float(w_bounds[2]),
                        "north": float(w_bounds[3]),
                    }
                    # Check intersection with Himachal Pradesh bounding box
                    hp = HIMACHAL_BOUNDS_WGS84
                    is_himachal_intersect = not (
                        wgs84_bounds["east"] < hp["west"] or
                        wgs84_bounds["west"] > hp["east"] or
                        wgs84_bounds["north"] < hp["south"] or
                        wgs84_bounds["south"] > hp["north"]
                    )
                except Exception:
                    wgs84_bounds = native_bounds
            else:
                wgs84_bounds = native_bounds

            # Compute statistics for Band 1 with safe subsampling for large rasters
            sample_step = max(1, int(max(width, height) / 500))
            band1_sample = src.read(1, out_shape=(max(1, height // sample_step), max(1, width // sample_step)))

            if nodata is not None:
                valid_mask = (band1_sample != nodata) & np.isfinite(band1_sample)
            else:
                valid_mask = np.isfinite(band1_sample)

            valid_pixels = band1_sample[valid_mask]
            stats = {}
            if len(valid_pixels) > 0:
                stats = {
                    "min": float(np.min(valid_pixels)),
                    "max": float(np.max(valid_pixels)),
                    "mean": float(np.mean(valid_pixels)),
                    "std": float(np.std(valid_pixels)),
                    "valid_fraction": float(len(valid_pixels) / band1_sample.size),
                }
            else:
                stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "valid_fraction": 0.0}

            meta_info = {
                "width": width,
                "height": height,
                "bands": count,
                "dtypes": dtypes,
                "nodata": nodata,
                "resolution": (float(res[0]), float(res[1])),
                "crs_str": crs_str,
                "epsg": epsg,
                "native_bounds": native_bounds,
                "wgs84_bounds": wgs84_bounds,
                "is_himachal_intersect": is_himachal_intersect,
                "stats": stats,
            }
            return meta_info, None

    except Exception as e:
        return None, f"Raster inspection failed: {e}"


def generate_raster_overlay(
    file_source,
    band: int = 1,
    colormap_name: str = "terrain",
    max_dim: int = 600,
    opacity: float = 0.7,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate an RGBA image overlay georeferenced in WGS84 for direct integration into Plotly maps.
    Safely downsamples and handles reprojection to EPSG:4326.
    """
    try:
        with open_raster_source(file_source) as src:
            if band > src.count or band < 1:
                return None, f"Band {band} not found. File contains {src.count} band(s)."

            src_crs = src.crs
            dst_crs = CRS.from_epsg(4326)

            # Determine downsampling resolution
            scale_factor = max(1.0, max(src.width, src.height) / float(max_dim))
            target_w = max(10, int(src.width / scale_factor))
            target_h = max(10, int(src.height / scale_factor))

            if src_crs and src_crs != dst_crs:
                # Reproject into WGS84
                transform_dst, width_dst, height_dst = calculate_default_transform(
                    src_crs, dst_crs, target_w, target_h, *src.bounds
                )
                destination = np.zeros((height_dst, width_dst), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, band),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform_dst,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
                data = destination
                # Bounds in WGS84
                w_bounds = transform_bounds(src_crs, dst_crs, *src.bounds)
                west, south, east, north = w_bounds
            else:
                # Native WGS84 or unspecified: read decimated array
                data = src.read(band, out_shape=(target_h, target_w)).astype(np.float32)
                west, south, east, north = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top

            nodata = src.nodata
            if nodata is not None:
                mask = (data == nodata) | ~np.isfinite(data)
            else:
                mask = ~np.isfinite(data)

            valid_data = data[~mask]
            if len(valid_data) == 0:
                return None, "Raster layer contains only NoData or non-finite values."

            # Robust percentile clipping for visual contrast
            p_min, p_max = np.percentile(valid_data, [2, 98])
            diff = float(p_max - p_min)
            if diff < 1e-5:
                normalized = np.full_like(data, 0.5, dtype=np.float32)
            else:
                normalized = np.clip((data - p_min) / diff, 0.0, 1.0)

            # Apply colormap
            try:
                cmap = plt.get_cmap(colormap_name)
            except Exception:
                cmap = plt.get_cmap("terrain")

            rgba = cmap(normalized)  # shape (H, W, 4)
            # Apply alpha opacity and mask NoData
            rgba[..., 3] = np.where(mask, 0.0, opacity)

            # Encode to PNG Base64
            buf = io.BytesIO()
            plt.imsave(buf, rgba, format="png")
            buf.seek(0)
            b64_str = "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")

            # Mapbox / Map coordinates: [top-left, top-right, bottom-right, bottom-left]
            # Each is [longitude, latitude]
            coordinates = [
                [float(west), float(north)],  # top-left
                [float(east), float(north)],  # top-right
                [float(east), float(south)],  # bottom-right
                [float(west), float(south)],  # bottom-left
            ]

            return {
                "source": b64_str,
                "coordinates": coordinates,
                "bounds_wgs84": {"west": west, "south": south, "east": east, "north": north},
                "p_min": float(p_min),
                "p_max": float(p_max),
                "colormap": colormap_name,
                "opacity": opacity,
            }, None

    except Exception as e:
        return None, f"Failed to generate raster overlay: {e}"


def query_raster_pixel_at_coord(
    file_source,
    lat: float,
    lon: float,
    band: int = 1
) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract the exact pixel value at a given (lat, lon) in WGS84 coordinates.
    Handles reprojection of query point if raster uses a projected CRS.
    """
    try:
        with open_raster_source(file_source) as src:
            if band > src.count or band < 1:
                return None, f"Invalid band {band}."

            crs = src.crs
            x_query, y_query = lon, lat

            if crs and crs != CRS.from_epsg(4326):
                # Reproject query point from WGS84 to native CRS
                xs, ys = transform(CRS.from_epsg(4326), crs, [lon], [lat])
                x_query, y_query = xs[0], ys[0]

            # Check if query point is inside native bounds
            if not (src.bounds.left <= x_query <= src.bounds.right and src.bounds.bottom <= y_query <= src.bounds.top):
                return None, f"Coordinate ({lat:.4f}°N, {lon:.4f}°E) lies outside the raster boundary."

            # Get row, col in native grid
            row, col = src.index(x_query, y_query)
            window = rasterio.windows.Window(col, row, 1, 1)
            val = src.read(band, window=window)[0, 0]

            if src.nodata is not None and val == src.nodata:
                return None, "Selected location corresponds to a NoData pixel."

            return float(val), None

    except Exception as e:
        return None, f"Pixel query error: {e}"
