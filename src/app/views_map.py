"""
RockWatch Web Application - Susceptibility Map View
Interactive geospatial visualization of 11,737 surveyed locations across Himachal Pradesh.
"""

import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.app.styles import get_metric_card
from src.app.raster_engine import (
    ensure_sample_dem,
    extract_raster_metadata,
    generate_raster_overlay,
    query_raster_pixel_at_coord,
)


REQUIRED_COLUMNS = [
    "sample_id",
    "latitude",
    "longitude",
    "class",
    "susceptibility_score",
    "susceptibility_class",
    "elevation_m",
    "slope_deg",
]

TIER_COLOR_MAP = {
    "Low": "#10b981",       # Emerald Green
    "Moderate": "#f59e0b",  # Amber Yellow
    "High": "#f97316",      # Orange
    "Very High": "#ef4444",  # Crimson Red
}


@st.cache_data
def load_map_data():
    """
    Load pre-computed susceptibility predictions for Himachal Pradesh.
    Attempts primary CSV load, falls back to GeoJSON if necessary, and validates data.
    """
    csv_path = Path("data/processed/himachal_susceptibility_predictions.csv")
    geojson_path = Path("data/processed/himachal_susceptibility_points.geojson")

    df = None
    load_source = None

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            load_source = str(csv_path)
        except Exception as e:
            st.warning(f"Could not read {csv_path}: {e}. Attempting GeoJSON fallback.")

    if df is None and geojson_path.exists():
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                gj = json.load(f)
            features = gj.get("features", [])
            records = []
            for feat in features:
                props = feat.get("properties", {})
                coords = feat.get("geometry", {}).get("coordinates", [None, None])
                record = {
                    "sample_id": props.get("sample_id"),
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "class": props.get("ground_truth_class", props.get("class", 0)),
                    "susceptibility_score": props.get("susceptibility_score"),
                    "susceptibility_class": props.get("susceptibility_class"),
                    "elevation_m": props.get("elevation_m", 0),
                    "slope_deg": props.get("slope_deg", 0),
                }
                records.append(record)
            df = pd.DataFrame(records)
            load_source = str(geojson_path)
        except Exception as e:
            st.error(f"Failed to load fallback GeoJSON from {geojson_path}: {e}")
            return None, f"Error: {e}"

    if df is None:
        return None, "Error: Susceptibility prediction files not found."

    # Validate required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return None, f"Error: Missing required columns in dataset: {missing_cols}"

    # Drop missing / non-finite coordinates
    df = df.dropna(subset=["latitude", "longitude", "susceptibility_score", "susceptibility_class"])
    df = df[np.isfinite(df["latitude"]) & np.isfinite(df["longitude"]) & np.isfinite(df["susceptibility_score"])]

    # Validate coordinate ranges for Himachal Pradesh (approx lat 29.5-33.5, lon 75.0-80.5)
    valid_coord_mask = (
        (df["latitude"] >= 28.5) & (df["latitude"] <= 34.5) &
        (df["longitude"] >= 74.5) & (df["longitude"] <= 81.0)
    )
    df = df[valid_coord_mask].copy()

    # Ensure susceptibility scores are in [0, 1]
    df["susceptibility_score"] = df["susceptibility_score"].clip(0.0, 1.0)

    # Add human-readable ground truth label
    df["ground_truth_label"] = df["class"].map({1: "Landslide Event", 0: "Pseudo-Absence"}).fillna("Unknown")

    return df, f"Loaded {len(df):,} records from {load_source}"


def create_scatter_map(plot_df, map_style="open-street-map", height=620, raster_layer=None):
    """
    Construct a responsive scatter map using Plotly's modern map engine.
    Compatible with both Plotly 7.x (px.scatter_map) and legacy Plotly (px.scatter_mapbox).
    Supports optional georeferenced raster image layer overlays (e.g. DEM, slope, aspect).
    """
    hover_config = {
        "sample_id": True,
        "latitude": ":.4f",
        "longitude": ":.4f",
        "elevation_m": ":.0f",
        "slope_deg": ":.1f",
        "susceptibility_score": ":.3f",
        "ground_truth_label": True,
        "susceptibility_class": False,
    }

    # Modern Plotly 6.0+ / 7.x uses px.scatter_map with map_style
    if hasattr(px, "scatter_map"):
        fig_map = px.scatter_map(
            plot_df,
            lat="latitude",
            lon="longitude",
            color="susceptibility_class",
            color_discrete_map=TIER_COLOR_MAP,
            category_orders={"susceptibility_class": ["Low", "Moderate", "High", "Very High"]},
            hover_name="susceptibility_class",
            hover_data=hover_config,
            zoom=6.8,
            center={"lat": 31.85, "lon": 77.20},
            map_style=map_style,
            height=height,
        )
    # Fallback for older Plotly versions (<6.0)
    elif hasattr(px, "scatter_mapbox"):
        fig_map = px.scatter_mapbox(
            plot_df,
            lat="latitude",
            lon="longitude",
            color="susceptibility_class",
            color_discrete_map=TIER_COLOR_MAP,
            category_orders={"susceptibility_class": ["Low", "Moderate", "High", "Very High"]},
            hover_name="susceptibility_class",
            hover_data=hover_config,
            zoom=6.8,
            center={"lat": 31.85, "lon": 77.20},
            mapbox_style=map_style,
            height=height,
        )
    # Ultimate fallback using go.Figure with Scattermap or Scattergeo
    else:
        fig_map = go.Figure()
        for tier, color in TIER_COLOR_MAP.items():
            sub = plot_df[plot_df["susceptibility_class"] == tier]
            if len(sub) > 0:
                fig_map.add_trace(
                    go.Scattergeo(
                        lon=sub["longitude"],
                        lat=sub["latitude"],
                        text=sub["susceptibility_class"],
                        name=tier,
                        marker=dict(size=6, color=color, opacity=0.8),
                    )
                )
        fig_map.update_layout(
            geo=dict(scope="asia", center=dict(lat=31.85, lon=77.20), projection_scale=15),
            height=height,
        )
        return fig_map

    # Apply raster image layer overlay underneath the point scatter traces
    if raster_layer and raster_layer.get("visible", True) and raster_layer.get("source"):
        layer_dict = {
            "sourcetype": "image",
            "source": raster_layer["source"],
            "coordinates": raster_layer["coordinates"],
            "opacity": float(raster_layer.get("opacity", 0.65)),
        }
        if hasattr(px, "scatter_map"):
            fig_map.update_layout(map=dict(layers=[layer_dict]))
        elif hasattr(px, "scatter_mapbox"):
            fig_map.update_layout(mapbox=dict(layers=[layer_dict]))

    fig_map.update_traces(marker=dict(size=6, opacity=0.82))
    fig_map.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            title=dict(text="<b>Susceptibility Tier</b>", font=dict(size=12)),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#cbd5e1",
            borderwidth=1,
        ),
    )
    return fig_map


def render_map_view():
    """Render the Interactive Susceptibility Map page."""
    st.markdown(
        """
        <div class="rock-brand-banner">
            <span class="rock-banner-badge">🗺️ INTERACTIVE GEOSPATIAL MAP</span>
            <h1 class="rock-banner-title">Himachal Pradesh Susceptibility Atlas</h1>
            <p class="rock-banner-subtitle">
                Explore 11,737 field-verified and remote-sensed evaluation coordinates across the rugged terrain
                of Himachal Pradesh, classified into 4 hazard tiers by the champion Gradient Boosting model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df, status_msg = load_map_data()

    if df is None or len(df) == 0:
        st.error(
            f"❌ Unable to load susceptibility data for the map.\n\n"
            f"**Details:** {status_msg}\n\n"
            f"Please ensure `data/processed/himachal_susceptibility_predictions.csv` or "
            f"`data/processed/himachal_susceptibility_points.geojson` is present and valid."
        )
        return

    # Filter Controls inside an expandable container / columns
    with st.expander("🛠️ Filter & Display Controls", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)

        with fcol1:
            tier_options = ["Low", "Moderate", "High", "Very High"]
            selected_tiers = st.multiselect(
                "Susceptibility Tiers:",
                options=tier_options,
                default=tier_options,
                help="Filter points based on the model's predicted susceptibility hazard tier.",
            )

            ground_truth_filter = st.radio(
                "Ground Truth Category:",
                options=["All Records", "Landslide Events (Class 1)", "Non-Landslides (Class 0)"],
                index=0,
                horizontal=True,
            )

        with fcol2:
            min_elev = int(df["elevation_m"].min())
            max_elev = int(df["elevation_m"].max())
            elev_range = st.slider(
                "Elevation Range (meters):",
                min_value=min_elev,
                max_value=max_elev,
                value=(min_elev, max_elev),
                step=50,
            )

            min_slope = float(df["slope_deg"].min())
            max_slope = float(df["slope_deg"].max())
            slope_range = st.slider(
                "Slope Angle (°):",
                min_value=0.0,
                max_value=float(np.ceil(max_slope)),
                value=(0.0, float(np.ceil(max_slope))),
                step=1.0,
            )

        with fcol3:
            map_style = st.selectbox(
                "Base Map Style:",
                options=["open-street-map", "carto-positron", "carto-darkmatter"],
                index=0,
                help="Choose background tile style. OpenStreetMap is optimized for mountain topography.",
            )

            max_pts = st.slider(
                "Max Data Points to Render:",
                min_value=1000,
                max_value=len(df),
                value=min(6000, len(df)),
                step=500,
                help="Rendering fewer points ensures high responsiveness on standard browsers.",
            )

    # -------------------------------------------------------------
    # GeoTIFF / Environmental Raster Layer Management
    # -------------------------------------------------------------
    raster_overlay = None
    with st.expander("🛰️ Environmental GeoTIFF & Raster Overlays (DEM, Slope, Aspect, LULC)", expanded=False):
        st.markdown(
            """
            <p style="color: #64748b; font-size: 0.88rem; margin-bottom: 0.8rem;">
                Upload or preview spatial raster layers (such as SRTM Digital Elevation Models, Slope grids,
                or Monsoon Precipitation rasters) to overlay environmental context directly beneath the susceptibility predictions.
            </p>
            """,
            unsafe_allow_html=True,
        )

        rcol1, rcol2 = st.columns([2, 1])
        with rcol1:
            uploaded_tif = st.file_uploader(
                "Upload GeoTIFF Raster (.tif, .tiff):",
                type=["tif", "tiff"],
                key="raster_file_uploader",
                help="Select a valid single- or multi-band GeoTIFF file. Large files are safely subsampled for fluid web rendering.",
            )
        with rcol2:
            st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
            load_sample = st.button("⚡ Load Sample Himachal DEM", use_container_width=True, help="Load a pre-configured 2.5km synthetic Himalayan DEM raster covering Himachal Pradesh.")

        active_raster_source = None
        raster_source_label = None

        if uploaded_tif is not None:
            active_raster_source = uploaded_tif
            raster_source_label = f"Uploaded File: {uploaded_tif.name}"
        elif load_sample or st.session_state.get("sample_raster_active", False):
            st.session_state["sample_raster_active"] = True
            active_raster_source = ensure_sample_dem()
            raster_source_label = "Sample Dataset: Himachal Pradesh Topographic DEM (SRTM Baseline)"

        if active_raster_source is not None:
            # Extract metadata
            meta, meta_err = extract_raster_metadata(active_raster_source)
            if meta_err:
                st.error(f"❌ Could not process GeoTIFF: {meta_err}")
            elif meta:
                st.success(f"✅ Active Raster Layer: **{raster_source_label}**")

                # Controls Row
                ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
                with ctrl1:
                    show_raster = st.checkbox("Enable Map Layer Overlay", value=True, key="chk_raster_visible")
                with ctrl2:
                    raster_opacity = st.slider("Layer Opacity:", min_value=0.1, max_value=1.0, value=0.65, step=0.05, key="sld_raster_opacity")
                with ctrl3:
                    raster_cmap = st.selectbox(
                        "Colormap Palette:",
                        options=["terrain", "viridis", "turbo", "plasma", "magma", "cividis", "gist_earth"],
                        index=0,
                        key="sel_raster_cmap",
                    )
                with ctrl4:
                    band_options = list(range(1, meta["bands"] + 1))
                    selected_band = st.selectbox("Raster Band:", options=band_options, index=0, key="sel_raster_band")

                # Raster Metadata Card
                st.markdown("##### 📋 Raster Spatial Metadata")
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                with m_c1:
                    st.metric("Grid Dimensions", f"{meta['width']} × {meta['height']} px", f"{meta['bands']} band(s)")
                with m_c2:
                    crs_display = f"EPSG:{meta['epsg']}" if meta["epsg"] else meta["crs_str"][:18]
                    st.metric("CRS / Projection", crs_display, "WGS84 reprojected" if meta["epsg"] != 4326 else "Native WGS84")
                with m_c3:
                    res_str = f"{meta['resolution'][0]:.4f}°, {meta['resolution'][1]:.4f}°" if meta["epsg"] == 4326 else f"{meta['resolution'][0]:.1f}m, {meta['resolution'][1]:.1f}m"
                    st.metric("Spatial Resolution", res_str, f"NoData: {meta['nodata']}")
                with m_c4:
                    st.metric("Value Range", f"{meta['stats']['min']:.1f} to {meta['stats']['max']:.1f}", f"Mean: {meta['stats']['mean']:.1f}")

                # Geographic Bounds Information
                w84 = meta["wgs84_bounds"]
                st.caption(
                    f"🌐 **Geographic Extent (WGS84):** West: `{w84['west']:.4f}°E`, East: `{w84['east']:.4f}°E`, "
                    f"South: `{w84['south']:.4f}°N`, North: `{w84['north']:.4f}°N` · "
                    f"**Himachal Regional Overlap:** {'✅ Confirmed' if meta['is_himachal_intersect'] else '⚠️ Out of State Bounds'}"
                )

                # Selected-Location Pixel Inspector Tool
                st.markdown("##### 📍 Query Raster Pixel Value at Specific Coordinate")
                p_c1, p_c2, p_c3 = st.columns([1.5, 1.5, 1])
                with p_c1:
                    query_lat = st.number_input("Latitude (°N):", min_value=28.0, max_value=35.0, value=31.1048, step=0.01, format="%.4f")
                with p_c2:
                    query_lon = st.number_input("Longitude (°E):", min_value=74.0, max_value=82.0, value=77.1734, step=0.01, format="%.4f")
                with p_c3:
                    st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
                    inspect_btn = st.button("🔍 Sample Pixel", use_container_width=True)

                if inspect_btn:
                    pixel_val, p_err = query_raster_pixel_at_coord(active_raster_source, query_lat, query_lon, band=selected_band)
                    if p_err:
                        st.warning(f"Query Result: {p_err}")
                    else:
                        st.info(f"🎯 **Pixel Value at ({query_lat:.4f}°N, {query_lon:.4f}°E):** `{pixel_val:.2f}` (Band {selected_band})")

                # Generate the overlay
                if show_raster:
                    overlay_dict, o_err = generate_raster_overlay(
                        active_raster_source,
                        band=selected_band,
                        colormap_name=raster_cmap,
                        max_dim=600,
                        opacity=raster_opacity,
                    )
                    if o_err:
                        st.warning(f"Could not render raster overlay: {o_err}")
                    else:
                        raster_overlay = overlay_dict

        # Scientific separation reminder
        st.markdown(
            """
            <div style="background: #f8fafc; border-left: 3px solid #64748b; padding: 0.6rem 0.9rem; border-radius: 4px; font-size: 0.8rem; color: #475569; margin-top: 0.8rem;">
                ⚠️ <strong>Scientific Requirement:</strong> Spatial raster overlays are provided strictly for geomorphic visualization and environmental context.
                Raster pixel values are <em>not</em> automatically passed into the point-based machine learning model inference engine.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Handle case where user unselects all tiers
    if not selected_tiers:
        st.warning("⚠️ No susceptibility tiers selected. Please select at least one tier (Low, Moderate, High, or Very High) in the filter above.")
        return

    # Apply Filters
    filtered_df = df[df["susceptibility_class"].isin(selected_tiers)]
    filtered_df = filtered_df[
        (filtered_df["elevation_m"] >= elev_range[0]) & (filtered_df["elevation_m"] <= elev_range[1])
    ]
    filtered_df = filtered_df[
        (filtered_df["slope_deg"] >= slope_range[0]) & (filtered_df["slope_deg"] <= slope_range[1])
    ]

    if ground_truth_filter == "Landslide Events (Class 1)":
        filtered_df = filtered_df[filtered_df["class"] == 1]
    elif ground_truth_filter == "Non-Landslides (Class 0)":
        filtered_df = filtered_df[filtered_df["class"] == 0]

    n_filtered = len(filtered_df)

    if n_filtered == 0:
        st.warning("⚠️ No geographical coordinates match the selected filter criteria. Please broaden your elevation, slope, or category filters.")
        return

    # Subsample if necessary for map rendering speed
    is_sampled = n_filtered > max_pts
    if is_sampled:
        plot_df = filtered_df.sample(n=max_pts, random_state=42)
        subsample_note = f"Displaying {len(plot_df):,} sampled locations out of {n_filtered:,} matching criteria (sampled for fluid browser performance)."
    else:
        plot_df = filtered_df
        subsample_note = f"Displaying all {n_filtered:,} matching locations without sampling."

    # Dynamic KPI Summary Cards
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(
            get_metric_card(
                "Visible Coordinates",
                f"{n_filtered:,}",
                f"{n_filtered / len(df) * 100:.1f}% of state dataset",
            ),
            unsafe_allow_html=True,
        )
    with mcol2:
        high_risk_count = (filtered_df["susceptibility_class"].isin(["High", "Very High"])).sum()
        pct_high = (high_risk_count / n_filtered * 100) if n_filtered > 0 else 0
        st.markdown(
            get_metric_card(
                "High & Very High Risk",
                f"{pct_high:.1f}%",
                f"{high_risk_count:,} critical locations",
            ),
            unsafe_allow_html=True,
        )
    with mcol3:
        avg_score = filtered_df["susceptibility_score"].mean() if n_filtered > 0 else 0
        st.markdown(
            get_metric_card(
                "Mean Susceptibility Score",
                f"{avg_score:.3f}",
                "Model probability scale [0, 1]",
            ),
            unsafe_allow_html=True,
        )
    with mcol4:
        avg_elev = filtered_df["elevation_m"].mean() if n_filtered > 0 else 0
        avg_slope = filtered_df["slope_deg"].mean() if n_filtered > 0 else 0
        st.markdown(
            get_metric_card(
                "Mean Terrain Profile",
                f"{avg_elev:.0f} m",
                f"Average slope: {avg_slope:.1f}°",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(f"<p style='color: #475569; font-size: 0.85rem; margin-top: 0.5rem;'>📍 <strong>Data Status:</strong> {subsample_note}</p>", unsafe_allow_html=True)

    # Render Scatter Map
    try:
        fig_map = create_scatter_map(plot_df, map_style=map_style, height=620, raster_layer=raster_overlay)
        st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error rendering geospatial map: {e}")
        st.info("Falling back to table view:")
        st.dataframe(plot_df[["latitude", "longitude", "susceptibility_class", "susceptibility_score", "elevation_m", "slope_deg"]].head(100))

    # Breakdown distribution charts below map
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    bcol1, bcol2 = st.columns(2)

    with bcol1:
        st.markdown("#### 📊 Hazard Tier Distribution (Filtered)")
        tier_counts = filtered_df["susceptibility_class"].value_counts().reindex(["Low", "Moderate", "High", "Very High"]).fillna(0)
        fig_bar = px.bar(
            x=tier_counts.index,
            y=tier_counts.values,
            color=tier_counts.index,
            color_discrete_map=TIER_COLOR_MAP,
            labels={"x": "Susceptibility Tier", "y": "Number of Coordinates"},
            text=[f"{int(v):,} ({v/n_filtered*100:.1f}%)" if n_filtered > 0 else "0" for v in tier_counts.values],
        )
        fig_bar.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    with bcol2:
        st.markdown("#### 🏔️ Susceptibility by Elevation Bands")
        # Elevation bins
        bins = [0, 1000, 2000, 3000, 4000, 6000]
        labels = ["<1000m", "1000-2000m", "2000-3000m", "3000-4000m", ">4000m"]
        filtered_df = filtered_df.copy()
        filtered_df["elev_band"] = pd.cut(filtered_df["elevation_m"], bins=bins, labels=labels, right=False)

        elev_tier_dist = pd.crosstab(
            filtered_df["elev_band"],
            filtered_df["susceptibility_class"],
            normalize="index"
        ).reindex(columns=["Low", "Moderate", "High", "Very High"]).fillna(0) * 100

        fig_elev = go.Figure()
        for tier in ["Low", "Moderate", "High", "Very High"]:
            if tier in elev_tier_dist.columns:
                fig_elev.add_trace(
                    go.Bar(
                        x=elev_tier_dist.index.astype(str),
                        y=elev_tier_dist[tier],
                        name=tier,
                        marker_color=TIER_COLOR_MAP[tier],
                    )
                )

        fig_elev.update_layout(
            barmode="stack",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="% of Band", range=[0, 100]),
            xaxis=dict(title="Elevation Band"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_elev, use_container_width=True)
