"""
RockWatch Web Application - Regional Analysis View
Zonal statistics, latitude corridors, and district-level susceptibility breakdowns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.app.styles import get_metric_card
from src.app.report_generator import export_regional_summary_csv, export_regional_geojson


@st.cache_data
def load_regional_data():
    """Load susceptibility predictions joined with GSI inventory district metadata."""
    df_pred = pd.read_csv("data/processed/himachal_susceptibility_predictions.csv")
    df_raw = pd.read_csv("data/raw/himachal_landslide/landslide_modelling_dataset_12409.csv")
    df_inv = pd.read_csv("data/raw/himachal_landslide/hp_landslide_inventory_clean.csv")

    m1 = pd.merge(df_pred, df_raw[["sample_id", "nearest_inventory_sno"]], on="sample_id", how="left")
    merged = pd.merge(m1, df_inv[["sno", "district"]], left_on="nearest_inventory_sno", right_on="sno", how="left")

    def assign_corridor(lat):
        if lat < 31.713044:
            return "Southern Outer Himalayas (< 31.71°N)"
        elif lat < 32.144235:
            return "Central Middle Himalayas (31.71° - 32.14°N)"
        else:
            return "Northern Higher Himalayas (≥ 32.14°N)"

    merged["corridor"] = merged["latitude"].apply(assign_corridor)

    # Elevation Bins
    elev_bins = [0, 1000, 2000, 3000, 4000, 6000]
    elev_labels = ["< 1,000 m (Sub-Himalayan)", "1,000 - 2,000 m (Lower)", "2,000 - 3,000 m (Mid)", "3,000 - 4,000 m (High)", "> 4,000 m (Alpine)"]
    merged["elev_band"] = pd.cut(merged["elevation_m"], bins=elev_bins, labels=elev_labels, right=False)

    # Slope Bins
    slope_bins = [0, 15, 25, 35, 45, 90]
    slope_labels = ["Gentle (< 15°)", "Moderate (15° - 25°)", "Moderately Steep (25° - 35°)", "Steep (35° - 45°)", "Extreme (> 45°)"]
    merged["slope_class"] = pd.cut(merged["slope_deg"], bins=slope_bins, labels=slope_labels, right=False)

    return merged


def render_regional_view():
    """Render the Regional Analysis Page."""
    st.markdown(
        """
        <div class="rock-brand-banner">
            <span class="rock-banner-badge">🏔️ GEOGRAPHIC & ZONAL PROFILING</span>
            <h1 class="rock-banner-title">Regional Susceptibility Analysis</h1>
            <p class="rock-banner-subtitle">
                Inspect terrain risk gradients across Himachal Pradesh's physiographic zones, latitudinal corridors,
                elevation belts, and documented administrative districts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_regional_data()

    # Corridor Metrics
    st.markdown("### 🧭 Latitudinal Himalayan Corridors")
    st.markdown(
        """
        <p style="color: #475569; font-size: 0.92rem;">
            Himachal Pradesh transitions from the Shiwalik foothills in the south to the Greater Himalayan crystalline
            ranges in the north. Our spatial validation split boundaries partition the state into three distinct geographic belts:
        </p>
        """,
        unsafe_allow_html=True,
    )

    corridor_stats = df.groupby("corridor").agg(
        Total_Points=("sample_id", "count"),
        Mean_Score=("susceptibility_score", "mean"),
        Pct_High_VHigh=("susceptibility_class", lambda s: (s.isin(["High", "Very High"])).mean() * 100),
        Mean_Elevation=("elevation_m", "mean"),
        Mean_Slope=("slope_deg", "mean"),
    ).reset_index()

    col_z1, col_z2, col_z3 = st.columns(3)
    corridors_list = [
        "Southern Outer Himalayas (< 31.71°N)",
        "Central Middle Himalayas (31.71° - 32.14°N)",
        "Northern Higher Himalayas (≥ 32.14°N)",
    ]
    cols = [col_z1, col_z2, col_z3]

    for i, corr_name in enumerate(corridors_list):
        row = corridor_stats[corridor_stats["corridor"] == corr_name].iloc[0]
        with cols[i]:
            st.markdown(
                f"""
                <div class="rock-card" style="padding: 1.2rem;">
                    <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Corridor {i+1}</div>
                    <h4 style="margin: 0.2rem 0 0.8rem 0; font-size: 1.05rem; color: #1e4533;">{corr_name.split(' (')[0]}</h4>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #0f172a;">{row['Pct_High_VHigh']:.1f}%</div>
                    <div style="font-size: 0.8rem; color: #9a3412; font-weight: 600; margin-bottom: 0.6rem;">High & Very High Hazard</div>
                    <div style="font-size: 0.85rem; color: #475569; line-height: 1.6; border-top: 1px solid #f1f5f9; padding-top: 0.5rem;">
                        📍 <strong>{row['Total_Points']:,}</strong> surveyed locations<br>
                        ⛰️ Mean Elevation: <strong>{row['Mean_Elevation']:.0f} m</strong><br>
                        📐 Mean Slope: <strong>{row['Mean_Slope']:.1f}°</strong><br>
                        📊 Average Score: <strong>{row['Mean_Score']:.3f}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Corridor Hazard Tier Breakdown Chart
    corr_tier_crosstab = pd.crosstab(
        df["corridor"],
        df["susceptibility_class"],
        normalize="index"
    ).reindex(columns=["Low", "Moderate", "High", "Very High"]).fillna(0) * 100

    tier_colors = {
        "Low": "#10b981",
        "Moderate": "#f59e0b",
        "High": "#f97316",
        "Very High": "#ef4444",
    }

    fig_corr = go.Figure()
    for tier in ["Low", "Moderate", "High", "Very High"]:
        fig_corr.add_trace(
            go.Bar(
                y=corr_tier_crosstab.index,
                x=corr_tier_crosstab[tier],
                name=tier,
                orientation="h",
                marker_color=tier_colors[tier],
            )
        )
    fig_corr.update_layout(
        barmode="stack",
        height=260,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="% of Locations in Corridor", range=[0, 100]),
        yaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # District Level Analysis
    st.markdown("### 🏛️ District Vulnerability Profiles (GSI Inventory Records)")
    st.markdown(
        """
        <div class="rock-disclaimer-box" style="margin-top: 0; margin-bottom: 1.2rem;">
            <strong>Data Lineage Note:</strong> Administrative district attributions are available for the
            <strong>5,859 historical landslide events</strong> documented by the Geological Survey of India (GSI).
            Pseudo-absence locations generated across unpopulated mountain slopes lack administrative boundaries and are excluded here.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_dist = df[df["district"].notnull()].copy()
    district_summary = df_dist.groupby("district").agg(
        Landslides=("sample_id", "count"),
        Mean_Score=("susceptibility_score", "mean"),
        Pct_High_VHigh=("susceptibility_class", lambda s: (s.isin(["High", "Very High"])).mean() * 100),
        Mean_Slope=("slope_deg", "mean"),
        Mean_Elevation=("elevation_m", "mean"),
    ).sort_values("Landslides", ascending=False).reset_index()

    col_d1, col_d2 = st.columns([3, 2])

    with col_d1:
        st.markdown("##### Historical Event Frequency & Mean Susceptibility by District")
        fig_dist_bar = px.bar(
            district_summary,
            x="district",
            y="Landslides",
            color="Pct_High_VHigh",
            color_continuous_scale="Reds",
            labels={
                "district": "District",
                "Landslides": "Documented Landslides",
                "Pct_High_VHigh": "% High/V.High Risk"
            },
            text="Landslides",
        )
        fig_dist_bar.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_dist_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_dist_bar, use_container_width=True)

    with col_d2:
        st.markdown("##### District Summary Statistics")
        dist_table = district_summary.copy()
        dist_table["Mean_Score"] = dist_table["Mean_Score"].map(lambda x: f"{x:.3f}")
        dist_table["Pct_High_VHigh"] = dist_table["Pct_High_VHigh"].map(lambda x: f"{x:.1f}%")
        dist_table["Mean_Slope"] = dist_table["Mean_Slope"].map(lambda x: f"{x:.1f}°")
        dist_table["Mean_Elevation"] = dist_table["Mean_Elevation"].map(lambda x: f"{x:.0f} m")
        dist_table.columns = ["District", "Events", "Mean Score", "% High/V.High", "Avg Slope", "Avg Elevation"]
        st.dataframe(dist_table, use_container_width=True, hide_index=True)

    # Elevation and Slope Susceptibility Matrix
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 📐 Geomorphic Terrain Breakdown")

    col_geo1, col_geo2 = st.columns(2)

    with col_geo1:
        st.markdown("##### Susceptibility by Elevation Belt")
        elev_crosstab = pd.crosstab(
            df["elev_band"],
            df["susceptibility_class"],
            normalize="index"
        ).reindex(columns=["Low", "Moderate", "High", "Very High"]).fillna(0) * 100

        fig_eb = go.Figure()
        for tier in ["Low", "Moderate", "High", "Very High"]:
            fig_eb.add_trace(
                go.Bar(
                    x=elev_crosstab.index.astype(str),
                    y=elev_crosstab[tier],
                    name=tier,
                    marker_color=tier_colors[tier],
                )
            )
        fig_eb.update_layout(
            barmode="stack",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="% in Elevation Belt", range=[0, 100]),
            xaxis=dict(title=""),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_eb, use_container_width=True)
        st.caption("ℹ️ Critical inflection occurs in the 1,000–3,000 m zone where steep slopes intersect high monsoon precipitation.")

    with col_geo2:
        st.markdown("##### Susceptibility by Slope Gradient Class")
        slope_crosstab = pd.crosstab(
            df["slope_class"],
            df["susceptibility_class"],
            normalize="index"
        ).reindex(columns=["Low", "Moderate", "High", "Very High"]).fillna(0) * 100

        fig_sb = go.Figure()
        for tier in ["Low", "Moderate", "High", "Very High"]:
            fig_sb.add_trace(
                go.Bar(
                    x=slope_crosstab.index.astype(str),
                    y=slope_crosstab[tier],
                    name=tier,
                    marker_color=tier_colors[tier],
                )
            )
        fig_sb.update_layout(
            barmode="stack",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="% in Slope Class", range=[0, 100]),
            xaxis=dict(title=""),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sb, use_container_width=True)
        st.caption("ℹ️ Slopes > 35° exhibit >70% High & Very High hazard predisposition, corresponding to active escarpment zones.")

    # ---------------------------------------------------------
    # REGIONAL DATA & SPATIAL GEOMETRY EXPORT SECTION
    # ---------------------------------------------------------
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 💾 Regional Data & Spatial Geometry Exports")
    st.markdown(
        """
        <p style="color: #64748b; font-size: 0.88rem; margin-bottom: 1rem;">
            Export authenticated regional susceptibility summaries (CSV) and geographical vector geometries (GeoJSON)
            for GIS and external spatial analysis (QGIS, ArcGIS, Python Geopandas). Original coordinates and model outputs are strictly preserved.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🛠️ Filter & Configure Spatial Export", expanded=True):
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            corridor_filter = st.selectbox(
                "Filter by Latitudinal Corridor:",
                options=["All Corridors"] + list(corridor_stats["corridor"].unique()),
                index=0,
            )
        with exp_col2:
            dist_options = ["All Districts"] + sorted([str(d) for d in df["district"].dropna().unique()])
            district_filter = st.selectbox(
                "Filter by Administrative District (GSI Inventory Points):",
                options=dist_options,
                index=0,
            )
        with exp_col3:
            tier_export_options = st.multiselect(
                "Susceptibility Tiers to Include:",
                options=["Low", "Moderate", "High", "Very High"],
                default=["Low", "Moderate", "High", "Very High"],
            )

    # Apply export filters
    export_df = df.copy()
    if corridor_filter != "All Corridors":
        export_df = export_df[export_df["corridor"] == corridor_filter]
    if district_filter != "All Districts":
        export_df = export_df[export_df["district"] == district_filter]
    if tier_export_options:
        export_df = export_df[export_df["susceptibility_class"].isin(tier_export_options)]
    else:
        st.warning("⚠️ No tiers selected for export. Please select at least one tier.")
        export_df = export_df.iloc[0:0]

    n_exp = len(export_df)
    st.markdown(
        f"<div style='font-size: 0.85rem; color: #475569; margin-bottom: 0.8rem;'>"
        f"📍 <strong>Filtered Points for Export:</strong> {n_exp:,} coordinates matching criteria</div>",
        unsafe_allow_html=True,
    )

    if n_exp > 0:
        exp_btn1, exp_btn2, exp_btn3 = st.columns(3)
        with exp_btn1:
            dist_csv = export_regional_summary_csv(district_summary)
            st.download_button(
                "📑 Download District Summary (CSV)",
                data=dist_csv,
                file_name="RockWatch_District_Vulnerability_Summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_btn2:
            corr_csv = export_regional_summary_csv(corridor_stats)
            st.download_button(
                "📊 Download Corridor Summary (CSV)",
                data=corr_csv,
                file_name="RockWatch_Corridor_Vulnerability_Summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_btn3:
            geojson_data = export_regional_geojson(export_df)
            st.download_button(
                "🗺️ Download Filtered Points (GeoJSON)",
                data=geojson_data,
                file_name="RockWatch_Himachal_Susceptibility_Points.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
