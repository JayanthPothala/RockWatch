"""
RockWatch Web Application - About the Project View
Methodology, data leakage safeguards, spatial validation design, scientific limitations, and disclaimers.
"""

import streamlit as st


def render_about_view():
    """Render the About, Methodology & Disclaimers Page."""
    st.markdown(
        """
        <div class="rock-brand-banner">
            <span class="rock-banner-badge">📚 SCIENTIFIC METHODOLOGY & GOVERNANCE</span>
            <h1 class="rock-banner-title">About the RockWatch Project</h1>
            <p class="rock-banner-subtitle">
                Comprehensive documentation of dataset provenance, data leakage prevention protocols,
                spatial validation architecture, mathematical re-standardisation, and operational limitations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Dataset Origin & Pedigree
    st.markdown(
        """
        <div class="rock-card">
            <h3 class="rock-card-title">🌐 1. Dataset Provenance & Data Sources</h3>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
                RockWatch integrates multi-source earth observation data covering the entire mountainous state of Himachal Pradesh
                (12 administrative districts, Western Himalayas, India). The ground truth consists of:
            </p>
            <ul style="color: #334155; font-size: 0.9rem; line-height: 1.8;">
                <li><strong>Historical Landslide Inventory (Class = 1):</strong> Compiled by the <em>Geological Survey of India (GSI)</em> through field mapping, disaster reconnaissance, and satellite imagery verification from 2014 to 2020.</li>
                <li><strong>Pseudo-Absence Locations (Class = 0):</strong> Stratified spatial sampling across mountain slopes without recorded instability, excluding confirmed landslide scars and high-risk river terraces.</li>
                <li><strong>Topography (SRTM DEM):</strong> 30-meter Shuttle Radar Topography Mission digital elevation model providing elevation, slope, and topographic aspect.</li>
                <li><strong>Vegetation (Sentinel-2 MSI):</strong> Multi-spectral Normalized Difference Vegetation Index (NDVI) measuring seasonal vegetation cover.</li>
                <li><strong>Hydrology & Climate:</strong> Flow accumulation derived from hydrological flow routing and precipitation estimates from ERA5-Land reanalysis.</li>
                <li><strong>Land Cover (ESA WorldCover):</strong> 10-meter global land cover mapping 9 surface categories.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Mathematical Leakage Prevention
    st.markdown(
        """
        <div class="rock-card">
            <h3 class="rock-card-title">🔒 2. Data Leakage Prevention Protocol</h3>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
                In geospatial machine learning, standardising an entire dataset before splitting introduces <strong>data leakage</strong>,
                as test and validation statistics inadvertently influence the training feature distribution. In RockWatch, we proved
                mathematically that training-only re-standardisation eliminates this contamination:
            </p>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.1rem; margin: 0.8rem 0; font-family: monospace; font-size: 0.88rem; color: #1e293b;">
                Let x be the raw feature, &mu;<sub>G</sub> and &sigma;<sub>G</sub> be the original global scaling constants.<br>
                The original feature was: z<sub>G</sub> = (x - &mu;<sub>G</sub>) / &sigma;<sub>G</sub><br><br>
                Re-standardising z<sub>G</sub> using training-only mean (&mu;<sub>T,z</sub>) and training-only std (&sigma;<sub>T,z</sub>):<br>
                z<sub>corr</sub> = (z<sub>G</sub> - &mu;<sub>T,z</sub>) / &sigma;<sub>T,z</sub><br>
                = [ (x - &mu;<sub>G</sub>)/&sigma;<sub>G</sub> - (&mu;<sub>T</sub> - &mu;<sub>G</sub>)/&sigma;<sub>G</sub> ] / [ &sigma;<sub>T</sub>/&sigma;<sub>G</sub> ]<br>
                = (x - &mu;<sub>T</sub>) / &sigma;<sub>T</sub>
            </div>
            <p style="color: #065f46; font-size: 0.9rem; font-weight: 600;">
                ✅ The global parameters &mu;<sub>G</sub> and &sigma;<sub>G</sub> cancel out completely. The resulting feature representation
                is mathematically identical to fitting directly on raw training features without cross-split leakage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Spatial Validation Design
    st.markdown(
        """
        <div class="rock-card">
            <h3 class="rock-card-title">🗺️ 3. Spatially Blocked Validation Strategy</h3>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
                Standard random k-fold cross-validation in spatial domains creates severe optimistic bias due to
                <strong>Tobler's First Law of Geography</strong>: nearby points share similar elevation, geology, and rainfall patterns.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-top: 0.8rem;">
                <div style="background: #fafafa; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem;">
                    <strong style="color: #1e4533;">Southern Split (TRAIN):</strong>
                    <p style="margin: 0.4rem 0 0 0; font-size: 0.85rem; color: #475569;">
                        8,188 rows (Latitude &lt; 31.713°N). Contains Outer Himalayas and Shiwalik ranges. Used exclusively to fit models and scalers.
                    </p>
                </div>
                <div style="background: #fafafa; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem;">
                    <strong style="color: #1e4533;">Central Split (VAL):</strong>
                    <p style="margin: 0.4rem 0 0 0; font-size: 0.85rem; color: #475569;">
                        1,731 rows (31.713°N ≤ Lat &lt; 32.144°N). Middle Himalayas. Used exclusively for hyperparameter tuning and threshold optimization.
                    </p>
                </div>
                <div style="background: #fafafa; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem;">
                    <strong style="color: #1e4533;">Northern Split (TEST):</strong>
                    <p style="margin: 0.4rem 0 0 0; font-size: 0.85rem; color: #475569;">
                        1,750 rows (Latitude ≥ 32.144°N). Greater Himalayas and alpine zones. Kept completely held out until final benchmark scoring.
                    </p>
                </div>
            </div>
            <p style="color: #64748b; font-size: 0.86rem; margin-top: 0.8rem;">
                🛡️ <strong>500 m Exclusion Buffer:</strong> A geographic buffer zone of ±500 m (~0.004505° latitude) was enforced around both split
                boundaries (excluding 68 boundary rows) to prevent edge point leakage across partitions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Known Scientific Limitations
    st.markdown(
        """
        <div class="rock-card">
            <h3 class="rock-card-title">⚠️ 4. Known Scientific Limitations</h3>
            <ul style="color: #334155; font-size: 0.9rem; line-height: 1.8;">
                <li><strong>Slope Quantisation Artefact:</strong> The source dataset's <code>Slope_z</code> predictor was originally derived from integer-degree slope (<code>round(slope_deg, 0)</code>), creating 76 discrete bins rather than a continuous curve. While tree-based ensembles are robust to small within-bin variations, this quantisation limits fine-scale sensitivity below ±0.5°.</li>
                <li><strong>Static Susceptibility vs. Real-Time Triggering:</strong> Susceptibility models evaluate <em>spatial predisposition</em> (where landslides are prone to occur based on morphology, hydrology, and lithology). They do NOT predict <em>when</em> a failure will occur. Real-time forecasting requires high-frequency temporal rainfall sensors, pore-pressure piezometers, and seismic triggers.</li>
                <li><strong>Spatial Independence Precaution:</strong> The 500 m boundary buffer mitigates immediate pixel adjacency, but long-range environmental autocorrelation in macro-climate and geological formations can span 5–50 km across mountain valleys.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5. Prominent Disaster Management Disclaimer
    st.markdown(
        """
        <div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.6rem; color: #991b1b; font-weight: 800; font-size: 1.15rem;">
                <span>🚨</span> CRITICAL DISASTER MANAGEMENT DISCLAIMER
            </div>
            <p style="color: #7f1d1d; font-size: 0.92rem; line-height: 1.6; margin-top: 0.6rem; margin-bottom: 0;">
                The RockWatch system is an academic research and environmental planning tool. It is designed for regional risk zoning,
                infrastructure corridor screening, and macro-scale civil defense planning. <strong>It MUST NOT be used as the sole basis
                for life-safety evacuations, building approvals, or site-specific construction engineering.</strong>
                All engineering interventions require site-level geotechnical drill cores, soil shear testing, and continuous
                hydrometric monitoring by certified geological authorities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 6. Technology Stack & Citations
    st.markdown(
        """
        <div class="rock-card">
            <h3 class="rock-card-title">💻 5. Technology Stack & Credits</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.8rem; font-size: 0.88rem; color: #475569;">
                <div>🐍 <strong>Python 3.14</strong></div>
                <div>⚡ <strong>Streamlit 1.67</strong></div>
                <div>📊 <strong>Plotly 7.1</strong></div>
                <div>🤖 <strong>Scikit-Learn 1.7</strong></div>
                <div>🐼 <strong>Pandas 2.2</strong></div>
                <div>🔢 <strong>NumPy 2.2</strong></div>
            </div>
            <p style="color: #64748b; font-size: 0.82rem; margin-top: 1rem; margin-bottom: 0;">
                RockWatch Framework · Developed for Himachal Pradesh Geospatial Hazard Assessment · 2026.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
