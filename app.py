"""
RockWatch - Intelligent Landslide Susceptibility Prediction System
Main Streamlit Application Entry Point
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.styles import CUSTOM_CSS
from src.app.views_home import render_home_view
from src.app.views_map import render_map_view
from src.app.views_predict import render_predict_view
from src.app.views_performance import render_performance_view
from src.app.views_regional import render_regional_view
from src.app.views_about import render_about_view


def main():
    """Main application orchestrator."""
    st.set_page_config(
        page_title="RockWatch - Landslide Susceptibility System",
        page_icon="⛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject environmental intelligence custom styles
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar Navigation & Information
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.8rem 0; text-align: left;">
                <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
                    ⛰️ RockWatch
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 500; margin-top: 0.2rem;">
                    Himachal Pradesh Landslide Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page_choice = st.radio(
            "Navigation Menu",
            options=[
                "🏠 Home Dashboard",
                "🗺️ Susceptibility Map",
                "⚡ Landslide Prediction",
                "📊 Model Performance",
                "🏔️ Regional Analysis",
                "ℹ️ About & Methodology",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Sidebar Quick Facts Widget
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 1rem; margin-top: 1rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #10b981; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 0.4rem;">
                    SYSTEM STATUS
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.6;">
                    • <strong>Model:</strong> Gradient Boosting v1<br>
                    • <strong>Test AUC:</strong> 0.8035 (±0.021)<br>
                    • <strong>Test F1:</strong> 0.7368 (T=0.37)<br>
                    • <strong>Dataset:</strong> 11,737 coordinates<br>
                    • <strong>Buffer:</strong> 500m Latitudinal Block<br>
                    • <strong>Region:</strong> Himachal Pradesh
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        st.caption("RockWatch v1.0 · Western Himalayas Framework")

    # Route to selected page module
    if "Home" in page_choice:
        render_home_view()
    elif "Map" in page_choice:
        render_map_view()
    elif "Prediction" in page_choice:
        render_predict_view()
    elif "Performance" in page_choice:
        render_performance_view()
    elif "Regional" in page_choice:
        render_regional_view()
    elif "About" in page_choice:
        render_about_view()


if __name__ == "__main__":
    main()
