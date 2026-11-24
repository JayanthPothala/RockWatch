"""
RockWatch Web Application - Model Performance View
Comprehensive benchmark analysis, ROC/PR curves, confusion matrices, and feature importances.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import joblib
from src.app.styles import get_metric_card


@st.cache_data
def load_benchmark_data():
    """Load benchmark and bootstrap summary data."""
    df_bench = pd.read_csv("models/evaluation/all_models_benchmark.csv")
    df_boot = pd.read_csv("models/evaluation/bootstrap_ci.csv")
    with open("models/evaluation/test_metrics_detailed.json", "r") as f:
        detailed_metrics = json.load(f)
    return df_bench, df_boot, detailed_metrics


@st.cache_data
def get_test_predictions():
    """Compute predictions for all models on the test set."""
    df_test = pd.read_csv("data/processed/features_test.csv")
    X_test = df_test.drop(columns=["class"])
    y_test = df_test["class"].values

    models = {
        "Gradient Boosting": joblib.load("models/trained/gradient_boosting_v1.joblib"),
        "Random Forest": joblib.load("models/trained/random_forest_v1.joblib"),
        "SVM (RBF Kernel)": joblib.load("models/trained/svm_rbf_v1.joblib"),
        "Decision Tree (d=5)": joblib.load("models/trained/decision_tree_v1.joblib"),
        "Logistic Regression": joblib.load("models/trained/logistic_regression_v1.joblib"),
    }

    probs = {}
    for name, m in models.items():
        if name == "Logistic Regression":
            lr_cols = [c for c in X_test.columns if c != "LULC_100"]
            probs[name] = m.predict_proba(X_test[lr_cols])[:, 1]
        else:
            probs[name] = m.predict_proba(X_test)[:, 1]

    return y_test, probs, models


def render_performance_view():
    """Render the Model Performance & Evaluation Page."""
    st.markdown(
        """
        <div class="rock-brand-banner">
            <span class="rock-banner-badge">📊 RIGOROUS SPATIAL EVALUATION</span>
            <h1 class="rock-banner-title">Model Performance & Benchmarks</h1>
            <p class="rock-banner-subtitle">
                Systematic evaluation across 6 model architectures evaluated on the spatially independent
                Northern Himachal test split with 500 m boundary exclusion buffers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_bench, df_boot, detailed_metrics = load_benchmark_data()
    y_test, probs, models = get_test_predictions()

    # Top KPI cards for the champion model
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            get_metric_card(
                "Champion Test ROC-AUC",
                "0.8035",
                "95% CI: [0.7816, 0.8231]",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            get_metric_card(
                "Champion Test PR-AUC",
                "0.7982",
                "Baseline Prior: 0.5223",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            get_metric_card(
                "Balanced F1 Score",
                "0.7368",
                "Optimal Threshold T = 0.37",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            get_metric_card(
                "Matthews Corr (MCC)",
                "0.4530",
                "Accuracy: 72.69% (N = 1,750)",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # Leaderboard Table
    st.markdown("### 🏆 Architecture Leaderboard (Test Split)")
    display_cols = [
        "Model", "Test_AUC", "Test_PR_AUC", "Test_F1_opt", "Test_MCC_opt", "Test_Accuracy_opt", "Spatial_CV_AUC"
    ]
    formatted_df = df_bench[display_cols].copy()
    formatted_df.columns = [
        "Model", "Test ROC-AUC", "Test PR-AUC", "Test F1", "Test MCC", "Test Accuracy", "5-Fold Spatial CV AUC"
    ]
    for col in ["Test ROC-AUC", "Test PR-AUC", "Test F1", "Test MCC", "Test Accuracy", "5-Fold Spatial CV AUC"]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) and isinstance(x, (int, float)) else ("N/A" if pd.isna(x) else str(x)))

    st.dataframe(formatted_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Interactive Comparison Charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 📈 Discrimination Power (ROC-AUC & PR-AUC)")
        # Merged data for error bars
        plot_bench = df_bench[df_bench["Model"] != "Dummy (Prior)"].copy()
        boot_merged = pd.merge(plot_bench, df_boot, on="Model", how="left")

        fig_auc = go.Figure()
        fig_auc.add_trace(
            go.Bar(
                x=boot_merged["Model"],
                y=boot_merged["Test_AUC"],
                name="ROC-AUC",
                marker_color="#1e4533",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=boot_merged["AUC_95_CI_Upper"] - boot_merged["Test_AUC"],
                    arrayminus=boot_merged["Test_AUC"] - boot_merged["AUC_95_CI_Lower"],
                ),
            )
        )
        fig_auc.add_trace(
            go.Bar(
                x=boot_merged["Model"],
                y=boot_merged["Test_PR_AUC"],
                name="PR-AUC",
                marker_color="#10b981",
            )
        )
        fig_auc.update_layout(
            barmode="group",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Metric Value", range=[0.6, 0.88]),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    with col_chart2:
        st.markdown("#### ⚖️ Classification Quality (F1 & MCC)")
        fig_f1 = go.Figure()
        fig_f1.add_trace(
            go.Bar(
                x=boot_merged["Model"],
                y=boot_merged["Test_F1_opt"],
                name="F1 Score",
                marker_color="#0284c7",
            )
        )
        fig_f1.add_trace(
            go.Bar(
                x=boot_merged["Model"],
                y=boot_merged["Test_MCC_opt"],
                name="MCC",
                marker_color="#f59e0b",
            )
        )
        fig_f1.update_layout(
            barmode="group",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Metric Value", range=[0.2, 0.85]),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    # Detailed Evaluation Tabs
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 🔬 In-Depth Diagnostic Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Confusion Matrices",
        "📉 ROC Curves",
        "📊 PR Curves",
        "🌲 Feature Importances",
        "📐 Probability Calibration"
    ])

    with tab1:
        st.markdown("##### Test Set Confusion Matrix at Optimized Decision Threshold")
        selected_cm_model = st.radio(
            "Select Model:",
            options=["Gradient Boosting (T = 0.37)", "Random Forest (T = 0.41)"],
            horizontal=True,
        )

        if "Gradient" in selected_cm_model:
            thresh = 0.37
            m_probs = probs["Gradient Boosting"]
            m_label = "Gradient Boosting"
        else:
            thresh = 0.41
            m_probs = probs["Random Forest"]
            m_label = "Random Forest"

        y_pred = (m_probs >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        cm_col1, cm_col2 = st.columns([1, 1])

        with cm_col1:
            z_text = [[f"TN: {tn:,}", f"FP: {fp:,}"], [f"FN: {fn:,}", f"TP: {tp:,}"]]
            fig_cm = px.imshow(
                cm,
                text_auto=False,
                labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Count"),
                x=["Non-Landslide (0)", "Landslide (1)"],
                y=["Non-Landslide (0)", "Landslide (1)"],
                color_continuous_scale="Greens",
            )
            fig_cm.update_traces(
                text=z_text,
                texttemplate="%{text}",
                textfont=dict(size=16, color="black"),
            )
            fig_cm.update_layout(height=320, margin=dict(l=40, r=40, t=20, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        with cm_col2:
            st.markdown(
                f"""
                <div class="rock-card" style="padding: 1.2rem;">
                    <h4 style="margin: 0 0 0.6rem 0; color: #1e4533;">{m_label} Diagnostic Ratios</h4>
                    <table style="width: 100%; font-size: 0.9rem; line-height: 1.8;">
                        <tr><td><strong>True Positives (TP):</strong></td><td>{tp:,} (Detected landslides)</td></tr>
                        <tr><td><strong>True Negatives (TN):</strong></td><td>{tn:,} (Confirmed stable points)</td></tr>
                        <tr><td><strong>False Positives (FP):</strong></td><td>{fp:,} (Safe areas flagged as hazard)</td></tr>
                        <tr><td><strong>False Negatives (FN):</strong></td><td>{fn:,} (Missed landslide events)</td></tr>
                        <tr style="border-top: 1px solid #cbd5e1;"><td><strong>Precision:</strong></td><td><strong>{tp/(tp+fp)*100:.2f}%</strong></td></tr>
                        <tr><td><strong>Recall (Sensitivity):</strong></td><td><strong>{tp/(tp+fn)*100:.2f}%</strong></td></tr>
                        <tr><td><strong>Specificity:</strong></td><td><strong>{tn/(tn+fp)*100:.2f}%</strong></td></tr>
                        <tr><td><strong>Overall Accuracy:</strong></td><td><strong>{(tp+tn)/len(y_test)*100:.2f}%</strong></td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab2:
        st.markdown("##### Receiver Operating Characteristic (ROC) Comparison")
        fig_roc = go.Figure()

        color_palette = {
            "Gradient Boosting": "#1e4533",
            "Random Forest": "#10b981",
            "SVM (RBF Kernel)": "#0284c7",
            "Decision Tree (d=5)": "#f59e0b",
            "Logistic Regression": "#8b5cf6",
        }

        for name, p in probs.items():
            fpr, tpr, _ = roc_curve(y_test, p)
            auc_val = df_bench.loc[df_bench["Model"] == name, "Test_AUC"].values[0]
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC = {auc_val:.3f})",
                    line=dict(color=color_palette.get(name, "#64748b"), width=2.5 if name == "Gradient Boosting" else 1.8),
                )
            )

        # Diagonal baseline
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance Level (AUC = 0.500)",
                line=dict(color="#94a3b8", dash="dash"),
            )
        )

        fig_roc.update_layout(
            height=420,
            xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1]),
            yaxis=dict(title="True Positive Rate (Sensitivity / Recall)", range=[0, 1.02]),
            margin=dict(l=30, r=30, t=20, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.55, y=0.1, bgcolor="rgba(255,255,255,0.9)", bordercolor="#cbd5e1", borderwidth=1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab3:
        st.markdown("##### Precision-Recall (PR) Curves")
        fig_pr = go.Figure()

        for name, p in probs.items():
            prec, rec, _ = precision_recall_curve(y_test, p)
            pr_auc = df_bench.loc[df_bench["Model"] == name, "Test_PR_AUC"].values[0]
            fig_pr.add_trace(
                go.Scatter(
                    x=rec,
                    y=prec,
                    mode="lines",
                    name=f"{name} (PR-AUC = {pr_auc:.3f})",
                    line=dict(color=color_palette.get(name, "#64748b"), width=2.5 if name == "Gradient Boosting" else 1.8),
                )
            )

        # Baseline prior
        prior = (y_test == 1).mean()
        fig_pr.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[prior, prior],
                mode="lines",
                name=f"No-Skill Baseline ({prior:.3f})",
                line=dict(color="#94a3b8", dash="dash"),
            )
        )

        fig_pr.update_layout(
            height=420,
            xaxis=dict(title="Recall (Sensitivity)", range=[0, 1]),
            yaxis=dict(title="Precision (Positive Predictive Value)", range=[0.4, 1.02]),
            margin=dict(l=30, r=30, t=20, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.03, y=0.12, bgcolor="rgba(255,255,255,0.9)", bordercolor="#cbd5e1", borderwidth=1),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    with tab4:
        st.markdown("##### Predictor Importance in Tree Ensembles")
        fi_model = st.radio(
            "Display Importances For:",
            options=["Gradient Boosting (MDI)", "Random Forest (MDI)"],
            horizontal=True,
        )

        feature_order = [
            "sin_aspect", "cos_aspect", "Dist_River_z_corr", "Elevation_z_corr",
            "FlowAccumulation_z_corr", "NDVI_z_corr", "Rainfall_z_corr", "Slope_z_corr",
            "LULC_10", "LULC_20", "LULC_30", "LULC_40", "LULC_50", "LULC_60",
            "LULC_70", "LULC_80", "LULC_100"
        ]

        if "Gradient" in fi_model:
            imp_vals = models["Gradient Boosting"].feature_importances_
            bar_color = "#1e4533"
        else:
            imp_vals = models["Random Forest"].feature_importances_
            bar_color = "#10b981"

        fi_df = pd.DataFrame({"Feature": feature_order, "Importance": imp_vals}).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=[bar_color],
        )
        fig_fi.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title="Relative Gini Impurity Reduction"),
            yaxis=dict(title="Model Feature"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.info("💡 **Key Finding:** Elevation, Slope, Rainfall, and NDVI dominate model split decisions (>92% of cumulative importance), consistent with geomorphological theory in the Himalayas.")

    with tab5:
        st.markdown("##### Reliability Calibration Diagrams")
        calib_data = detailed_metrics.get("calibration", {})

        fig_calib = go.Figure()
        fig_calib.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect Calibration",
                line=dict(color="#94a3b8", dash="dash"),
            )
        )

        for m_name, c_vals in calib_data.items():
            if m_name in ["Gradient Boosting", "Random Forest", "Logistic Regression"]:
                fig_calib.add_trace(
                    go.Scatter(
                        x=c_vals["prob_pred"],
                        y=c_vals["prob_true"],
                        mode="lines+markers",
                        name=f"{m_name} (Brier: {c_vals['brier_score']:.3f}, ECE: {c_vals['ece']:.3f})",
                        line=dict(color=color_palette.get(m_name, "#64748b"), width=2),
                    )
                )

        fig_calib.update_layout(
            height=420,
            xaxis=dict(title="Mean Predicted Probability", range=[0, 1]),
            yaxis=dict(title="Observed Landslide Fraction", range=[0, 1]),
            margin=dict(l=30, r=30, t=20, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.03, y=0.88, bgcolor="rgba(255,255,255,0.9)", bordercolor="#cbd5e1", borderwidth=1),
        )
        st.plotly_chart(fig_calib, use_container_width=True)

    # Statistical Significance Hypothesis Testing Box
    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 🧪 Statistical Significance Hypothesis Testing")
    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        st.markdown(
            """
            <div class="rock-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #1e4533;">Gradient Boosting vs. Random Forest</h4>
                <div style="font-size: 0.9rem; color: #334155; line-height: 1.6;">
                    <ul>
                        <li><strong>Delta ROC-AUC:</strong> +0.0172 (1.72% gain)</li>
                        <li><strong>95% Paired Bootstrap CI:</strong> [+0.0104, +0.0248]</li>
                        <li><strong>p-value:</strong> &lt; 0.001 (Statistically significant)</li>
                    </ul>
                    <span style="font-size: 0.85rem; color: #065f46; font-weight: 600;">
                        ✅ Gradient Boosting significantly outperforms Random Forest under spatial holdout.
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with stat_col2:
        st.markdown(
            """
            <div class="rock-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #1e4533;">Gradient Boosting vs. Logistic Regression</h4>
                <div style="font-size: 0.9rem; color: #334155; line-height: 1.6;">
                    <ul>
                        <li><strong>Delta ROC-AUC:</strong> +0.1037 (10.37% gain)</li>
                        <li><strong>95% Paired Bootstrap CI:</strong> [+0.0835, +0.1238]</li>
                        <li><strong>p-value:</strong> &lt; 0.001 (Statistically significant)</li>
                    </ul>
                    <span style="font-size: 0.85rem; color: #065f46; font-weight: 600;">
                        ✅ Non-linear tree interactions are essential; linear models severely underfit.
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
