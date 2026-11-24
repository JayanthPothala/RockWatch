"""
RockWatch - Phase 5: Deep Model Evaluation, Calibration & Diagnostics
======================================================================
Executes in-depth evaluation on the held-out spatial test set:
1. Probability Calibration & Reliability Curves (Brier score & ECE)
2. Normalized Confusion Matrices across top models
3. Operating Threshold Policy Analysis (Early Warning, Balanced, Infrastructure)
4. Non-parametric Bootstrap Confidence Intervals (95% CI, 1,000 resamples)
5. Physical Error Profiling (FP vs FN environmental attribute comparison)
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.calibration import CalibratedClassifierCV

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_processed_data
from src.evaluation.calibration import evaluate_calibration, compute_expected_calibration_error
from src.models.baseline import ALL_FEATURES, LR_FEATURES


def run_model_evaluation():
    print("=" * 80)
    print("ROCKWATCH - PHASE 5: MODEL EVALUATION, CALIBRATION & DIAGNOSTICS")
    print("=" * 80)

    # 1. Load Data
    print("\n[1] Loading processed test and validation datasets...")
    f_val = load_processed_data("val", features_only=True)
    f_test = load_processed_data("test", features_only=True)
    test_spatial = load_processed_data("test", features_only=False)

    y_val = f_val["class"].values
    y_test = f_test["class"].values

    X_val = f_val[ALL_FEATURES].values
    X_test = f_test[ALL_FEATURES].values

    # 2. Load Models
    models_dir = PROJECT_ROOT / "models" / "trained"
    eval_dir = PROJECT_ROOT / "models" / "evaluation"
    fig_dir = PROJECT_ROOT / "reports" / "figures"

    models = {
        "Gradient Boosting": (joblib.load(models_dir / "gradient_boosting_v1.joblib"), ALL_FEATURES, 0.37),
        "Random Forest": (joblib.load(models_dir / "random_forest_v1.joblib"), ALL_FEATURES, 0.41),
        "SVM (RBF Kernel)": (joblib.load(models_dir / "svm_rbf_v1.joblib"), ALL_FEATURES, 0.30),
        "Decision Tree (d=5)": (joblib.load(models_dir / "decision_tree_v1.joblib"), ALL_FEATURES, 0.22),
        "Logistic Regression": (joblib.load(models_dir / "logistic_regression_v1.joblib"), LR_FEATURES, 0.43),
    }

    # Generate test probabilities
    test_probs = {}
    for name, (clf, feats, _) in models.items():
        X = f_test[feats].values
        test_probs[name] = clf.predict_proba(X)[:, 1]

    # 3. Probability Calibration Analysis
    print("\n[2] Probability Calibration & Reliability Assessment (Test Set)...")
    calib_results = {}
    print(f"  {'Model':22s} {'Brier Score':>12s} {'ECE':>10s}")
    print("  " + "-" * 46)

    for name, p_te in test_probs.items():
        cal = evaluate_calibration(y_test, p_te, n_bins=10)
        calib_results[name] = cal
        print(f"  {name:22s} {cal['brier_score']:12.4f} {cal['ece']:10.4f}")

    # Plot Calibration Curves
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = {
        "Gradient Boosting": "#d7191c",
        "Random Forest": "#2b83ba",
        "SVM (RBF Kernel)": "#abdda4",
        "Decision Tree (d=5)": "#fdae61",
        "Logistic Regression": "#74a9cf",
    }

    for name, cal in calib_results.items():
        ax.plot(
            cal["prob_pred"],
            cal["prob_true"],
            marker="o",
            label=f"{name} (Brier={cal['brier_score']:.3f}, ECE={cal['ece']:.3f})",
            color=colors[name],
            linewidth=2.0 if "Gradient" in name or "Random" in name else 1.3,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    ax.set_title("Reliability Diagrams (Probability Calibration on Held-out Spatial Test Set)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10, fontweight="bold")
    ax.set_ylabel("Empirical Fraction of Landslides", fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    p_cal = fig_dir / "calibration_curves.png"
    fig.savefig(p_cal, bbox_inches="tight")
    print(f"  Saved calibration plot: {p_cal.name}")

    # 4. Confusion Matrices
    print("\n[3] Generating Confusion Matrices across Top Models...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
    axes = axes.flatten()

    cm_models = ["Gradient Boosting", "Random Forest", "Decision Tree (d=5)", "Logistic Regression"]
    for i, name in enumerate(cm_models):
        clf, feats, thresh = models[name]
        p_te = test_probs[name]
        y_pred = (p_te >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if i % 2 == 0 else "Oranges",
            cbar=False,
            ax=axes[i],
            annot_kws={"size": 13, "fontweight": "bold"},
        )
        axes[i].set_title(f"{name} (Threshold = {thresh:.2f})\nAccuracy: {np.mean(y_test==y_pred)*100:.1f}% | F1: {f1_score(y_test, y_pred):.3f}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted Class (0: Non-Landslide, 1: Landslide)", fontsize=9)
        axes[i].set_ylabel("True Class (0: Non-Landslide, 1: Landslide)", fontsize=9)

    fig.suptitle("RockWatch - Confusion Matrices on Held-out Spatial Test Set", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    p_cm = fig_dir / "confusion_matrices.png"
    fig.savefig(p_cm, bbox_inches="tight")
    print(f"  Saved confusion matrices plot: {p_cm.name}")

    # 5. Operating Policy Threshold Analysis (Champion: Gradient Boosting)
    print("\n[4] Operating Threshold Sensitivity Analysis for Gradient Boosting...")
    p_gb = test_probs["Gradient Boosting"]
    threshold_range = np.linspace(0.05, 0.95, 91)

    sens_data = []
    for t in threshold_range:
        yp = (p_gb >= t).astype(int)
        sens_data.append({
            "Threshold": t,
            "Precision": precision_score(y_test, yp, zero_division=0),
            "Recall": recall_score(y_test, yp, zero_division=0),
            "F1": f1_score(y_test, yp, zero_division=0),
            "MCC": matthews_corrcoef(y_test, yp),
            "Accuracy": np.mean(y_test == yp),
        })
    sens_df = pd.DataFrame(sens_data)

    # Define 3 Operational Policies
    # 1. Early Warning: Recall >= 0.90
    ew_row = sens_df[sens_df["Recall"] >= 0.90].iloc[-1]
    # 2. Balanced: Max F1
    bal_row = sens_df.loc[sens_df["F1"].idxmax()]
    # 3. Infrastructure: Precision >= 0.85
    infra_candidates = sens_df[sens_df["Precision"] >= 0.85]
    infra_row = infra_candidates.iloc[0] if len(infra_candidates) > 0 else sens_df.loc[sens_df["Precision"].idxmax()]

    print("  === THREE OPERATIONAL POLICY THRESHOLDS ===")
    print(f"  1. Early Warning Policy (High Recall):")
    print(f"     Threshold = {ew_row['Threshold']:.2f} | Recall: {ew_row['Recall']*100:.1f}% | Precision: {ew_row['Precision']*100:.1f}% | F1: {ew_row['F1']:.4f}")
    print(f"  2. Balanced Civil Defense Policy (Optimal F1):")
    print(f"     Threshold = {bal_row['Threshold']:.2f} | Recall: {bal_row['Recall']*100:.1f}% | Precision: {bal_row['Precision']*100:.1f}% | F1: {bal_row['F1']:.4f}")
    print(f"  3. Capital Infrastructure Planning Policy (High Precision):")
    print(f"     Threshold = {infra_row['Threshold']:.2f} | Recall: {infra_row['Recall']*100:.1f}% | Precision: {infra_row['Precision']*100:.1f}% | F1: {infra_row['F1']:.4f}")

    # Plot Threshold Sensitivity
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    ax.plot(sens_df["Threshold"], sens_df["Precision"], label="Precision", color="#2b83ba", linewidth=2.0)
    ax.plot(sens_df["Threshold"], sens_df["Recall"], label="Recall (Sensitivity)", color="#d7191c", linewidth=2.0)
    ax.plot(sens_df["Threshold"], sens_df["F1"], label="F1-Score", color="#1b9e77", linewidth=2.2)
    ax.plot(sens_df["Threshold"], sens_df["MCC"], label="Matthews Correlation (MCC)", color="#7570b3", linestyle="--", linewidth=1.5)

    # Highlight operational thresholds
    ax.axvline(ew_row["Threshold"], color="#d7191c", linestyle=":", label=f"Early Warning (T={ew_row['Threshold']:.2f}, Rec=90%)")
    ax.axvline(bal_row["Threshold"], color="#1b9e77", linestyle=":", label=f"Balanced (T={bal_row['Threshold']:.2f}, F1={bal_row['F1']:.3f})")
    ax.axvline(infra_row["Threshold"], color="#2b83ba", linestyle=":", label=f"Infrastructure (T={infra_row['Threshold']:.2f}, Prec={infra_row['Precision']*100:.0f}%)")

    ax.set_title("Operational Decision Curves: Metrics vs Susceptibility Threshold (Gradient Boosting)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Susceptibility Probability Threshold", fontsize=10, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=10, fontweight="bold")
    ax.legend(loc="center left", frameon=True, fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    p_sens = fig_dir / "threshold_sensitivity.png"
    fig.savefig(p_sens, bbox_inches="tight")
    print(f"  Saved threshold sensitivity plot: {p_sens.name}")

    # 6. Non-Parametric Bootstrap (1,000 resamples) for 95% Confidence Intervals
    print("\n[5] Running 1,000 Bootstrap Resamples for 95% Confidence Intervals & Significance...")
    np.random.seed(42)
    n_boot = 1000
    n_samples = len(y_test)

    boot_metrics = {name: {"auc": [], "f1": []} for name in models}
    diff_gb_rf = []
    diff_gb_lr = []

    for _ in range(n_boot):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_b = y_test[indices]

        # Check that both classes exist
        if len(np.unique(y_b)) < 2:
            continue

        b_aucs = {}
        for name, (_, _, thresh) in models.items():
            p_b = test_probs[name][indices]
            auc_b = roc_auc_score(y_b, p_b)
            f1_b = f1_score(y_b, (p_b >= thresh).astype(int), zero_division=0)
            boot_metrics[name]["auc"].append(auc_b)
            boot_metrics[name]["f1"].append(f1_b)
            b_aucs[name] = auc_b

        diff_gb_rf.append(b_aucs["Gradient Boosting"] - b_aucs["Random Forest"])
        diff_gb_lr.append(b_aucs["Gradient Boosting"] - b_aucs["Logistic Regression"])

    ci_table = []
    for name in models:
        aucs = np.array(boot_metrics[name]["auc"])
        f1s = np.array(boot_metrics[name]["f1"])
        ci_table.append({
            "Model": name,
            "Mean_Test_AUC": float(np.mean(aucs)),
            "AUC_95_CI_Lower": float(np.percentile(aucs, 2.5)),
            "AUC_95_CI_Upper": float(np.percentile(aucs, 97.5)),
            "Mean_Test_F1": float(np.mean(f1s)),
            "F1_95_CI_Lower": float(np.percentile(f1s, 2.5)),
            "F1_95_CI_Upper": float(np.percentile(f1s, 97.5)),
        })

    ci_df = pd.DataFrame(ci_table).sort_values("Mean_Test_AUC", ascending=False)
    ci_df.to_csv(eval_dir / "bootstrap_ci.csv", index=False)

    print("\n  === BOOTSTRAP 95% CONFIDENCE INTERVALS ===")
    print(ci_df.to_string(index=False, formatters={
        "Mean_Test_AUC": "{:.4f}".format,
        "AUC_95_CI_Lower": "{:.4f}".format,
        "AUC_95_CI_Upper": "{:.4f}".format,
        "Mean_Test_F1": "{:.4f}".format,
        "F1_95_CI_Lower": "{:.4f}".format,
        "F1_95_CI_Upper": "{:.4f}".format,
    }))

    # Statistical significance of GB superiority
    diff_rf = np.array(diff_gb_rf)
    diff_lr = np.array(diff_gb_lr)
    p_rf_better = np.mean(diff_rf <= 0)
    p_lr_better = np.mean(diff_lr <= 0)
    print(f"\n  Statistical Significance Testing (Paired Bootstrap):")
    print(f"    GB vs Random Forest: Delta AUC = +{np.mean(diff_rf):.4f} (95% CI: [{np.percentile(diff_rf, 2.5):.4f}, {np.percentile(diff_rf, 97.5):.4f}]), p-value = {p_rf_better:.4f}")
    print(f"    GB vs Logistic Reg : Delta AUC = +{np.mean(diff_lr):.4f} (95% CI: [{np.percentile(diff_lr, 2.5):.4f}, {np.percentile(diff_lr, 97.5):.4f}]), p-value = {p_lr_better:.4f}")

    # 7. Physical Error Profiling (FP vs FN)
    print("\n[6] Environmental Error Profiling (Gradient Boosting Residuals)...")
    yp_gb = (p_gb >= 0.37).astype(int)
    test_eval_df = test_spatial.copy()
    test_eval_df["pred"] = yp_gb
    test_eval_df["prob"] = p_gb

    # Assign outcome categories
    conditions = [
        (test_eval_df["class"] == 1) & (test_eval_df["pred"] == 1),
        (test_eval_df["class"] == 0) & (test_eval_df["pred"] == 0),
        (test_eval_df["class"] == 0) & (test_eval_df["pred"] == 1),
        (test_eval_df["class"] == 1) & (test_eval_df["pred"] == 0),
    ]
    choices = ["True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"]
    test_eval_df["outcome"] = np.select(conditions, choices, default="Other")

    error_summary = test_eval_df.groupby("outcome").agg(
        Count=("sample_id", "count"),
        Elevation_m=("elevation_m", "mean"),
        Slope_deg=("slope_deg", "mean"),
        Rainfall_z=("Rainfall_z", "mean"),
        NDVI_z=("NDVI_z", "mean"),
    ).reset_index()

    print(error_summary.to_string(index=False, formatters={
        "Elevation_m": "{:.1f} m".format,
        "Slope_deg": "{:.1f}°".format,
        "Rainfall_z": "{:+.3f}".format,
        "NDVI_z": "{:+.3f}".format,
    }))

    # Plot Error Profile
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    palette = {"True Positive (TP)": "#d95f02", "True Negative (TN)": "#1b9e77", "False Positive (FP)": "#e7298a", "False Negative (FN)": "#7570b3"}

    sns.boxplot(data=test_eval_df, x="outcome", y="elevation_m", palette=palette, ax=axes[0], order=choices)
    axes[0].set_title("Elevation Profile by Prediction Outcome", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Elevation (m)", fontsize=10)
    axes[0].tick_params(axis="x", rotation=15)

    sns.boxplot(data=test_eval_df, x="outcome", y="slope_deg", palette=palette, ax=axes[1], order=choices)
    axes[1].set_title("Slope Profile by Prediction Outcome", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Slope (degrees)", fontsize=10)
    axes[1].tick_params(axis="x", rotation=15)

    fig.suptitle("RockWatch - Error Profiling for Gradient Boosting on Test Split", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    p_err = fig_dir / "error_analysis_profile.png"
    fig.savefig(p_err, bbox_inches="tight")
    print(f"  Saved error profile plot: {p_err.name}")

    # 8. Save Detailed Metrics
    detailed_metrics = {
        "calibration": calib_results,
        "operational_policies": {
            "early_warning": ew_row.to_dict(),
            "balanced": bal_row.to_dict(),
            "infrastructure": infra_row.to_dict(),
        },
        "statistical_tests": {
            "gb_vs_rf": {"delta_auc": float(np.mean(diff_rf)), "ci_95": [float(np.percentile(diff_rf, 2.5)), float(np.percentile(diff_rf, 97.5))], "p_value": float(p_rf_better)},
            "gb_vs_lr": {"delta_auc": float(np.mean(diff_lr)), "ci_95": [float(np.percentile(diff_lr, 2.5)), float(np.percentile(diff_lr, 97.5))], "p_value": float(p_lr_better)},
        },
    }

    with open(eval_dir / "test_metrics_detailed.json", "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2)

    print("\nPhase 5 evaluation execution complete.")


if __name__ == "__main__":
    run_model_evaluation()
