"""
RockWatch - Phase 3: Train & Evaluate Baseline Models
=====================================================
Trains and evaluates three baseline models:
1. DummyClassifier (lower-bound prior baseline)
2. LogisticRegression (L2 regularized, 16 features, interpretable linear model)
3. DecisionTreeClassifier (shallow tree, max_depth=5, 17 features, interpretable nonlinear model)

Saves:
- Trained models in models/trained/ (*.joblib)
- Metrics in models/evaluation/baseline_results.csv & baseline_metrics.json
- ROC curves in reports/figures/roc_curves_baselines.png
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_processed_data
from src.evaluation.metrics import evaluate_predictions, find_optimal_threshold
from src.models.baseline import (
    create_dummy_baseline,
    create_logistic_regression,
    create_decision_tree,
    ALL_FEATURES,
    LR_FEATURES,
)


def run_baseline_training():
    print("=" * 80)
    print("ROCKWATCH - PHASE 3: BASELINE MODEL TRAINING & EVALUATION")
    print("=" * 80)

    # 1. Load Data
    print("\n[1] Loading processed feature matrices...")
    f_train = load_processed_data("train", features_only=True)
    f_val = load_processed_data("val", features_only=True)
    f_test = load_processed_data("test", features_only=True)

    y_train = f_train["class"].values
    y_val = f_val["class"].values
    y_test = f_test["class"].values

    print(f"  Train: {len(f_train):,} rows | class1: {(y_train==1).sum():,} ({np.mean(y_train)*100:.1f}%)")
    print(f"  Val  : {len(f_val):,} rows | class1: {(y_val==1).sum():,} ({np.mean(y_val)*100:.1f}%)")
    print(f"  Test : {len(f_test):,} rows | class1: {(y_test==1).sum():,} ({np.mean(y_test)*100:.1f}%)")

    # 2. Define Models
    models_config = {
        "Dummy (Prior)": {
            "model": create_dummy_baseline(strategy="prior"),
            "features": ALL_FEATURES,
            "filename": "dummy_baseline.joblib",
        },
        "Logistic Regression": {
            "model": create_logistic_regression(C=1.0, max_iter=1000, random_state=42),
            "features": LR_FEATURES,
            "filename": "logistic_regression_v1.joblib",
        },
        "Decision Tree (d=5)": {
            "model": create_decision_tree(max_depth=5, min_samples_leaf=20, random_state=42),
            "features": ALL_FEATURES,
            "filename": "decision_tree_v1.joblib",
        },
    }

    results_table = []
    full_metrics = {}
    roc_data = {}

    models_dir = PROJECT_ROOT / "models" / "trained"
    eval_dir = PROJECT_ROOT / "models" / "evaluation"
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 3. Train & Evaluate Each Baseline
    print("\n[2] Training and evaluating baseline models...")

    for name, cfg in models_config.items():
        print(f"\n--- {name} ---")
        clf = cfg["model"]
        feats = cfg["features"]
        X_tr = f_train[feats].values
        X_va = f_val[feats].values
        X_te = f_test[feats].values

        # Fit model
        clf.fit(X_tr, y_train)

        # Predict probabilities
        p_train = clf.predict_proba(X_tr)[:, 1]
        p_val = clf.predict_proba(X_va)[:, 1]
        p_test = clf.predict_proba(X_te)[:, 1]

        # Find optimal threshold on validation set (maximizing F1)
        opt_thresh, opt_f1 = find_optimal_threshold(y_val, p_val, metric="f1")
        print(f"  Validation Optimal Threshold: {opt_thresh:.2f} (Val F1: {opt_f1:.4f})")

        # Evaluate on all splits
        m_train_def = evaluate_predictions(y_train, p_train, threshold=0.50)
        m_val_def = evaluate_predictions(y_val, p_val, threshold=0.50)
        m_val_opt = evaluate_predictions(y_val, p_val, threshold=opt_thresh)
        m_test_def = evaluate_predictions(y_test, p_test, threshold=0.50)
        m_test_opt = evaluate_predictions(y_test, p_test, threshold=opt_thresh)

        # Store ROC curve data
        roc_data[name] = {
            "val": (y_val, p_val),
            "test": (y_test, p_test),
        }

        # Save trained model
        save_path = models_dir / cfg["filename"]
        joblib.dump(clf, save_path)
        print(f"  Saved model: {save_path.name}")

        # Record summary
        results_table.append({
            "Model": name,
            "Train_AUC": m_train_def["auc_roc"],
            "Val_AUC": m_val_def["auc_roc"],
            "Test_AUC": m_test_def["auc_roc"],
            "Val_PR_AUC": m_val_def["pr_auc"],
            "Test_PR_AUC": m_test_def["pr_auc"],
            "Val_F1_opt": m_val_opt["f1"],
            "Test_F1_opt": m_test_opt["f1"],
            "Test_MCC_opt": m_test_opt["mcc"],
            "Test_Accuracy_opt": m_test_opt["accuracy"],
            "Opt_Threshold": opt_thresh,
        })

        full_metrics[name] = {
            "features": feats,
            "optimal_threshold": opt_thresh,
            "train_default_thresh_0.50": m_train_def,
            "val_default_thresh_0.50": m_val_def,
            "val_optimal_thresh": m_val_opt,
            "test_default_thresh_0.50": m_test_def,
            "test_optimal_thresh": m_test_opt,
        }

    # 4. Display Results Summary Table
    print("\n" + "=" * 80)
    print("BASELINE PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 80)
    res_df = pd.DataFrame(results_table)
    print(res_df.to_string(index=False, formatters={
        "Train_AUC": "{:.4f}".format,
        "Val_AUC": "{:.4f}".format,
        "Test_AUC": "{:.4f}".format,
        "Val_PR_AUC": "{:.4f}".format,
        "Test_PR_AUC": "{:.4f}".format,
        "Val_F1_opt": "{:.4f}".format,
        "Test_F1_opt": "{:.4f}".format,
        "Test_MCC_opt": "{:.4f}".format,
        "Test_Accuracy_opt": "{:.4f}".format,
        "Opt_Threshold": "{:.2f}".format,
    }))

    # 5. Save Artifacts
    p_csv = eval_dir / "baseline_results.csv"
    res_df.to_csv(p_csv, index=False)
    print(f"\nSaved benchmark table: {p_csv.name}")

    p_json = eval_dir / "baseline_metrics.json"
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"Saved complete metrics: {p_json.name}")

    # 6. Plot ROC Curves
    print("\n[3] Generating ROC curves plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    colors = {"Dummy (Prior)": "gray", "Logistic Regression": "#2b83ba", "Decision Tree (d=5)": "#d7191c"}
    styles = {"Dummy (Prior)": ":", "Logistic Regression": "-", "Decision Tree (d=5)": "-"}

    # Validation ROC
    ax1 = axes[0]
    for name, data in roc_data.items():
        y_true, y_prob = data["val"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=colors[name], linestyle=styles[name], linewidth=2.0)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.0)
    ax1.set_title("Validation Split ROC Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontsize=10)
    ax1.set_ylabel("True Positive Rate", fontsize=10)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Test ROC
    ax2 = axes[1]
    for name, data in roc_data.items():
        y_true, y_prob = data["test"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=colors[name], linestyle=styles[name], linewidth=2.0)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.0)
    ax2.set_title("Held-out Spatial Test ROC Curves (Chamba)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("False Positive Rate", fontsize=10)
    ax2.set_ylabel("True Positive Rate", fontsize=10)
    ax2.legend(loc="lower right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("RockWatch - Baseline Models Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    p_roc = fig_dir / "roc_curves_baselines.png"
    fig.savefig(p_roc, bbox_inches="tight")
    print(f"Saved ROC curves plot: {p_roc.name}")

    # 7. Print Logistic Regression Coefficients
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION COEFFICIENT INTERPRETATION")
    print("=" * 80)
    lr_model = joblib.load(models_dir / "logistic_regression_v1.joblib")
    coef_df = pd.DataFrame({
        "Feature": LR_FEATURES,
        "Coefficient": lr_model.coef_[0],
        "Odds_Ratio": np.exp(lr_model.coef_[0]),
    }).sort_values("Coefficient", ascending=False)
    print(f"Intercept: {lr_model.intercept_[0]:.4f}")
    print(coef_df.to_string(index=False, formatters={"Coefficient": "{:+.4f}".format, "Odds_Ratio": "{:.4f}".format}))

    print("\nPhase 3 execution complete.")


if __name__ == "__main__":
    run_baseline_training()
