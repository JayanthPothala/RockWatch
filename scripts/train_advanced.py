"""
RockWatch - Phase 4: Advanced Model Training & Spatial Evaluation
==================================================================
Trains and benchmarks advanced models:
1. Random Forest Classifier (Bagged de-correlated tree ensemble)
2. Gradient Boosting Classifier (Boosted regularized decision trees)
3. Support Vector Machine (RBF Kernel with Platt probability scaling)

Performs:
- 5-Fold Spatial Block Cross-Validation on Training set (coordinates from train_spatial.csv)
- Hyperparameter validation against central HP validation partition
- Out-of-region evaluation on northern HP (Chamba) held-out test split
- Feature importance analysis across tree ensembles
- Comparative ROC & PR curve plotting against Phase 3 baselines
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_processed_data
from src.evaluation.metrics import evaluate_predictions, find_optimal_threshold
from src.evaluation.spatial_cv import SpatialBlockKFold
from src.models.advanced import (
    create_random_forest,
    create_gradient_boosting,
    create_svm_rbf,
)
from src.models.baseline import ALL_FEATURES, LR_FEATURES


def run_advanced_training():
    print("=" * 80)
    print("ROCKWATCH - PHASE 4: ADVANCED MODEL TRAINING & SPATIAL EVALUATION")
    print("=" * 80)

    # 1. Load Data
    print("\n[1] Loading processed feature matrices and spatial coordinates...")
    f_train = load_processed_data("train", features_only=True)
    f_val = load_processed_data("val", features_only=True)
    f_test = load_processed_data("test", features_only=True)

    train_spatial = load_processed_data("train", features_only=False)
    coords_train = train_spatial[["latitude", "longitude"]].values

    y_train = f_train["class"].values
    y_val = f_val["class"].values
    y_test = f_test["class"].values

    X_train = f_train[ALL_FEATURES].values
    X_val = f_val[ALL_FEATURES].values
    X_test = f_test[ALL_FEATURES].values

    print(f"  Train: {len(f_train):,} rows | Val: {len(f_val):,} rows | Test: {len(f_test):,} rows")

    # 2. Define Advanced Models
    advanced_models = {
        "Random Forest": {
            "model_fn": lambda: create_random_forest(n_estimators=300, max_depth=12, min_samples_leaf=10, random_state=42),
            "filename": "random_forest_v1.joblib",
        },
        "Gradient Boosting": {
            "model_fn": lambda: create_gradient_boosting(n_estimators=200, learning_rate=0.05, max_depth=4, min_samples_leaf=15, subsample=0.85, random_state=42),
            "filename": "gradient_boosting_v1.joblib",
        },
        "SVM (RBF Kernel)": {
            "model_fn": lambda: create_svm_rbf(C=1.0, gamma="scale", probability=True, random_state=42),
            "filename": "svm_rbf_v1.joblib",
        },
    }

    # 3. Spatial Block Cross-Validation (5 Folds on Train Split)
    print("\n[2] Executing 5-Fold Spatial Block Cross-Validation on Training Set...")
    spatial_cv = SpatialBlockKFold(n_splits=5, random_state=42)
    cv_scores = {name: [] for name in advanced_models}

    for fold, (tr_idx, va_idx) in enumerate(spatial_cv.split(X_train, y_train, coords_train), 1):
        X_tr_f, y_tr_f = X_train[tr_idx], y_train[tr_idx]
        X_va_f, y_va_f = X_train[va_idx], y_train[va_idx]

        for name, cfg in advanced_models.items():
            clf_fold = cfg["model_fn"]()
            clf_fold.fit(X_tr_f, y_tr_f)
            p_va_f = clf_fold.predict_proba(X_va_f)[:, 1]
            fold_auc = roc_auc_score(y_va_f, p_va_f)
            cv_scores[name].append(fold_auc)

    print("  Spatial Block CV Results (5 Folds):")
    for name, scores in cv_scores.items():
        print(f"    {name:20s}: Mean AUC = {np.mean(scores):.4f} (+/- {np.std(scores):.4f}) | Folds: {[round(s, 3) for s in scores]}")

    # 4. Train Full Models & Evaluate
    print("\n[3] Fitting final advanced models on full training set (N=8,188)...")
    models_dir = PROJECT_ROOT / "models" / "trained"
    eval_dir = PROJECT_ROOT / "models" / "evaluation"
    fig_dir = PROJECT_ROOT / "reports" / "figures"

    results_table = []
    full_metrics = {}
    roc_curves_dict = {}
    pr_curves_dict = {}

    for name, cfg in advanced_models.items():
        print(f"\n--- Training {name} ---")
        clf = cfg["model_fn"]()
        clf.fit(X_train, y_train)

        # Predict
        p_train = clf.predict_proba(X_train)[:, 1]
        p_val = clf.predict_proba(X_val)[:, 1]
        p_test = clf.predict_proba(X_test)[:, 1]

        # Calibrate optimal threshold on validation set
        opt_thresh, opt_f1 = find_optimal_threshold(y_val, p_val, metric="f1")
        print(f"  Optimal Validation Threshold: {opt_thresh:.2f} (Val F1: {opt_f1:.4f})")

        # Evaluate
        m_train = evaluate_predictions(y_train, p_train, threshold=0.50)
        m_val = evaluate_predictions(y_val, p_val, threshold=opt_thresh)
        m_test = evaluate_predictions(y_test, p_test, threshold=opt_thresh)

        # Save model
        save_path = models_dir / cfg["filename"]
        joblib.dump(clf, save_path)
        print(f"  Saved: {save_path.name}")

        roc_curves_dict[name] = {
            "val": (y_val, p_val),
            "test": (y_test, p_test),
        }
        pr_curves_dict[name] = {
            "val": (y_val, p_val),
            "test": (y_test, p_test),
        }

        results_table.append({
            "Model": name,
            "Spatial_CV_AUC": float(np.mean(cv_scores[name])),
            "Train_AUC": m_train["auc_roc"],
            "Val_AUC": m_val["auc_roc"],
            "Test_AUC": m_test["auc_roc"],
            "Val_PR_AUC": m_val["pr_auc"],
            "Test_PR_AUC": m_test["pr_auc"],
            "Val_F1_opt": m_val["f1"],
            "Test_F1_opt": m_test["f1"],
            "Test_MCC_opt": m_test["mcc"],
            "Test_Accuracy_opt": m_test["accuracy"],
            "Opt_Threshold": opt_thresh,
        })

        full_metrics[name] = {
            "spatial_cv_scores": cv_scores[name],
            "optimal_threshold": opt_thresh,
            "train_metrics": m_train,
            "val_metrics": m_val,
            "test_metrics": m_test,
        }

    # 5. Display Advanced Results Table
    print("\n" + "=" * 80)
    print("ADVANCED MODEL PERFORMANCE BENCHMARK")
    print("=" * 80)
    adv_df = pd.DataFrame(results_table)
    print(adv_df.to_string(index=False, formatters={
        "Spatial_CV_AUC": "{:.4f}".format,
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

    # Save advanced artifacts
    adv_df.to_csv(eval_dir / "advanced_results.csv", index=False)
    with open(eval_dir / "advanced_metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    # 6. Build Master All-Models Benchmark (Baselines + Advanced)
    base_df = pd.read_csv(eval_dir / "baseline_results.csv")
    base_df["Spatial_CV_AUC"] = np.nan
    all_models_df = pd.concat([base_df, adv_df], ignore_index=True).sort_values("Test_AUC", ascending=False)
    all_models_df.to_csv(eval_dir / "all_models_benchmark.csv", index=False)

    print("\n" + "=" * 80)
    print("OVERALL PROJECT BENCHMARK (ALL 6 MODELS RANKED BY TEST AUC)")
    print("=" * 80)
    print(all_models_df[["Model", "Val_AUC", "Test_AUC", "Test_PR_AUC", "Test_F1_opt", "Test_MCC_opt", "Test_Accuracy_opt"]].to_string(index=False, formatters={
        "Val_AUC": "{:.4f}".format,
        "Test_AUC": "{:.4f}".format,
        "Test_PR_AUC": "{:.4f}".format,
        "Test_F1_opt": "{:.4f}".format,
        "Test_MCC_opt": "{:.4f}".format,
        "Test_Accuracy_opt": "{:.4f}".format,
    }))

    # 7. Feature Importance Analysis
    print("\n[4] Generating Feature Importance Comparison Plot...")
    rf_clf = joblib.load(models_dir / "random_forest_v1.joblib")
    gb_clf = joblib.load(models_dir / "gradient_boosting_v1.joblib")

    imp_df = pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Random_Forest": rf_clf.feature_importances_,
        "Gradient_Boosting": gb_clf.feature_importances_,
    }).sort_values("Random_Forest", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x = np.arange(len(imp_df))
    width = 0.38

    ax.barh(x - width / 2, imp_df["Random_Forest"], height=width, label="Random Forest (Gini)", color="#2b83ba")
    ax.barh(x + width / 2, imp_df["Gradient_Boosting"], height=width, label="Gradient Boosting", color="#d7191c")

    ax.set_yticks(x)
    ax.set_yticklabels(imp_df["Feature"], fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance", fontsize=11, fontweight="bold")
    ax.set_title("Feature Importance Comparison: Random Forest vs Gradient Boosting", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    p_imp = fig_dir / "feature_importance_comparison.png"
    fig.savefig(p_imp, bbox_inches="tight")
    print(f"  Saved: {p_imp.name}")

    # 8. Comparative ROC & PR Curves for All Models
    print("\n[5] Generating Comparative ROC and Precision-Recall Curves for All Models...")
    # Load baselines
    lr_clf = joblib.load(models_dir / "logistic_regression_v1.joblib")
    dt_clf = joblib.load(models_dir / "decision_tree_v1.joblib")

    all_test_predictions = {
        "Logistic Regression": lr_clf.predict_proba(f_test[LR_FEATURES].values)[:, 1],
        "Decision Tree (d=5)": dt_clf.predict_proba(X_test)[:, 1],
        "Random Forest": rf_clf.predict_proba(X_test)[:, 1],
        "Gradient Boosting": gb_clf.predict_proba(X_test)[:, 1],
        "SVM (RBF Kernel)": joblib.load(models_dir / "svm_rbf_v1.joblib").predict_proba(X_test)[:, 1],
    }

    # Plot ROC curves on Test set
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    ax_roc, ax_pr = axes[0], axes[1]

    colors = {
        "Logistic Regression": "#74a9cf",
        "Decision Tree (d=5)": "#fdae61",
        "SVM (RBF Kernel)": "#abdda4",
        "Random Forest": "#2b83ba",
        "Gradient Boosting": "#d7191c",
    }

    for name, p_te in all_test_predictions.items():
        # ROC
        fpr, tpr, _ = roc_curve(y_test, p_te)
        roc_score = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_score:.3f})", color=colors[name], linewidth=2.2 if "Forest" in name or "Gradient" in name else 1.5)

        # PR
        prec, rec, _ = precision_recall_curve(y_test, p_te)
        pr_score = auc(rec, prec)
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {pr_score:.3f})", color=colors[name], linewidth=2.2 if "Forest" in name or "Gradient" in name else 1.5)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.0)
    ax_roc.set_title("Held-out Spatial Test ROC Curves (Chamba Corridor)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate", fontsize=10)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=9)
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    baseline_pr = np.mean(y_test)
    ax_pr.axhline(baseline_pr, color="k", linestyle="--", alpha=0.4, label=f"Baseline ({baseline_pr:.2f})")
    ax_pr.set_title("Held-out Spatial Test Precision-Recall Curves", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.legend(loc="lower left", frameon=True, fontsize=9)
    ax_pr.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("RockWatch - Overall Model Benchmark on Held-Out Spatial Test Set", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    p_curves = fig_dir / "roc_curves_all_models.png"
    fig.savefig(p_curves, bbox_inches="tight")
    print(f"  Saved: {p_curves.name}")

    print("\nPhase 4 execution complete.")


if __name__ == "__main__":
    run_advanced_training()
