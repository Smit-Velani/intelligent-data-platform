"""
explainer.py — Model explainability: SHAP global/local explanations, LIME
cross-check, and calibration (reliability) curves.

Key design decisions worth defending in an interview:

1. EXPLAINER CHOICE DEPENDS ON THE WINNING MODEL, NOT A FIXED ASSUMPTION.
   model_selector.py picks the model with the lowest expected business cost —
   that could be XGBoost/RandomForest (tree-based) OR LogisticRegression
   (linear) OR a NeuralNetwork/SVM (neither). Each needs a different SHAP
   explainer:
     - Tree models   -> shap.TreeExplainer   (exact, O(n) per tree, fast)
     - Linear models -> shap.LinearExplainer (closed-form, fast)
     - Anything else -> shap.KernelExplainer (model-agnostic but O(2^features)
       in the worst case — NEVER run this on the full dataset or with a full
       background set, or it will time out. It is deliberately bounded here:
       background summarized via shap.kmeans to ~30-50 synthetic points, and
       only a small sample of rows (default 100) is explained.)

2. SHAP summary/bar plots are computed over a SAMPLE of the test set
   (default 500 rows), not the full 284,807-row fraud dataset. Beyond a
   few hundred points a beeswarm plot is visually unreadable anyway, and
   for KernelExplainer specifically, running on the full dataset is not
   just slow but computationally infeasible.

3. LIME is used as a spot-check on individual predictions, not the primary
   explainer. SHAP has consistent, additive attributions (Shapley values
   sum to the prediction); LIME approximates locally with a linear surrogate
   model, which is a useful sanity check when the two disagree sharply on a
   given instance, but LIME's attributions are not guaranteed to be additive
   or globally consistent the way SHAP's are.

4. Calibration matters separately from discrimination. A model can have
   great AUC-PR/AUC-ROC (ranks positives above negatives well) while still
   outputting poorly calibrated probabilities (e.g. always predicting 0.9
   for anything it thinks is fraud, regardless of true risk). Reliability
   curves + Brier score catch this; AUC alone does not.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering for a server environment
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


TREE_MODEL_NAMES = {"RandomForestClassifier", "XGBClassifier", "RandomForestRegressor", "XGBRegressor"}
LINEAR_MODEL_NAMES = {"LogisticRegression", "LinearRegression"}

MAX_SHAP_SAMPLE = 500          # rows used for global summary/bar plots
MAX_KERNEL_EXPLAIN_ROWS = 100  # rows explained when falling back to KernelExplainer
KERNEL_BACKGROUND_SIZE = 40    # kmeans-summarized background points for KernelExplainer


def _unwrap_estimator(fitted_model):
    """
    model_selector.py may hand back an imblearn Pipeline (SMOTE + model) or a
    raw sklearn/xgboost estimator. SHAP needs the actual fitted predictor,
    not the SMOTE resampling step, so unwrap if necessary.
    """
    if hasattr(fitted_model, "named_steps") and "model" in getattr(fitted_model, "named_steps", {}):
        return fitted_model.named_steps["model"]
    return fitted_model


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@dataclass
class ExplainabilityReport:
    explainer_type: str                    # "tree" | "linear" | "kernel" | "unavailable"
    global_importance: list                # [{feature, mean_abs_shap}], sorted desc
    summary_plot_base64: Optional[str]
    bar_plot_base64: Optional[str]
    calibration_plot_base64: Optional[str]
    brier_score: Optional[float]
    notes: list


def _choose_shap_explainer(estimator, X_background: np.ndarray):
    """Returns (explainer, explainer_type_str, notes)."""
    notes = []
    model_class = type(estimator).__name__

    if not HAS_SHAP:
        notes.append("shap not installed — run `pip install shap` (see requirements.txt).")
        return None, "unavailable", notes

    if model_class in TREE_MODEL_NAMES:
        notes.append(f"{model_class} is tree-based -> using shap.TreeExplainer (exact, fast).")
        return shap.TreeExplainer(estimator), "tree", notes

    if model_class in LINEAR_MODEL_NAMES:
        notes.append(f"{model_class} is linear -> using shap.LinearExplainer (closed-form).")
        try:
            explainer = shap.LinearExplainer(estimator, X_background)
            return explainer, "linear", notes
        except Exception as e:
            notes.append(f"LinearExplainer failed ({e}); falling back to KernelExplainer.")

    notes.append(
        f"{model_class} is neither tree nor linear -> using shap.KernelExplainer. "
        f"This is model-agnostic but exponential in the worst case, so the background "
        f"is summarized to {KERNEL_BACKGROUND_SIZE} points via k-means and only up to "
        f"{MAX_KERNEL_EXPLAIN_ROWS} rows are explained, not the full dataset."
    )
    background_summary = shap.kmeans(X_background, min(KERNEL_BACKGROUND_SIZE, len(X_background)))
    predict_fn = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
    explainer = shap.KernelExplainer(predict_fn, background_summary)
    return explainer, "kernel", notes


def generate_explainability_report(
    fitted_model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray],
    feature_names: list,
    problem_type: str = "classification",
) -> ExplainabilityReport:
    estimator = _unwrap_estimator(fitted_model)
    explainer, explainer_type, notes = _choose_shap_explainer(estimator, X_train)

    global_importance = []
    summary_b64 = None
    bar_b64 = None

    if explainer is not None:
        sample_size = min(MAX_SHAP_SAMPLE if explainer_type != "kernel" else MAX_KERNEL_EXPLAIN_ROWS, len(X_test))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), size=sample_size, replace=False)
        X_sample = X_test[idx]

        shap_values = explainer.shap_values(X_sample)
        # Binary classifiers with TreeExplainer return a list [class0, class1] in
        # older shap versions, or a single 2D array for class 1 in newer ones.
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        global_importance = [
            {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
            for i in order
        ]

        # Bar chart: top 15 features by mean |SHAP value|
        top_n = min(15, len(feature_names))
        fig, ax = plt.subplots(figsize=(7, 5))
        top = global_importance[:top_n][::-1]
        ax.barh([t["feature"] for t in top], [t["mean_abs_shap"] for t in top], color="#4C72B0")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title("Global Feature Importance (SHAP)")
        bar_b64 = _fig_to_base64(fig)

        # Summary/beeswarm-style plot via shap's own plotting if available
        try:
            fig2 = plt.figure(figsize=(7, 5))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            summary_b64 = _fig_to_base64(fig2)
        except Exception as e:
            notes.append(f"shap.summary_plot failed ({e}); bar chart still available.")

    # Calibration curve — only meaningful for classification with probability outputs
    calibration_b64 = None
    brier = None
    if problem_type == "classification" and y_test is not None and hasattr(estimator, "predict_proba"):
        try:
            y_proba = estimator.predict_proba(X_test)[:, 1]
            brier = round(float(brier_score_loss(y_test, y_proba)), 4)
            frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")

            fig3, ax3 = plt.subplots(figsize=(5.5, 5.5))
            ax3.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
            ax3.plot(mean_pred, frac_pos, marker="o", color="#DD8452", label="Model")
            ax3.set_xlabel("Mean predicted probability")
            ax3.set_ylabel("Fraction of positives")
            ax3.set_title(f"Reliability Curve (Brier score = {brier})")
            ax3.legend()
            calibration_b64 = _fig_to_base64(fig3)
        except Exception as e:
            notes.append(f"Calibration curve failed ({e}).")

    return ExplainabilityReport(
        explainer_type=explainer_type,
        global_importance=global_importance,
        summary_plot_base64=summary_b64,
        bar_plot_base64=bar_b64,
        calibration_plot_base64=calibration_b64,
        brier_score=brier,
        notes=notes,
    )


def explain_single_prediction(
    fitted_model,
    X_train: np.ndarray,
    X_row: np.ndarray,
    feature_names: list,
    top_n: int = 8,
) -> dict:
    """
    Per-prediction SHAP explanation (the 'why did the model flag THIS
    transaction' view), plus a LIME cross-check when available.
    """
    estimator = _unwrap_estimator(fitted_model)
    explainer, explainer_type, notes = _choose_shap_explainer(estimator, X_train)

    result = {"explainer_type": explainer_type, "shap_contributions": [], "lime_contributions": [], "notes": notes}
    if explainer is None:
        return result

    X_row_2d = X_row.reshape(1, -1)
    shap_values = explainer.shap_values(X_row_2d)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    contributions = shap_values[0]

    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    result["shap_contributions"] = [
        {"feature": feature_names[i], "shap_value": round(float(contributions[i]), 6), "value": round(float(X_row[i]), 4)}
        for i in order
    ]

    if HAS_LIME:
        try:
            lime_explainer = LimeTabularExplainer(
                X_train, feature_names=feature_names, class_names=["not_fraud", "fraud"],
                mode="classification", discretize_continuous=True,
            )
            predict_fn = estimator.predict_proba
            lime_exp = lime_explainer.explain_instance(X_row, predict_fn, num_features=top_n)
            result["lime_contributions"] = [{"feature": f, "weight": round(float(w), 6)} for f, w in lime_exp.as_list()]
        except Exception as e:
            result["notes"].append(f"LIME explanation failed ({e}).")
    else:
        result["notes"].append("lime not installed — run `pip install lime` for the cross-check view.")

    return result
