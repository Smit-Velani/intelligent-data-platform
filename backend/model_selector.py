"""
model_selector.py — Automated multi-model training, cost-aware selection, and
a human-readable decision log explaining WHY the winning model won.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, confusion_matrix, mean_squared_error, r2_score,
    brier_score_loss,
)
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

warnings.filterwarnings("ignore")

SVM_ROW_LIMIT = 20_000  # above this, SVM training time becomes impractical

# Speed-scaling thresholds for large datasets.
SMOTE_ROW_LIMIT = 50_000
LARGE_DATA_ROWS = 50_000


@dataclass
class CostMatrix:
    false_negative_cost: float = 500.0
    false_positive_cost: float = 15.0
    true_positive_benefit: float = 0.0
    true_negative_benefit: float = 0.0
    min_recall: float = 0.5

    def expected_cost(self, y_true, y_pred) -> float:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = (
            fn * self.false_negative_cost
            + fp * self.false_positive_cost
            - tp * self.true_positive_benefit
            - tn * self.true_negative_benefit
        )
        return float(cost)


@dataclass
class ModelResult:
    name: str
    metrics: dict
    fit_time_seconds: float
    excluded_reason: Optional[str] = None


@dataclass
class SelectionResult:
    problem_type: str
    results: list
    best_model_name: str
    best_estimator: object
    decision_log: str
    leaderboard: list = field(default_factory=list)


def _make_smote_pipeline(estimator, use_smote: bool):
    if use_smote and HAS_IMBLEARN:
        return ImbPipeline(steps=[("smote", SMOTE(random_state=42)), ("model", estimator)])
    return estimator


def get_classification_zoo(scale_pos_weight: float, n_rows: int) -> dict:
    zoo = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=42,
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
            random_state=42,
        ),
    }
    if HAS_XGBOOST:
        zoo["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            use_label_encoder=False, n_jobs=-1, random_state=42,
        )
    if n_rows <= SVM_ROW_LIMIT:
        zoo["SVM"] = SVC(probability=True, class_weight="balanced", random_state=42)
    return zoo


def get_regression_zoo() -> dict:
    zoo = {
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
        "NeuralNetwork": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42),
    }
    if HAS_XGBOOST:
        zoo["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    return zoo


def _evaluate_classification(estimator, X_train, y_train, cv_splitter, use_smote: bool) -> dict:
    pipeline = _make_smote_pipeline(estimator, use_smote)
    y_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv_splitter, method="predict_proba", n_jobs=-1
    )[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "auc_roc": round(float(roc_auc_score(y_train, y_proba)), 4),
        "auc_pr": round(float(average_precision_score(y_train, y_proba)), 4),
        "f1": round(float(f1_score(y_train, y_pred)), 4),
        "precision": round(float(precision_score(y_train, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_train, y_pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_train, y_proba)), 4),
        "y_proba_cv": y_proba,
        "y_pred_cv": y_pred,
    }


def _evaluate_regression(estimator, X_train, y_train, cv_splitter) -> dict:
    y_pred = cross_val_predict(estimator, X_train, y_train, cv=cv_splitter, n_jobs=-1)
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_train, y_pred))), 4),
        "r2": round(float(r2_score(y_train, y_pred)), 4),
    }


def run_selection(
    preprocess_result,
    cost_matrix: Optional[CostMatrix] = None,
    use_smote: bool = True,
) -> SelectionResult:
    problem_type = preprocess_result.problem_type
    X_train, y_train = preprocess_result.X_train, preprocess_result.y_train
    cv_splitter = preprocess_result.cv_splitter
    n_rows = X_train.shape[0]
    cost_matrix = cost_matrix or CostMatrix()

    results: list[ModelResult] = []
    log_lines = [f"Model Selection Decision Log — problem_type={problem_type}, n_rows={n_rows}\n" + "=" * 60]

    # --- Speed scaling for large datasets ---
    speed_notes = []
    if problem_type == "classification" and use_smote and n_rows > SMOTE_ROW_LIMIT:
        use_smote = False
        speed_notes.append(
            f"SMOTE auto-disabled: {n_rows:,} training rows exceeds the {SMOTE_ROW_LIMIT:,}-row "
            f"threshold where SMOTE-in-fold becomes the dominant cost (it would synthesize tens of "
            f"thousands of rows per fold). Imbalance is instead handled by class_weight='balanced' "
            f"and scale_pos_weight, which are effectively free at this scale."
        )

    if n_rows > LARGE_DATA_ROWS and cv_splitter is not None and getattr(cv_splitter, "n_splits", 5) > 3:
        from sklearn.model_selection import StratifiedKFold, KFold
        random_state = getattr(cv_splitter, "random_state", 42)
        if problem_type == "classification":
            cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        else:
            cv_splitter = KFold(n_splits=3, shuffle=True, random_state=random_state)
        speed_notes.append(
            f"Cross-validation reduced to 3 folds (from 5) because {n_rows:,} rows is large enough "
            f"that 3-fold gives a stable estimate at ~40% less compute."
        )

    if speed_notes:
        log_lines.append("\nSpeed scaling (large dataset):")
        for note in speed_notes:
            log_lines.append(f"  - {note}")

    if problem_type == "classification":
        zoo = get_classification_zoo(preprocess_result.scale_pos_weight or 1.0, n_rows)

        if n_rows > SVM_ROW_LIMIT:
            log_lines.append(
                f"SVM excluded: {n_rows} rows exceeds the {SVM_ROW_LIMIT}-row practical "
                f"limit for SVM's O(n^2)-O(n^3) training complexity. Kept for datasets "
                f"below that threshold."
            )

        scored = []
        for name, estimator in zoo.items():
            t0 = time.time()
            metrics = _evaluate_classification(estimator, X_train, y_train, cv_splitter, use_smote)
            fit_time = round(time.time() - t0, 2)

            cost = cost_matrix.expected_cost(y_train, metrics["y_pred_cv"])
            metrics_public = {k: v for k, v in metrics.items() if k not in ("y_proba_cv", "y_pred_cv")}
            metrics_public["expected_cost_usd"] = round(cost, 2)

            results.append(ModelResult(name=name, metrics=metrics_public, fit_time_seconds=fit_time))
            scored.append((name, metrics_public, fit_time))

        qualified = [s for s in scored if s[1]["recall"] >= cost_matrix.min_recall]
        disqualified_note = None
        if not qualified:
            disqualified_note = (
                f"No model met the min_recall={cost_matrix.min_recall} floor. "
                f"Falling back to the highest-recall model available; treat this "
                f"as a signal the dataset/features need more work, not a deployable result."
            )
            qualified = sorted(scored, key=lambda t: -t[1]["recall"])[:1] + \
                        [s for s in scored if s[0] != sorted(scored, key=lambda t: -t[1]["recall"])[0][0]]

        qualified.sort(key=lambda t: t[1]["expected_cost_usd"])
        lowest_cost = qualified[0][1]["expected_cost_usd"]
        contenders = [s for s in qualified if s[1]["expected_cost_usd"] <= lowest_cost * 1.02 or abs(s[1]["expected_cost_usd"] - lowest_cost) < 1]
        contenders.sort(key=lambda t: t[2])
        best_name = contenders[0][0]

        log_lines.append("\nLeaderboard (sorted by expected business cost, lower is better):")
        for name, m, ft in scored:
            flag = "" if m["recall"] >= cost_matrix.min_recall else "  [BELOW MIN RECALL — disqualified]"
            log_lines.append(
                f"  {name:<18} cost=${m['expected_cost_usd']:>10,.2f}  "
                f"AUC-PR={m['auc_pr']:.4f}  AUC-ROC={m['auc_roc']:.4f}  "
                f"F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
                f"fit={ft}s{flag}"
            )
        if disqualified_note:
            log_lines.append(f"\nNOTE: {disqualified_note}")

        winner_metrics = next(m for n, m, _ in scored if n == best_name)
        runner_up = next(((n, m, ft) for n, m, ft in qualified if n != best_name), None)
        log_lines.append(f"\nWINNER: {best_name}")
        if winner_metrics["recall"] >= cost_matrix.min_recall:
            log_lines.append(
                f"Reason: lowest expected business cost (${winner_metrics['expected_cost_usd']:,.2f}) "
                f"among models meeting the min_recall={cost_matrix.min_recall} floor "
                f"(false_negative_cost=${cost_matrix.false_negative_cost}, "
                f"false_positive_cost=${cost_matrix.false_positive_cost})."
            )
        else:
            log_lines.append(
                f"Reason: FALLBACK — no model met min_recall={cost_matrix.min_recall}; "
                f"{best_name} had the highest recall ({winner_metrics['recall']:.4f}) among a bad field. "
                f"Do not deploy; revisit features/imbalance handling first."
            )
        if runner_up:
            gap = runner_up[1]["expected_cost_usd"] - winner_metrics["expected_cost_usd"]
            if gap == 0:
                log_lines.append(
                    f"Tied on expected cost with {runner_up[0]} (both ${winner_metrics['expected_cost_usd']:,.2f}); "
                    f"{best_name} won the tie-break for faster fit time."
                )
            else:
                comparator = "cheaper" if gap >= 0 else "more expensive"
                log_lines.append(
                    f"${abs(gap):,.2f} {comparator} than the next qualifying model ({runner_up[0]}), "
                    f"which scored {runner_up[1]['auc_pr']:.4f} AUC-PR vs {winner_metrics['auc_pr']:.4f}."
                )
        cheaper_disqualified = [s for s in scored if s[1]["recall"] < cost_matrix.min_recall and s[1]["expected_cost_usd"] < winner_metrics["expected_cost_usd"]]
        if cheaper_disqualified:
            names = ", ".join(f"{n} (${m['expected_cost_usd']:,.2f})" for n, m, _ in cheaper_disqualified)
            log_lines.append(
                f"Note: {names} scored lower raw cost but were disqualified for recall < {cost_matrix.min_recall} "
                f"— the recall floor overrode pure cost minimization here by design."
            )

        best_estimator_raw = zoo[best_name]
        best_pipeline = _make_smote_pipeline(best_estimator_raw, use_smote)
        best_pipeline.fit(X_train, y_train)

        leaderboard = [{"model": n, **m} for n, m, _ in scored]

        return SelectionResult(
            problem_type=problem_type,
            results=results,
            best_model_name=best_name,
            best_estimator=best_pipeline,
            decision_log="\n".join(log_lines),
            leaderboard=leaderboard,
        )

    elif problem_type == "regression":
        zoo = {k: v for k, v in get_regression_zoo().items() if v is not None}
        scored = []
        for name, estimator in zoo.items():
            t0 = time.time()
            metrics = _evaluate_regression(estimator, X_train, y_train, cv_splitter)
            fit_time = round(time.time() - t0, 2)
            results.append(ModelResult(name=name, metrics=metrics, fit_time_seconds=fit_time))
            scored.append((name, metrics, fit_time))

        scored.sort(key=lambda t: t[1]["rmse"])
        best_name = scored[0][0]

        log_lines.append("\nLeaderboard (sorted by RMSE, lower is better):")
        for name, m, ft in scored:
            log_lines.append(f"  {name:<18} RMSE={m['rmse']:.4f}  R2={m['r2']:.4f}  fit={ft}s")
        log_lines.append(f"\nWINNER: {best_name} — lowest RMSE ({scored[0][1]['rmse']:.4f}).")

        best_estimator = zoo[best_name]
        best_estimator.fit(X_train, y_train)

        return SelectionResult(
            problem_type=problem_type,
            results=results,
            best_model_name=best_name,
            best_estimator=best_estimator,
            decision_log="\n".join(log_lines),
            leaderboard=[{"model": n, **m} for n, m, _ in scored],
        )

    else:
        raise ValueError(
            "run_selection() only handles classification/regression. "
            "Clustering is scored differently (silhouette) — see clustering path in main.py."
        )