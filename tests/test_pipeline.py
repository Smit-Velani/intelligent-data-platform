"""
test_pipeline.py — Automated tests for the IDP core ML modules.
Run with:  pytest -v
"""
import numpy as np
import pandas as pd
import pytest

from backend.preprocessor import (
    preprocess, detect_problem_type, compute_scale_pos_weight, compute_class_balance,
)
from backend.model_selector import CostMatrix
from backend.drift_detector import detect_drift, _compute_psi


@pytest.fixture
def imbalanced_df():
    rng = np.random.default_rng(0)
    n = 2000
    amount = rng.exponential(80, n)
    time_ = rng.integers(0, 86400, n)
    prob = 1 / (1 + np.exp(-(0.02 * (amount - 200))))
    label = (rng.random(n) < prob * 0.05).astype(int)
    return pd.DataFrame({"amount": amount, "time": time_, "is_fraud": label})


def test_detect_problem_type_binary():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    assert detect_problem_type(df, "target") == "classification"


def test_detect_problem_type_regression():
    df = pd.DataFrame({"x": range(50), "target": np.linspace(0, 100, 50)})
    assert detect_problem_type(df, "target") == "regression"


def test_detect_problem_type_clustering_when_no_target():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert detect_problem_type(df, None) == "clustering"


def test_scale_pos_weight_math():
    y = pd.Series([0] * 90 + [1] * 10)
    assert compute_scale_pos_weight(y) == pytest.approx(9.0)


def test_class_balance_sums_to_one():
    y = pd.Series([0] * 80 + [1] * 20)
    balance = compute_class_balance(y)
    assert sum(balance.values()) == pytest.approx(1.0, abs=1e-6)


def test_stratified_split_preserves_ratio(imbalanced_df):
    res = preprocess(imbalanced_df, target_col="is_fraud", n_splits=3)
    assert abs(res.y_train.mean() - res.y_test.mean()) < 0.01


def test_preprocess_detects_classification(imbalanced_df):
    res = preprocess(imbalanced_df, target_col="is_fraud", n_splits=3)
    assert res.problem_type == "classification"
    assert res.scale_pos_weight is not None
    assert res.scale_pos_weight > 1


def test_preprocess_handles_missing_values():
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "b": [1, 0, 1, 0, 1, 0],
    })
    res = preprocess(df, target_col="b", n_splits=2)
    assert not np.isnan(res.X_train).any()


def test_preprocess_encodes_categoricals():
    df = pd.DataFrame({
        "cat": ["x", "y", "z", "x", "y", "z"] * 4,
        "num": list(range(24)),
        "target": [0, 1] * 12,
    })
    res = preprocess(df, target_col="target", n_splits=2)
    assert res.X_train.shape[1] > 1
    assert np.issubdtype(res.X_train.dtype, np.number)


def test_cost_matrix_expected_cost():
    cm = CostMatrix(false_negative_cost=500, false_positive_cost=15)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    assert cm.expected_cost(y_true, y_pred) == pytest.approx(515)


def test_cost_matrix_perfect_prediction_zero_cost():
    cm = CostMatrix(false_negative_cost=500, false_positive_cost=15)
    y = np.array([0, 1, 0, 1])
    assert cm.expected_cost(y, y) == 0.0


def test_no_drift_on_identical_distributions():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (1000, 2))
    report = detect_drift(x, x.copy(), ["f1", "f2"])
    assert report.overall_status == "stable"
    assert report.drifted_feature_count == 0


def test_drift_detected_on_shifted_distribution():
    rng = np.random.default_rng(2)
    train = rng.normal(0, 1, (1000, 1))
    incoming = rng.normal(5, 1, (500, 1))
    report = detect_drift(train, incoming, ["shifted"])
    assert report.overall_status == "significant_drift"
    assert report.drifted_feature_count == 1


def test_psi_zero_for_same_data():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 2000)
    psi = _compute_psi(x, x)
    assert psi < 0.01