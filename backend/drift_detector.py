"""
drift_detector.py — Compares live/inference feature distributions against the
training distribution to flag when a deployed model may be going stale.

Two complementary tests per numeric feature:
  - PSI (Population Stability Index): industry-standard drift metric.
      PSI < 0.1  -> no significant drift
      0.1 - 0.25 -> moderate drift, monitor
      > 0.25     -> significant drift, retrain likely needed
  - KS-test (Kolmogorov-Smirnov): statistical test for whether two samples
    come from the same distribution; gives a p-value alongside PSI's score.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.stats import ks_2samp


PSI_BINS = 10
PSI_NO_DRIFT = 0.1
PSI_MODERATE_DRIFT = 0.25


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    status: str  # "stable" | "moderate_drift" | "significant_drift"


@dataclass
class DriftReport:
    features: list  # list[FeatureDrift]
    overall_status: str
    drifted_feature_count: int
    total_feature_count: int


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """
    PSI = sum over bins of (current% - reference%) * ln(current% / reference%)
    Bin edges are derived from the REFERENCE (training) distribution's quantiles,
    so bins reflect what "normal" looked like at training time.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # not enough variation to bin meaningfully

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _status_from_psi(psi: float) -> str:
    if psi < PSI_NO_DRIFT:
        return "stable"
    if psi < PSI_MODERATE_DRIFT:
        return "moderate_drift"
    return "significant_drift"


def detect_drift(
    X_train: np.ndarray,
    X_incoming: np.ndarray,
    feature_names: list,
) -> DriftReport:
    features = []
    for i, name in enumerate(feature_names):
        ref_col = X_train[:, i]
        cur_col = X_incoming[:, i]

        psi = _compute_psi(ref_col, cur_col)
        ks_stat, ks_p = ks_2samp(ref_col, cur_col)
        status = _status_from_psi(psi)

        features.append(FeatureDrift(
            feature=name, psi=round(psi, 4),
            ks_statistic=round(float(ks_stat), 4), ks_pvalue=round(float(ks_p), 6),
            status=status,
        ))

    drifted = [f for f in features if f.status != "stable"]
    significant = [f for f in features if f.status == "significant_drift"]

    if significant:
        overall = "significant_drift"
    elif drifted:
        overall = "moderate_drift"
    else:
        overall = "stable"

    return DriftReport(
        features=features,
        overall_status=overall,
        drifted_feature_count=len(drifted),
        total_feature_count=len(features),
    )
