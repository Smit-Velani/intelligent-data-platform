"""
data_quality.py — Pre-modeling data inspection and leakage detection.

Runs a battery of checks on the raw uploaded DataFrame BEFORE any modeling,
surfacing issues that would otherwise silently corrupt results:

  - Target leakage: features almost perfectly correlated with the target.
  - Missing data: columns with high missing rates.
  - Constant / near-constant columns: zero predictive value.
  - Duplicate rows.
  - High-cardinality categoricals: likely IDs, not features.
  - Outliers: numeric columns with heavy tails (IQR method).
  - Class imbalance severity (for classification targets).

Each finding has a severity: "high" | "medium" | "info". Nothing here mutates
the data — it only reports. Cleaning happens in preprocessor.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


LEAKAGE_CORR_THRESHOLD = 0.95
HIGH_MISSING_THRESHOLD = 0.30
MODERATE_MISSING_THRESHOLD = 0.05
HIGH_CARDINALITY_RATIO = 0.5
OUTLIER_IQR_MULTIPLIER = 1.5
SEVERE_IMBALANCE_RATIO = 100


@dataclass
class QualityFinding:
    check: str
    severity: str
    message: str
    columns: list = field(default_factory=list)


@dataclass
class QualityReport:
    n_rows: int
    n_cols: int
    findings: list
    high_count: int
    medium_count: int
    info_count: int
    quality_score: int


def _detect_leakage(df: pd.DataFrame, target_col: Optional[str]) -> list:
    findings = []
    if target_col is None or target_col not in df.columns:
        return findings

    numeric = df.select_dtypes(include=[np.number])
    if target_col not in numeric.columns:
        try:
            y = pd.factorize(df[target_col])[0]
        except Exception:
            return findings
    else:
        y = numeric[target_col].values

    for col in numeric.columns:
        if col == target_col:
            continue
        try:
            corr = np.corrcoef(numeric[col].fillna(numeric[col].median()), y)[0, 1]
        except Exception:
            continue
        if np.isnan(corr):
            continue
        if abs(corr) >= LEAKAGE_CORR_THRESHOLD:
            findings.append(QualityFinding(
                check="target_leakage",
                severity="high",
                message=f"'{col}' is {abs(corr):.2f} correlated with the target — likely leakage. "
                        f"If this feature would not be available at prediction time, remove it.",
                columns=[col],
            ))
    return findings


def _detect_missing(df: pd.DataFrame) -> list:
    findings = []
    miss = df.isna().mean()
    for col, rate in miss.items():
        if rate >= HIGH_MISSING_THRESHOLD:
            findings.append(QualityFinding(
                check="missing_data", severity="high",
                message=f"'{col}' is {rate*100:.0f}% missing — consider dropping it or imputing carefully.",
                columns=[col],
            ))
        elif rate >= MODERATE_MISSING_THRESHOLD:
            findings.append(QualityFinding(
                check="missing_data", severity="medium",
                message=f"'{col}' is {rate*100:.0f}% missing.",
                columns=[col],
            ))
    return findings


def _detect_constant(df: pd.DataFrame) -> list:
    findings = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=False)
        if nunique <= 1:
            findings.append(QualityFinding(
                check="constant_column", severity="medium",
                message=f"'{col}' has a single value — it carries no predictive signal.",
                columns=[col],
            ))
    return findings


def _detect_duplicates(df: pd.DataFrame) -> list:
    findings = []
    dup = int(df.duplicated().sum())
    if dup > 0:
        findings.append(QualityFinding(
            check="duplicate_rows", severity="medium" if dup > len(df) * 0.01 else "info",
            message=f"{dup:,} duplicate row(s) found ({dup/len(df)*100:.1f}% of data).",
            columns=[],
        ))
    return findings


def _detect_high_cardinality(df: pd.DataFrame, target_col: Optional[str]) -> list:
    findings = []
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in obj_cols:
        if col == target_col:
            continue
        ratio = df[col].nunique() / len(df)
        if ratio >= HIGH_CARDINALITY_RATIO:
            findings.append(QualityFinding(
                check="high_cardinality", severity="info",
                message=f"'{col}' has {df[col].nunique():,} unique values ({ratio*100:.0f}% of rows) — "
                        f"likely an ID or free-text column, not a useful feature.",
                columns=[col],
            ))
    return findings


def _detect_outliers(df: pd.DataFrame, target_col: Optional[str]) -> list:
    findings = []
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        if col == target_col:
            continue
        s = numeric[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        pct = ((s < lower) | (s > upper)).mean()
        if pct >= 0.10:
            findings.append(QualityFinding(
                check="outliers", severity="info",
                message=f"'{col}' has {pct*100:.0f}% outliers (IQR method) — consider scaling or transform.",
                columns=[col],
            ))
    return findings


def _detect_imbalance(df: pd.DataFrame, target_col: Optional[str]) -> list:
    findings = []
    if target_col is None or target_col not in df.columns:
        return findings
    vc = df[target_col].value_counts(dropna=True)
    if len(vc) < 2 or len(vc) > 20:
        return findings
    majority, minority = vc.iloc[0], vc.iloc[-1]
    if minority == 0:
        return findings
    ratio = majority / minority
    if ratio >= SEVERE_IMBALANCE_RATIO:
        findings.append(QualityFinding(
            check="class_imbalance", severity="high",
            message=f"Severe class imbalance ({ratio:.0f}:1). Accuracy will be misleading — "
                    f"use AUC-PR, recall, and cost-aware selection.",
            columns=[target_col],
        ))
    elif ratio >= 10:
        findings.append(QualityFinding(
            check="class_imbalance", severity="medium",
            message=f"Class imbalance ({ratio:.0f}:1). Consider resampling or class weights.",
            columns=[target_col],
        ))
    return findings


def analyze_quality(df: pd.DataFrame, target_col: Optional[str] = None) -> QualityReport:
    findings = []
    findings += _detect_leakage(df, target_col)
    findings += _detect_missing(df)
    findings += _detect_constant(df)
    findings += _detect_duplicates(df)
    findings += _detect_high_cardinality(df, target_col)
    findings += _detect_outliers(df, target_col)
    findings += _detect_imbalance(df, target_col)

    high = sum(1 for f in findings if f.severity == "high")
    medium = sum(1 for f in findings if f.severity == "medium")
    info = sum(1 for f in findings if f.severity == "info")

    score = max(0, 100 - (high * 20 + medium * 8 + info * 2))

    return QualityReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        findings=findings,
        high_count=high,
        medium_count=medium,
        info_count=info,
        quality_score=score,
    )