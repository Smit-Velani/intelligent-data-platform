"""
preprocessor.py — Automated data cleaning, encoding, and imbalance-aware splitting.

Responsibilities:
  1. Load raw CSV into a DataFrame
  2. Detect + fill missing values (median for numeric, mode for categorical)
  3. Detect categorical columns -> encode (OneHot for low-cardinality, label for high)
  4. Detect numerical columns -> scale (StandardScaler)
  5. Detect problem type from the target column (classification / regression / clustering)
  6. Produce an IMBALANCE-AWARE, STRATIFIED train/test split for classification,
     so rare classes (e.g. 0.17% fraud) are represented proportionally in both sets
     instead of a plain random split which can starve the minority class from the
     test set entirely.

This module deliberately does NOT do model training or selection — see model_selector.py.
It also does NOT do drift comparisons — see drift.py. Single responsibility per module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


HIGH_CARDINALITY_THRESHOLD = 15  # above this many unique values, use label encoding
MISSING_DROP_THRESHOLD = 0.6     # drop a column if more than 60% of it is missing


@dataclass
class PreprocessResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    problem_type: str                # "classification" | "regression" | "clustering"
    class_balance: Optional[dict] = None   # {class_label: proportion} for classification
    scale_pos_weight: Optional[float] = None  # for XGBoost imbalance handling
    cv_splitter: object = None       # StratifiedKFold or KFold, ready to use
    dropped_columns: list = field(default_factory=list)
    encoders: dict = field(default_factory=dict)
    scaler: object = None


def load_csv(file_path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(file_path_or_buffer)
    df.columns = [c.strip() for c in df.columns]
    return df


def _drop_sparse_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    dropped = []
    for col in df.columns:
        missing_frac = df[col].isna().mean()
        if missing_frac > MISSING_DROP_THRESHOLD:
            dropped.append(col)
    return df.drop(columns=dropped), dropped


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            fill_val = mode.iloc[0] if not mode.empty else "missing"
            df[col] = df[col].fillna(fill_val)
    return df


def _encode_categoricals(df: pd.DataFrame, exclude: list) -> tuple[pd.DataFrame, dict]:
    encoders = {}
    cat_cols = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique <= HIGH_CARDINALITY_THRESHOLD:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoders[col] = {"type": "onehot", "categories": list(df[col].unique()) if col in df.columns else None}
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = {"type": "label", "encoder": le}
    return df, encoders


def detect_problem_type(df: pd.DataFrame, target_col: Optional[str]) -> str:
    """
    Decision rule:
      - No target column selected  -> clustering
      - Target has exactly 2 unique values -> binary classification
      - Target is numeric with > 15 unique values -> regression
      - Target is numeric/categorical with <= 15 unique values -> multiclass classification
    """
    if target_col is None or target_col not in df.columns:
        return "clustering"

    n_unique = df[target_col].nunique()
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])

    if n_unique == 2:
        return "classification"
    if is_numeric and n_unique > HIGH_CARDINALITY_THRESHOLD:
        return "regression"
    return "classification"  # multiclass


def compute_class_balance(y: pd.Series) -> dict:
    counts = y.value_counts(normalize=True)
    return {str(k): round(float(v), 6) for k, v in counts.items()}


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    XGBoost's scale_pos_weight = (# negative) / (# positive).
    This is the standard fix for imbalanced binary classification —
    it upweights the minority class's contribution to the loss function
    instead of treating every row equally, which is what causes models
    to just always predict the majority class on rare-event data.
    """
    counts = y.value_counts()
    if len(counts) != 2:
        return 1.0
    majority = counts.max()
    minority = counts.min()
    if minority == 0:
        return 1.0
    return float(majority / minority)


def preprocess(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
) -> PreprocessResult:
    df = df.copy()
    df, dropped = _drop_sparse_columns(df)
    df = _impute_missing(df)

    problem_type = detect_problem_type(df, target_col)

    if problem_type == "clustering":
        df, encoders = _encode_categoricals(df, exclude=[])
        scaler = StandardScaler()
        X = scaler.fit_transform(df.values)
        return PreprocessResult(
            X_train=X, X_test=np.empty((0, X.shape[1])),
            y_train=np.empty(0), y_test=np.empty(0),
            feature_names=list(df.columns),
            problem_type=problem_type,
            dropped_columns=dropped,
            encoders=encoders,
            scaler=scaler,
        )

    df, encoders = _encode_categoricals(df, exclude=[target_col])
    y = df[target_col]
    X_df = df.drop(columns=[target_col])
    feature_names = list(X_df.columns)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    class_balance = None
    scale_pos_weight = None
    cv_splitter = None

    if problem_type == "classification":
        class_balance = compute_class_balance(y)
        scale_pos_weight = compute_scale_pos_weight(y)

        # STRATIFIED split — preserves the minority class ratio in both
        # train and test sets. A plain random split on a 0.17% positive
        # class can (and often does) leave the test set with too few
        # fraud cases to evaluate precision/recall meaningfully.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y.values, test_size=test_size, stratify=y.values, random_state=random_state
        )
        # Stratified K-Fold for cross-validation during model selection —
        # every fold keeps the same class ratio as the full dataset.
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    else:  # regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y.values, test_size=test_size, random_state=random_state
        )
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return PreprocessResult(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        problem_type=problem_type,
        class_balance=class_balance,
        scale_pos_weight=scale_pos_weight,
        cv_splitter=cv_splitter,
        dropped_columns=dropped,
        encoders=encoders,
        scaler=scaler,
    )
