#!/usr/bin/env python3
"""Leakage-free optical-network fault detection and optional gain estimation.

This script is designed for lightpath-level optical telemetry datasets. It supports:
- automatic delimiter/header inference for .txt/.csv files,
- lightpath-level leakage-free train/test splitting,
- normal-only autoencoder anomaly detection,
- validation-only threshold calibration,
- raw-feature and optional physics-feature ablation,
- optional supervised gain regression when a gain target is supplied.

The code intentionally avoids hard-coded absolute paths and personal environment settings.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 42

GROUP_CANDIDATES = [
    "lightpath_id", "lightpath", "lp_id", "path_id", "route_id", "connection_id"
]
LABEL_CANDIDATES = [
    "failure_type", "failure", "label", "class", "condition", "fault", "target"
]
NORMAL_NAMES = {"no-failure", "no_failure", "normal", "healthy", "0", "none"}


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV/TXT data with delimiter inference and robust fallback."""
    attempts = [
        dict(sep=None, engine="python"),
        dict(sep=","),
        dict(sep=";"),
        dict(sep=r"\s+", engine="python"),
        dict(sep="\t"),
    ]
    last_error = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 2:
                return df
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
    raise ValueError(f"Could not parse {path}: {last_error}")


def find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for c in columns:
        low = str(c).strip().lower()
        if any(candidate in low for candidate in candidates):
            return c
    return None


def infer_group_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"Group column '{explicit}' not found.")
        return explicit
    found = find_column(df.columns, GROUP_CANDIDATES)
    if found:
        return found
    # Conservative fallback: create one group per row rather than risk leakage.
    df["__lightpath_id__"] = np.arange(len(df), dtype=np.int64)
    return "__lightpath_id__"


def infer_label_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"Label column '{explicit}' not found.")
        return explicit
    found = find_column(df.columns, LABEL_CANDIDATES)
    if found:
        return found
    raise KeyError(
        "Could not infer the failure-label column. Use --label-column COLUMN_NAME."
    )


def encode_failure_target(series: pd.Series, normal_label: str | None) -> np.ndarray:
    values = series.astype(str).str.strip().str.lower()
    if normal_label is not None:
        normal = str(normal_label).strip().lower()
        return (values != normal).astype(int).to_numpy()
    return (~values.isin(NORMAL_NAMES)).astype(int).to_numpy()


def numeric_features(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    x = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")
    x = x.select_dtypes(include=[np.number]).copy()
    if x.empty:
        raise ValueError("No numeric feature columns were found after exclusions.")
    x = x.replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True)
    x = x.fillna(med).fillna(0.0)
    # Remove constant columns, which do not help the autoencoder.
    nonconstant = x.nunique(dropna=False) > 1
    x = x.loc[:, nonconstant]
    if x.empty:
        raise ValueError("All numeric features are constant.")
    return x


def add_physics_features(x: pd.DataFrame) -> pd.DataFrame:
    """Add generic, scale-agnostic interaction features when relevant columns exist.

    These are optional and deliberately conservative. They do not assume a specific
    physical simulator and are intended for ablation, not as a claim of compensation.
    """
    out = x.copy()
    cols = {str(c).lower(): c for c in x.columns}

    def pick(tokens: list[str]) -> str | None:
        for low, original in cols.items():
            if all(t in low for t in tokens):
                return original
        return None

    length = pick(["length"]) or pick(["distance"])
    power = pick(["power"])
    osnr = pick(["osnr"])
    snr = pick(["snr"])
    dispersion = pick(["dispersion"])

    if length and power:
        out["physics_power_x_length"] = x[power] * x[length]
    if length and dispersion:
        out["physics_dispersion_x_length"] = x[dispersion] * x[length]
    if power:
        out["physics_power_squared"] = np.square(x[power])
    if osnr and snr and osnr != snr:
        out["physics_osnr_minus_snr"] = x[osnr] - x[snr]
    return out


def leakage_free_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def fit_autoencoder(
    x_train_normal: np.ndarray,
    random_state: int,
    hidden: tuple[int, ...] = (32, 8, 32),
) -> MLPRegressor:
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-5,
        batch_size=min(256, max(16, len(x_train_normal))),
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
        verbose=False,
    )
    model.fit(x_train_normal, x_train_normal)
    return model


def reconstruction_error(model: MLPRegressor, x: np.ndarray) -> np.ndarray:
    recon = model.predict(x)
    return np.mean(np.square(x - recon), axis=1)


def calibrate_threshold(
    errors: np.ndarray,
    y: np.ndarray,
    percentile: float,
) -> float:
    normal_errors = errors[y == 0]
    if normal_errors.size == 0:
        raise ValueError("Validation split contains no normal samples for threshold calibration.")
    return float(np.percentile(normal_errors, percentile))


def evaluate_binary(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores > threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "threshold": threshold,
    }
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = roc_auc_score(y, scores)
    else:
        metrics["roc_auc"] = float("nan")
    return {k: float(v) for k, v in metrics.items()}


def run_detector(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
    label_col: str,
    normal_label: str | None,
    percentile: float,
    physics_features: bool,
    random_state: int,
) -> tuple[dict[str, float], list[str]]:
    excluded = {group_col, label_col}
    x_train_df = numeric_features(train_df, excluded)
    feature_names = list(x_train_df.columns)

    # Align test columns to training columns.
    x_test_df = test_df.reindex(columns=feature_names).copy()
    x_test_df = x_test_df.replace([np.inf, -np.inf], np.nan)
    x_test_df = x_test_df.fillna(x_train_df.median()).fillna(0.0)

    if physics_features:
        x_train_df = add_physics_features(x_train_df)
        x_test_df = add_physics_features(x_test_df)
        x_test_df = x_test_df.reindex(columns=x_train_df.columns, fill_value=0.0)
        feature_names = list(x_train_df.columns)

    y_train = encode_failure_target(train_df[label_col], normal_label)
    y_test = encode_failure_target(test_df[label_col], normal_label)

    # Validation is selected only from training lightpaths.
    groups = train_df[group_col].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    fit_idx, val_idx = next(gss.split(x_train_df, y_train, groups))

    x_fit = x_train_df.iloc[fit_idx].to_numpy(dtype=float)
    y_fit = y_train[fit_idx]
    x_val = x_train_df.iloc[val_idx].to_numpy(dtype=float)
    y_val = y_train[val_idx]
    x_test = x_test_df.to_numpy(dtype=float)

    scaler = StandardScaler()
    x_fit_s = scaler.fit_transform(x_fit)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    normal_mask = y_fit == 0
    if normal_mask.sum() < 10:
        raise ValueError("Too few normal training samples to fit the autoencoder.")

    ae = fit_autoencoder(x_fit_s[normal_mask], random_state=random_state)
    val_scores = reconstruction_error(ae, x_val_s)
    threshold = calibrate_threshold(val_scores, y_val, percentile)
    test_scores = reconstruction_error(ae, x_test_s)
    metrics = evaluate_binary(y_test, test_scores, threshold)
    metrics.update({
        "n_train_rows": float(len(train_df)),
        "n_test_rows": float(len(test_df)),
        "n_train_lightpaths": float(train_df[group_col].nunique()),
        "n_test_lightpaths": float(test_df[group_col].nunique()),
        "n_features": float(len(feature_names)),
    })
    return metrics, feature_names


def run_gain_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
    label_col: str,
    gain_target: str,
    physics_features: bool,
    random_state: int,
) -> dict[str, float]:
    if gain_target not in train_df.columns or gain_target not in test_df.columns:
        raise KeyError(f"Gain target '{gain_target}' must exist in both train and test data.")

    excluded = {group_col, label_col, gain_target}
    x_train = numeric_features(train_df, excluded)
    x_test = test_df.reindex(columns=x_train.columns).copy()
    x_test = x_test.replace([np.inf, -np.inf], np.nan).fillna(x_train.median()).fillna(0.0)
    if physics_features:
        x_train = add_physics_features(x_train)
        x_test = add_physics_features(x_test).reindex(columns=x_train.columns, fill_value=0.0)

    y_train = pd.to_numeric(train_df[gain_target], errors="coerce")
    y_test = pd.to_numeric(test_df[gain_target], errors="coerce")
    train_mask = y_train.notna().to_numpy()
    test_mask = y_test.notna().to_numpy()

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        l2_regularization=1e-3,
        random_state=random_state,
    )
    model.fit(x_train.to_numpy()[train_mask], y_train.to_numpy()[train_mask])
    pred = model.predict(x_test.to_numpy()[test_mask])
    truth = y_test.to_numpy()[test_mask]
    rmse = math.sqrt(mean_squared_error(truth, pred))
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(truth, pred)),
        "n_test": float(len(truth)),
    }


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.train and args.test:
        return read_table(Path(args.train)), read_table(Path(args.test))

    if not args.data:
        raise ValueError("Provide --data FILE or both --train FILE and --test FILE.")

    df = read_table(Path(args.data))
    group_col = infer_group_column(df, args.group_column)
    train_df, test_df = leakage_free_split(
        df, group_col, args.test_size, args.seed
    )
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", help="Single dataset file; split by lightpath internally.")
    parser.add_argument("--train", help="Predefined training file.")
    parser.add_argument("--test", help="Predefined test file.")
    parser.add_argument("--group-column", help="Lightpath/group identifier column.")
    parser.add_argument("--label-column", help="Failure label column.")
    parser.add_argument("--normal-label", help="Exact label representing No-Failure/normal samples.")
    parser.add_argument("--gain-target", help="Optional numeric target for gain estimation.")
    parser.add_argument("--physics-features", action="store_true", help="Enable conservative physics-inspired feature interactions.")
    parser.add_argument("--threshold-percentile", type=float, default=99.0, help="Normal-validation reconstruction-error percentile used as anomaly threshold.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Lightpath-level test fraction when --data is used.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    if bool(args.train) ^ bool(args.test):
        parser.error("--train and --test must be supplied together.")

    train_df, test_df = load_data(args)

    # Infer metadata using the training table and apply same names to test table.
    group_col = infer_group_column(train_df, args.group_column)
    if group_col not in test_df.columns:
        raise KeyError(f"Group column '{group_col}' is missing from test data.")
    label_col = infer_label_column(train_df, args.label_column)
    if label_col not in test_df.columns:
        raise KeyError(f"Label column '{label_col}' is missing from test data.")

    overlap = set(train_df[group_col].unique()).intersection(set(test_df[group_col].unique()))
    if overlap:
        raise RuntimeError(
            f"Leakage detected: {len(overlap)} lightpath/group IDs occur in both train and test."
        )

    detector_metrics, features = run_detector(
        train_df=train_df,
        test_df=test_df,
        group_col=group_col,
        label_col=label_col,
        normal_label=args.normal_label,
        percentile=args.threshold_percentile,
        physics_features=args.physics_features,
        random_state=args.seed,
    )

    output = {
        "detector": detector_metrics,
        "metadata": {
            "group_column": group_col,
            "label_column": label_col,
            "features": features,
            "physics_features": bool(args.physics_features),
            "seed": int(args.seed),
        },
    }

    if args.gain_target:
        output["gain_regression"] = run_gain_regression(
            train_df,
            test_df,
            group_col,
            label_col,
            args.gain_target,
            args.physics_features,
            args.seed,
        )

    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
