"""
Production scoring function for SBA loan default prediction.

This module replicates the exact preprocessing, encoding, and feature
engineering pipeline from training. It is the authoritative implementation;
scoring_notebook.ipynb calls this or mirrors its logic.
"""

import os

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import clean
from src.features import engineer


def _load_artifacts(model_type: str) -> dict:
    paths = {
        "sklearn": "artifacts/sklearn/sklearn_artifacts.pkl",
        "lgbm":    "artifacts/lgbm/lgbm_artifacts.pkl",
        "h2o":     "artifacts/h2o/h2o_artifacts.pkl",
    }
    if model_type not in paths:
        raise ValueError(f"model_type must be 'sklearn', 'lgbm', or 'h2o', got '{model_type}'")
    path = paths[model_type]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def _apply_encoding(df: pd.DataFrame, arts: dict) -> pd.DataFrame:
    """Apply OHE, target encoding, and feature engineering using saved artifacts."""
    ohe_cols    = arts["ohe_cols"]
    te_cols     = arts["te_cols"]
    te_encoder  = arts["te_encoder"]
    woe_encoder = arts["woe_encoder"]

    # One-hot encoding
    df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols)

    # Target encoding (transform only — do NOT fit)
    for col in te_cols:
        if col in df.columns:
            df[f"{col}_trg"] = te_encoder.transform(df[te_cols])[col]
    df = df.drop(columns=[c for c in te_cols if c in df.columns])

    # Feature engineering (transform only)
    df, _ = engineer(df, woe_encoder=woe_encoder, fit=False)

    return df


def project_1_scoring(data: pd.DataFrame, model_type: str = "lgbm") -> pd.DataFrame:
    """
    Score new SBA loan records using a trained model.

    Parameters
    ----------
    data : pd.DataFrame
        Input data in the same schema as the training CSV, without MIS_Status.
    model_type : str
        'lgbm' (default), 'sklearn', or 'h2o'.
        H2O requires an active h2o.init() connection.

    Returns
    -------
    pd.DataFrame
        Columns: index, label, probability_0, probability_1.
        One row per input record. No records are dropped.
    """
    df = clean(data)
    arts = _load_artifacts(model_type)
    df = _apply_encoding(df, arts)

    threshold = arts["optimal_threshold"]

    if model_type in ("sklearn", "lgbm"):
        feature_col_key = "feature_columns" if model_type == "lgbm" else "train_columns"
        train_cols = arts[feature_col_key]
        model      = arts["model"]

        df = df.reindex(columns=train_cols, fill_value=0)

        if model_type == "sklearn":
            scaler     = arts["scaler"]
            sc_cols    = arts["scaler_cols"]
            sc_present = [c for c in sc_cols if c in df.columns]
            df[sc_present] = scaler.transform(df[sc_present])

        probs  = model.predict_proba(df)
        prob_0 = probs[:, 0]
        prob_1 = probs[:, 1]

    else:  # h2o
        import h2o

        feature_cols = arts["feature_cols"]
        df = df.reindex(columns=feature_cols, fill_value=0)

        # Pick the most recently saved H2O model file (deterministic)
        h2o_dir = "artifacts/h2o"
        model_files = sorted(
            [
                os.path.join(h2o_dir, f)
                for f in os.listdir(h2o_dir)
                if not f.endswith((".pkl", ".gitkeep"))
            ],
            key=os.path.getmtime,
            reverse=True,
        )
        if not model_files:
            raise FileNotFoundError("No H2O model file found in artifacts/h2o/")

        h2o_model = h2o.load_model(model_files[0])
        hf        = h2o.H2OFrame(df)
        preds     = h2o_model.predict(hf).as_data_frame()
        prob_0    = preds["p0"].values
        prob_1    = preds["p1"].values

    labels = (prob_1 >= threshold).astype(int)

    return pd.DataFrame({
        "index":         data.index,
        "label":         labels,
        "probability_0": prob_0,
        "probability_1": prob_1,
    })
