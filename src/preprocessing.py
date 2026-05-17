"""
Data cleaning pipeline for SBA loan records.

All transformations here are stateless (no fit required) — they apply the
same deterministic rules used during training to new data at scoring time.
"""

import numpy as np
import pandas as pd


_COLS_TO_DROP = ["City", "Bank", "BankState", "Zip", "BalanceGross", "MIS_Status", "index"]

_VALID_BINARY = {"Y": 1, "1": 1, 1: 1, "N": 0, "0": 0, 0: 0}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full deterministic cleaning pipeline to a raw SBA loans DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data in the same schema as the training CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for encoding and feature engineering.
    """
    df = df.copy()

    # Drop columns not used in modeling
    df = df.drop(columns=[c for c in _COLS_TO_DROP if c in df.columns])

    # NAICS: treat 0 as missing, then extract 2-digit sector code
    if "NAICS" in df.columns:
        df["NAICS"] = df["NAICS"].replace(0, np.nan).fillna(0).astype(int)
        df["NAICS_sector"] = (df["NAICS"] // 10000).astype(int)
        df = df.drop(columns=["NAICS"])

    # State: fill 12 genuine missing values
    if "State" in df.columns:
        df["State"] = df["State"].fillna("Unknown")

    # NewExist: 0.0 is invalid; replace with 1 (existing business)
    if "NewExist" in df.columns:
        df["NewExist"] = df["NewExist"].replace(0.0, 1.0).fillna(1.0).astype(int)

    # RevLineCr / LowDoc: messy string → binary 1/0; invalid → 0
    for col in ["RevLineCr", "LowDoc"]:
        if col in df.columns:
            df[col] = df[col].map(_VALID_BINARY).fillna(0).astype(int)

    # FranchiseCode → is_franchise binary flag
    # SBA convention: 1 = not a franchise, 0 = uncoded, >1 = franchise ID
    if "FranchiseCode" in df.columns:
        df["is_franchise"] = (df["FranchiseCode"] > 1).astype(int)
        df = df.drop(columns=["FranchiseCode"])

    # DisbursementGross: clip to avoid division-by-zero in ratios
    # (training dropped DisbursementGross==0 rows; scoring clips instead)
    if "DisbursementGross" in df.columns:
        df["DisbursementGross"] = df["DisbursementGross"].clip(lower=0.01)

    return df
