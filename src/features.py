"""
Feature engineering for SBA loan records.

All 12 engineered features are computed here. The WOE encoder must be
pre-fitted on training data and passed in for scoring-time transforms.
"""

import numpy as np
import pandas as pd


def engineer(df: pd.DataFrame, woe_encoder, fit: bool = False, y=None):
    """
    Apply all 12 feature engineering steps.

    Parameters
    ----------
    df : pd.DataFrame
        Encoded DataFrame (after OHE and target encoding).
    woe_encoder : category_encoders.WOEEncoder or None
        Fitted WOE encoder. Pass None only when fit=True.
    fit : bool
        If True, fit the WOE encoder on training data (requires y).
        If False, apply transform only (for validation/test/scoring).
    y : array-like, optional
        Target labels. Required when fit=True.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 12 additional engineered columns.
    woe_encoder : fitted WOEEncoder instance.
    """
    import category_encoders as ce

    df = df.copy()

    # 1. SBA guarantee coverage ratio
    df["sba_coverage_ratio"] = df["SBA_Appv"] / (df["GrAppv"] + 1e-6)

    # 2. Loan amount per employee (size-normalized exposure)
    df["loan_per_employee"] = df["GrAppv"] / (df["NoEmp"] + 1)

    # 3. Disbursement ratio (actual vs. approved)
    df["disbursement_ratio"] = df["DisbursementGross"] / (df["GrAppv"] + 1e-6)

    # 4. Job creation intent relative to current workforce
    df["job_creation_ratio"] = df["CreateJob"] / (df["NoEmp"] + 1)

    # 5–7. Log transforms for skewed monetary and count variables
    df["log_GrAppv"]            = np.log1p(df["GrAppv"])
    df["log_NoEmp"]             = np.log1p(df["NoEmp"])
    df["log_DisbursementGross"] = np.log1p(df["DisbursementGross"])

    # 8. Job retention rate
    df["retained_to_emp_ratio"] = df["RetainedJob"] / (df["NoEmp"] + 1)

    # 9. Interaction: low-documentation AND franchise (elevated risk combination)
    df["is_low_doc_franchise"] = df["LowDoc"] * df["is_franchise"]

    # 10. SBA guaranteed exposure per employee
    df["sba_appv_per_emp"] = df["SBA_Appv"] / (df["NoEmp"] + 1)

    # 11. Total employment impact
    df["total_jobs"] = df["CreateJob"] + df["RetainedJob"]

    # 12. WOE encoding of quantile-binned loan amount
    df["GrAppv_bin"] = pd.qcut(df["GrAppv"], q=5, labels=False, duplicates="drop")

    if fit:
        woe_encoder = ce.WOEEncoder(cols=["GrAppv_bin"])
        woe_encoder.fit(df[["GrAppv_bin"]], y)

    df["GrAppv_bin_woe"] = woe_encoder.transform(df[["GrAppv_bin"]])["GrAppv_bin"]
    df = df.drop(columns=["GrAppv_bin"])

    return df, woe_encoder


ENGINEERED_COLS = [
    "sba_coverage_ratio",
    "loan_per_employee",
    "disbursement_ratio",
    "job_creation_ratio",
    "log_GrAppv",
    "log_NoEmp",
    "log_DisbursementGross",
    "retained_to_emp_ratio",
    "is_low_doc_franchise",
    "sba_appv_per_emp",
    "total_jobs",
    "GrAppv_bin_woe",
]
