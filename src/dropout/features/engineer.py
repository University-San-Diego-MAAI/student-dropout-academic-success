"""Feature engineering and preprocessing."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ORDINAL_COLS = [
    "Application mode",
    "Course",
    "Previous qualification",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

OHE_COLS = ["Marital Status", "Nacionality"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the DataFrame."""
    df = df.copy()
    df["approval_rate_s1"] = (
        df["Curricular units 1st sem (approved)"]
        / (df["Curricular units 1st sem (enrolled)"] + 1e-6)
    )
    df["approval_rate_s2"] = (
        df["Curricular units 2nd sem (approved)"]
        / (df["Curricular units 2nd sem (enrolled)"] + 1e-6)
    )
    df["grade_delta"] = (
        df["Curricular units 2nd sem (grade)"]
        - df["Curricular units 1st sem (grade)"]
    )
    df["financial_risk"] = (
        df["Debtor"].astype(int)
        + (1 - df["Tuition fees up to date"].astype(int))
        + (1 - df["Scholarship holder"].astype(int))
    ) / 3.0
    return df


def add_derived_features_dict(features: dict) -> dict:
    """Add derived features to a single-record dict (for API use)."""
    features = dict(features)
    features["approval_rate_s1"] = (
        features["Curricular units 1st sem (approved)"]
        / (features["Curricular units 1st sem (enrolled)"] + 1e-6)
    )
    features["approval_rate_s2"] = (
        features["Curricular units 2nd sem (approved)"]
        / (features["Curricular units 2nd sem (enrolled)"] + 1e-6)
    )
    features["grade_delta"] = (
        features["Curricular units 2nd sem (grade)"]
        - features["Curricular units 1st sem (grade)"]
    )
    features["financial_risk"] = (
        int(features["Debtor"])
        + (1 - int(features["Tuition fees up to date"]))
        + (1 - int(features["Scholarship holder"]))
    ) / 3.0
    return features


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for the dataset."""
    return ColumnTransformer(
        transformers=[
            (
                "ord",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                ORDINAL_COLS,
            ),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                OHE_COLS,
            ),
            ("num", "passthrough", numeric_cols),
        ]
    )
