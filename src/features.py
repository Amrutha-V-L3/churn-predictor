"""Shared feature definitions so training and serving can never drift apart."""
from __future__ import annotations

import pandas as pd

TARGET = "Churn"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ENGINEERED = ["avg_monthly_spend", "num_services"]

_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix raw-data quirks in the Telco dataset."""
    df = df.copy()
    # TotalCharges is blank for brand-new customers (tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df.drop(columns=["customerID"], errors="ignore")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in _SERVICE_COLS) + (
        df["InternetService"] != "No"
    ).astype(int)
    return df
