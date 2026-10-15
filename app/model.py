"""Loads the trained pipeline and turns raw customer records into predictions."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import CATEGORICAL, ENGINEERED, NUMERIC, add_features

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
RAW_FEATURES = NUMERIC + CATEGORICAL


class ChurnModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.pipeline = joblib.load(model_dir / "model.joblib")
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.threshold: float = self.metadata["threshold"]
        self.baseline: dict = self.metadata["baseline"]

    def _proba(self, raw: pd.DataFrame) -> pd.Series:
        X = add_features(raw)[NUMERIC + ENGINEERED + CATEGORICAL]
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=raw.index)

    def risk_level(self, p: float) -> str:
        if p >= max(0.6, self.threshold + 0.2):
            return "High"
        if p >= self.threshold:
            return "Medium"
        return "Low"

    def predict(self, records: list[dict], n_drivers: int = 3) -> list[dict]:
        raw = pd.DataFrame(records)[RAW_FEATURES]
        base_proba = self._proba(raw)

        # Explanation by occlusion: swap each feature for a typical retained
        # customer's value and measure how much the churn probability drops.
        variants = []
        for i in range(len(raw)):
            for feat in RAW_FEATURES:
                row = raw.iloc[i].copy()
                row[feat] = self.baseline[feat]
                variants.append(row)
        variant_proba = self._proba(pd.DataFrame(variants).reset_index(drop=True)).to_numpy()

        results = []
        k = len(RAW_FEATURES)
        for i in range(len(raw)):
            p = float(base_proba.iloc[i])
            impacts = p - variant_proba[i * k : (i + 1) * k]
            ranked = sorted(zip(RAW_FEATURES, impacts, strict=True), key=lambda t: t[1], reverse=True)
            drivers = [
                {"feature": f, "value": records[i][f], "impact": round(float(d), 4)}
                for f, d in ranked[:n_drivers]
                if d > 0.005
            ]
            results.append(
                {
                    "churn_probability": round(p, 4),
                    "will_churn": p >= self.threshold,
                    "risk_level": self.risk_level(p),
                    "threshold": self.threshold,
                    "top_drivers": drivers,
                }
            )
        return results


@lru_cache(maxsize=1)
def get_model() -> ChurnModel:
    return ChurnModel()
