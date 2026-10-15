"""Train and evaluate customer churn models, then save the best pipeline.

Usage:
    python -m src.train --data data/telco_churn.csv --out models/
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL, ENGINEERED, NUMERIC, TARGET, add_features, clean

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC + ENGINEERED),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pick the probability cut-off that maximises F1 on the validation data."""
    grid = np.linspace(0.2, 0.8, 61)
    scores = [f1_score(y_true, proba >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
    }


def main(data_path: str, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = add_features(clean(pd.read_csv(data_path)))
    X = df[NUMERIC + ENGINEERED + CATEGORICAL]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    leaderboard = {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        leaderboard[name] = round(float(scores.mean()), 4)
        print(f"{name:<22} CV ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(leaderboard, key=leaderboard.get)
    print(f"\nBest model: {best_name}")

    best = Pipeline([("prep", build_preprocessor()), ("model", candidate_models()[best_name])])
    # Out-of-fold probabilities give an unbiased basis for choosing the threshold
    oof_proba = cross_val_predict(
        best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = best_threshold(y_train.to_numpy(), oof_proba)
    best.fit(X_train, y_train)
    test_proba = best.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_proba, threshold)
    print(f"Hold-out metrics (threshold={threshold:.2f}): {test_metrics}")

    # Refit on all data for the deployed artefact
    best.fit(X, y)
    joblib.dump(best, out / "model.joblib")

    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_rows": len(df),
        "churn_rate": round(float(y.mean()), 4),
        "threshold": round(threshold, 3),
        "cv_roc_auc": leaderboard,
        "test_metrics": test_metrics,
        "features": {"numeric": NUMERIC, "engineered": ENGINEERED, "categorical": CATEGORICAL},
        # Typical retained customer, used as the reference point for explanations
        "baseline": {
            **{c: float(df.loc[y == 0, c].median()) for c in NUMERIC},
            **{c: str(df.loc[y == 0, c].mode()[0]) for c in CATEGORICAL},
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model and metadata to {out}/")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco_churn.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    main(args.data, args.out)
