"""Train a simple regression model on Disney movie grosses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATA_PATH = Path("data/disney_movies_total_gross.csv")
METRICS_PATH = Path("artifacts/metrics.json")


@dataclass
class Dataset:
    features: pd.DataFrame
    target: pd.Series


def clean_currency(series: pd.Series) -> pd.Series:
    """Convert currency strings like `$120,000` to floats."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_dataset(path: Path) -> Dataset:
    """Load, clean, and engineer features for the modeling pipeline."""
    df = pd.read_csv(path)

    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year
    df["title_length"] = df["movie_title"].str.len()

    df["inflation_adjusted_gross"] = clean_currency(
        df["inflation_adjusted_gross"]
    )

    df = df.dropna(
        subset=["release_year", "inflation_adjusted_gross", "title_length"]
    )

    feature_cols = [
        "release_year",
        "title_length",
        "genre",
        "MPAA_rating",
    ]

    return Dataset(features=df[feature_cols], target=df["inflation_adjusted_gross"])


def build_pipeline() -> Pipeline:
    """Construct a preprocessing + model pipeline."""
    numeric_features = ["release_year", "title_length"]
    categorical_features = ["genre", "MPAA_rating"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute scalar regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }

    positive_mask = np.abs(y_true) > 1e-9
    if positive_mask.any():
        mape = np.mean(
            np.abs((y_true[positive_mask] - y_pred[positive_mask]) / y_true[positive_mask])
        )
        metrics["mape"] = float(mape)
    else:
        metrics["mape"] = float("nan")

    return metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    dataset = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = evaluate(y_test, predictions)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Evaluation metrics (test set):")
    for name, value in metrics.items():
        if math.isnan(value):
            display_value = "nan"
        elif name == "r2":
            display_value = f"{value:0.3f}"
        else:
            display_value = f"{value:0.2f}"
        print(f"  {name.upper():<4}: {display_value}")


if __name__ == "__main__":
    main()
