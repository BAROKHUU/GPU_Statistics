"""Preprocessing and prediction models."""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NUMERIC_COLUMNS


def make_preprocessor(inputs: list[str]) -> ColumnTransformer:
    numeric = [x for x in inputs if x in NUMERIC_COLUMNS]
    category = [x for x in inputs if x not in numeric]
    num_steps = Pipeline([
        ("fill", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_steps = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("number", num_steps, numeric),
        ("category", cat_steps, category),
    ])


def make_model(name: str, inputs: list[str], log_target: bool):
    """Create a fresh pipeline; missing values are learned only from the training set."""
    estimator = (
        Ridge(alpha=10.0) if name == "Ridge" else
        RandomForestRegressor(n_estimators=120, min_samples_leaf=3,
                              random_state=42, n_jobs=-1)
    )
    pipeline = Pipeline([
        ("prepare", make_preprocessor(inputs)),
        ("regression", estimator),
    ])
    if log_target:
        # MAE remains in the original USD units because predict() automatically applies expm1.
        return TransformedTargetRegressor(
            regressor=pipeline, func=np.log1p, inverse_func=np.expm1,
            check_inverse=True,
        )
    return pipeline


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }
