"""Run statistical summaries and regression evaluation for one target variable."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from .config import TASKS
from .regression import make_model, metrics
from .statistics import correlation_table, describe_target
from .visualization import exploratory_charts, model_chart


def analyze_task(df: pd.DataFrame, target: str, out: Path) -> list[str]:
    config = TASKS[target]
    inputs = config["inputs"]
    subset = df.loc[df[target].notna(), inputs + [target]].copy()
    if (subset[target] <= 0).any():
        raise ValueError(f"{target} contains non-positive values; please review the CSV")

    stats = describe_target(df, target)
    pd.DataFrame([stats]).to_csv(out / f"{target}_statistics.csv", index=False)
    correlation_table(subset, target).to_csv(
        out / f"{target}_correlation.csv", index=False
    )
    exploratory_charts(df, target, out)

    X, y = subset[inputs], subset[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name in ("Ridge", "RandomForest"):
        candidate = make_model(name, inputs, config["log_target"])
        scores = -cross_val_score(candidate, X_train, y_train,
                                  cv=cv, scoring="neg_mean_absolute_error")
        cv_rows.append({"model": name, "CV_MAE": scores.mean(),
                        "CV_MAE_std": scores.std(ddof=1)})
    cv_table = pd.DataFrame(cv_rows).sort_values("CV_MAE")
    cv_table.to_csv(out / f"{target}_model_comparison.csv", index=False)

    # Choose the model using cross-validation on the training set; the test set is used only once.
    best_name = str(cv_table.iloc[0]["model"])
    best = make_model(best_name, inputs, config["log_target"])
    best.fit(X_train, y_train)
    prediction = np.maximum(best.predict(X_test), 0)
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    baseline_prediction = baseline.predict(X_test)
    test_metrics = pd.DataFrame([
        {"model": "Baseline_median", **metrics(y_test, baseline_prediction)},
        {"model": best_name, **metrics(y_test, prediction)},
    ])
    test_metrics.to_csv(out / f"{target}_test_results.csv", index=False)
    pd.DataFrame({"actual": y_test, "predicted": prediction,
                  "error": y_test - prediction}).to_csv(
        out / f"{target}_test_predictions.csv", index=False
    )
    model_chart(y_test, prediction, target, out)

    result = test_metrics.iloc[1]
    return [
        f"{target}: {len(subset)} target values / {len(df)} GPUs; "
        f"train={len(X_train)}, test={len(X_test)}.",
        f"Summary: mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
        f"95% CI for mean=[{stats['ci95_low']:.2f}, {stats['ci95_high']:.2f}].",
        f"Selected {best_name} based on the average 5-fold CV MAE on the train set "
        f"({cv_table.iloc[0]['CV_MAE']:.2f}).",
        f"Test: MAE={result['MAE']:.2f}, RMSE={result['RMSE']:.2f}, "
        f"R2={result['R2']:.3f}; baseline MAE={test_metrics.iloc[0]['MAE']:.2f}.",
    ]
