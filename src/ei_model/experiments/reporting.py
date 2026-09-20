"""CSV/figure output: aggregates only, per the human-subjects-data rule (docs/data.md)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .protocols import ProtocolDataset
from .runner import ModelResult, results_to_records


def write_results_csv(results: list[ModelResult], path: str | Path) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results_to_records(results))
    df.to_csv(path, index=False)
    return df


_MODEL_RESULT_BOOL_FIELDS = ("available",)
_MODEL_RESULT_INT_FIELDS = ("n_rows", "n_features", "n_folds", "vs_baseline_n_folds")
_MODEL_RESULT_OPTIONAL_FLOAT_FIELDS = (
    "r2_mean",
    "r2_std",
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "vs_baseline_fraction_improved",
    "vs_baseline_wilcoxon_p",
)


def read_results_csv(path: str | Path) -> list[ModelResult]:
    """Reload a previously checkpointed `results/<name>.csv` as `ModelResult`s.

    Used to resume a run without recomputing (dataset, model) pairs already
    scored in an earlier attempt -- see `runner.run_config`'s `existing`
    parameter. Returns `[]` if `path` doesn't exist yet (nothing to resume
    from). `unavailable_reason`/`vs_baseline_*` may come back as NaN from a
    round-tripped CSV (pandas reads an empty cell as NaN, not ""), so those
    are normalized back to the same defaults `ModelResult` itself uses.
    """
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []

    records = df.to_dict(orient="records")
    results = []
    for rec in records:
        kwargs = dict(rec)
        for field in _MODEL_RESULT_BOOL_FIELDS:
            kwargs[field] = bool(kwargs.get(field))
        for field in _MODEL_RESULT_INT_FIELDS:
            value = kwargs.get(field)
            kwargs[field] = 0 if pd.isna(value) else int(value)
        for field in _MODEL_RESULT_OPTIONAL_FLOAT_FIELDS:
            value = kwargs.get(field)
            kwargs[field] = None if pd.isna(value) else float(value)
        fit_seconds = kwargs.get("fit_seconds")
        kwargs["fit_seconds"] = 0.0 if pd.isna(fit_seconds) else float(fit_seconds)
        reason = kwargs.get("unavailable_reason")
        kwargs["unavailable_reason"] = "" if pd.isna(reason) else str(reason)
        results.append(ModelResult(**kwargs))
    return results


def plot_r2_bars(
    results: list[ModelResult],
    protocol: str,
    out_path: str | Path,
    title: str | None = None,
) -> None:
    """Grouped bar chart of R2 +/- sd, one group per dataset, one bar per model."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results_to_records(results))
    df = df[(df["protocol"] == protocol) & (df["available"])]
    if df.empty:
        return

    datasets = sorted(df["dataset_key"].unique())
    models = sorted(df["model_key"].unique())
    x = np.arange(len(datasets))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(datasets)), 5))
    for i, model in enumerate(models):
        sub = df[df["model_key"] == model].set_index("dataset_key").reindex(datasets)
        ax.bar(
            x + i * width,
            sub["r2_mean"].to_numpy(dtype=float),
            width,
            yerr=sub["r2_std"].to_numpy(dtype=float),
            capsize=3,
            label=model,
        )

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("R2 (mean +/- sd across CV folds)")
    ax.set_title(title or f"{protocol}: R2 by dataset and model")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def permutation_importance_by_family(
    dataset: ProtocolDataset,
    fit_predict_factory,
    out_path: str | Path,
    n_repeats: int = 10,
    random_state: int = 42,
    title: str | None = None,
) -> pd.DataFrame:
    """Fit once on a train/test split, permutation-importance on the held-out
    split, aggregate |importance| by feature family, and plot + return it.

    Uses permutation importance (not SHAP) so it works uniformly across the
    tree/linear/boosted model families without an extra dependency; SHAP
    (already an optional extra, see pyproject.toml) is left for a follow-up
    if a specific model warrants the extra cost.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=0.2, random_state=random_state
    )
    estimator = fit_predict_factory(random_state)
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("estimator", estimator)])
    pipe.fit(X_train, y_train)

    result = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="r2",
        n_jobs=-1,
    )
    importances = pd.Series(result.importances_mean, index=dataset.X.columns)

    family_importance = {}
    for family, cols in dataset.feature_families.items():
        cols = [c for c in cols if c in importances.index]
        if cols:
            family_importance[family] = float(importances[cols].sum())
    family_series = pd.Series(family_importance).sort_values(ascending=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(family_series))))
    ax.barh(family_series.index, family_series.to_numpy())
    ax.set_xlabel("Sum of permutation importance (drop in R2 when shuffled)")
    ax.set_title(title or f"{dataset.key}: feature-family importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return family_series.rename("importance").reset_index().rename(columns={"index": "family"})
