"""Evaluation protocols: repeated K-fold CV, nested CV, and single holdout.

All three report R^2, RMSE, and MAE as a `MetricSummary` (mean +/- std across
folds/repeats; a single holdout has n_folds=1 and std=0). The holdout
function mirrors the 90/10 train/validation split used throughout the
owner's original scripts (test_size=0.10, random_state=42; see
`ei_model.models.original_ann.prepare_data`, ported from `First.py`).

These utilities are model-agnostic: they operate on a `fit_predict` callback
so the same evaluation code can score the Keras baseline, a gradient-boosted
tree, or any other estimator.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# Fit on (X_train, y_train), predict on X_test, return predictions.
FitPredict = Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]

# Given an outer training fold, perform inner-CV model/hyperparameter
# selection, fit the chosen model, and return a predict(X_test) function.
ModelSelector = Callable[[pd.DataFrame, pd.Series], Callable[[pd.DataFrame], np.ndarray]]

DEFAULT_HOLDOUT_TEST_SIZE = 0.10
DEFAULT_RANDOM_STATE = 42


@dataclass
class MetricSummary:
    """Mean +/- std of R2, RMSE, and MAE across the scored folds/repeats."""

    r2_mean: float
    r2_std: float
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    n_folds: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _score_fold(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float]:
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def _summarize(r2s: list[float], rmses: list[float], maes: list[float]) -> MetricSummary:
    if not r2s:
        raise ValueError("Cannot summarize metrics from zero folds.")
    return MetricSummary(
        r2_mean=float(np.mean(r2s)),
        r2_std=float(np.std(r2s)),
        rmse_mean=float(np.mean(rmses)),
        rmse_std=float(np.std(rmses)),
        mae_mean=float(np.mean(maes)),
        mae_std=float(np.std(maes)),
        n_folds=len(r2s),
    )


def repeated_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    fit_predict: FitPredict,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> MetricSummary:
    """Repeat K-fold CV `n_repeats` times (a different shuffle each time) and pool all fold scores.

    `fit_predict(X_train, y_train, X_test)` fits on the training fold and
    returns predictions for the held-out fold.
    """
    r2s: list[float] = []
    rmses: list[float] = []
    maes: list[float] = []
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            y_pred = fit_predict(X_train, y_train, X_test)
            r2, rmse, mae = _score_fold(y_test, y_pred)
            r2s.append(r2)
            rmses.append(rmse)
            maes.append(mae)
    return _summarize(r2s, rmses, maes)


@dataclass
class FoldScore:
    """R2/RMSE/MAE for one scored fold, plus which repeat/fold it came from."""

    repeat: int
    fold: int
    r2: float
    rmse: float
    mae: float
    n_test: int


def repeated_kfold_cv_detailed(
    X: pd.DataFrame,
    y: pd.Series,
    fit_predict: FitPredict,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[MetricSummary, list[FoldScore]]:
    """Like `repeated_kfold_cv`, but also returns the per-fold scores.

    The per-fold scores are what paired comparisons (e.g. a model vs. the
    original ANN, fold-for-fold, since both use the same `random_state`/
    `n_splits`/`n_repeats` and so see identical splits) need: a fraction of
    folds improved and a Wilcoxon signed-rank test, both computed in
    `ei_model.experiments.stats`.
    """
    fold_scores: list[FoldScore] = []
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state + repeat)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            y_pred = fit_predict(X_train, y_train, X_test)
            r2, rmse, mae = _score_fold(y_test, y_pred)
            fold_scores.append(
                FoldScore(repeat=repeat, fold=fold, r2=r2, rmse=rmse, mae=mae, n_test=len(test_idx))
            )
    summary = _summarize(
        [s.r2 for s in fold_scores], [s.rmse for s in fold_scores], [s.mae for s in fold_scores]
    )
    return summary, fold_scores


def nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_selector: ModelSelector,
    outer_splits: int = 5,
    outer_repeats: int = 1,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> MetricSummary:
    """Nested CV: an outer loop for an unbiased estimate, around a caller-supplied inner selector.

    `model_selector(X_train, y_train)` is expected to run its own inner CV
    (e.g. sklearn's `GridSearchCV`/`RandomizedSearchCV`) on the outer training
    fold, fit the selected model, and return a `predict(X_test)` function.
    Only the outer folds are scored, so the reported metrics are not biased
    by the inner model/hyperparameter selection.
    """
    r2s: list[float] = []
    rmses: list[float] = []
    maes: list[float] = []
    for repeat in range(outer_repeats):
        kf = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state + repeat)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            predict = model_selector(X_train, y_train)
            y_pred = predict(X_test)
            r2, rmse, mae = _score_fold(y_test, y_pred)
            r2s.append(r2)
            rmses.append(rmse)
            maes.append(mae)
    return _summarize(r2s, rmses, maes)


def single_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    fit_predict: FitPredict,
    test_size: float = DEFAULT_HOLDOUT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> MetricSummary:
    """A single holdout split, mirroring the original 90/10 protocol.

    Defaults match `ei_model.models.original_ann` (test_size=0.10,
    random_state=42), which in turn mirrors the original `First.py`.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    y_pred = fit_predict(X_train, y_train, X_test)
    r2, rmse, mae = _score_fold(y_test, y_pred)
    return _summarize([r2], [rmse], [mae])
