"""Tests for ei_model.evaluation, run on the synthetic fixture with lightweight estimators."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ei_model.evaluation import (
    nested_cv,
    repeated_kfold_cv,
    repeated_kfold_cv_detailed,
    single_holdout,
)


def _ridge_fit_predict(X_train, y_train, X_test):
    pipe = make_pipeline(SimpleImputer(strategy="mean"), Ridge(alpha=1.0, random_state=0))
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test)


def _ridge_model_selector(X_train, y_train):
    pipe = make_pipeline(SimpleImputer(strategy="mean"), Ridge())
    grid = GridSearchCV(pipe, {"ridge__alpha": [0.1, 1.0, 10.0]}, cv=3)
    grid.fit(X_train, y_train)
    return lambda X_test: grid.predict(X_test)


def _assert_valid_summary(summary):
    for value in (
        summary.r2_mean,
        summary.r2_std,
        summary.rmse_mean,
        summary.rmse_std,
        summary.mae_mean,
        summary.mae_std,
    ):
        assert np.isfinite(value)
    assert summary.rmse_mean >= 0
    assert summary.mae_mean >= 0
    assert summary.rmse_std >= 0
    assert summary.mae_std >= 0


def test_repeated_kfold_cv_pools_all_folds(synthetic_X_y):
    X, y = synthetic_X_y
    summary = repeated_kfold_cv(X, y, _ridge_fit_predict, n_splits=5, n_repeats=3)

    assert summary.n_folds == 5 * 3
    _assert_valid_summary(summary)


def test_repeated_kfold_cv_is_deterministic(synthetic_X_y):
    X, y = synthetic_X_y
    a = repeated_kfold_cv(X, y, _ridge_fit_predict, n_splits=4, n_repeats=2)
    b = repeated_kfold_cv(X, y, _ridge_fit_predict, n_splits=4, n_repeats=2)

    assert a.as_dict() == b.as_dict()


def test_nested_cv_scores_only_outer_folds(synthetic_X_y):
    X, y = synthetic_X_y
    summary = nested_cv(X, y, _ridge_model_selector, outer_splits=3, outer_repeats=2)

    assert summary.n_folds == 3 * 2
    _assert_valid_summary(summary)


def test_single_holdout_matches_original_90_10_protocol(synthetic_X_y):
    X, y = synthetic_X_y
    summary = single_holdout(X, y, _ridge_fit_predict)

    assert summary.n_folds == 1
    assert summary.r2_std == 0.0
    assert summary.rmse_std == 0.0
    assert summary.mae_std == 0.0
    _assert_valid_summary(summary)


def test_single_holdout_respects_custom_test_size(synthetic_X_y):
    X, y = synthetic_X_y
    sizes = []

    def capturing_fit_predict(X_train, y_train, X_test):
        sizes.append(len(X_test))
        return _ridge_fit_predict(X_train, y_train, X_test)

    single_holdout(X, y, capturing_fit_predict, test_size=0.2, random_state=0)

    assert sizes == [round(len(X) * 0.2)]


def test_repeated_kfold_cv_detailed_matches_summary_and_returns_fold_scores(synthetic_X_y):
    X, y = synthetic_X_y
    summary, fold_scores = repeated_kfold_cv_detailed(
        X, y, _ridge_fit_predict, n_splits=4, n_repeats=2
    )

    assert len(fold_scores) == 4 * 2
    assert summary.n_folds == 4 * 2
    assert summary.r2_mean == np.mean([s.r2 for s in fold_scores])
    keys = {(s.repeat, s.fold) for s in fold_scores}
    assert keys == {(r, f) for r in range(2) for f in range(4)}


def test_repeated_kfold_cv_detailed_same_seed_matches_repeated_kfold_cv(synthetic_X_y):
    X, y = synthetic_X_y
    plain = repeated_kfold_cv(X, y, _ridge_fit_predict, n_splits=4, n_repeats=2)
    summary, _ = repeated_kfold_cv_detailed(X, y, _ridge_fit_predict, n_splits=4, n_repeats=2)

    assert plain.as_dict() == summary.as_dict()


def test_single_holdout_with_original_ann_tiny_epochs(synthetic_X_y):
    """End-to-end: evaluate the ported original ANN via the holdout protocol."""
    from ei_model.models import original_ann

    X, y = synthetic_X_y

    def tiny_ann_fit_predict(X_train, y_train, X_test):
        columns = X_train.columns
        imputer = SimpleImputer(strategy="mean")
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=columns)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=columns)

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=columns)

        model = original_ann.build_model(X_train_scaled.shape[1])
        model.fit(X_train_scaled, y_train, epochs=1, batch_size=16, verbose=0)
        return model.predict(X_test_scaled, verbose=0).flatten()

    summary = single_holdout(X, y, tiny_ann_fit_predict)

    assert summary.n_folds == 1
    assert np.isfinite(summary.r2_mean)
    assert summary.rmse_mean >= 0
