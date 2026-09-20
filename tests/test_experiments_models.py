import numpy as np
import pandas as pd
import pytest

from ei_model.experiments import models as m


@pytest.fixture
def tiny_xy():
    rng = np.random.default_rng(0)
    n, p = 40, 6
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    X.iloc[0, 0] = np.nan  # exercise the imputer
    y = pd.Series(rng.normal(size=n))
    return X, y


def _check_predictions(preds, n_expected):
    arr = np.asarray(preds)
    assert arr.shape == (n_expected,)
    assert np.all(np.isfinite(arr))


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (m.mean_baseline_spec, {}),
        (m.ridge_spec, {}),
        (m.elastic_net_spec, {}),
        (m.random_forest_spec, {}),
        (m.stacking_spec, {}),
    ],
)
def test_fast_models_fit_predict(tiny_xy, builder, kwargs):
    X, y = tiny_xy
    spec = builder(**kwargs)
    X_train, X_test = X.iloc[:30], X.iloc[30:]
    y_train = y.iloc[:30]
    preds = spec.fit_predict(X_train, y_train, X_test)
    _check_predictions(preds, len(X_test))


def test_xgboost_spec_tuned_fit_predict(tiny_xy):
    X, y = tiny_xy
    spec = m.xgboost_spec(seed=0, n_iter=2, inner_cv=2)
    X_train, X_test = X.iloc[:30], X.iloc[30:]
    preds = spec.fit_predict(X_train, y.iloc[:30], X_test)
    _check_predictions(preds, len(X_test))


def test_lightgbm_spec_tuned_fit_predict(tiny_xy):
    X, y = tiny_xy
    spec = m.lightgbm_spec(seed=0, n_iter=2, inner_cv=2)
    X_train, X_test = X.iloc[:30], X.iloc[30:]
    preds = spec.fit_predict(X_train, y.iloc[:30], X_test)
    _check_predictions(preds, len(X_test))


def test_original_ann_spec_tiny_epochs(tiny_xy):
    X, y = tiny_xy
    spec = m.original_ann_spec(seed=0, epochs=2)
    X_train, X_test = X.iloc[:30], X.iloc[30:]
    preds = spec.fit_predict(X_train, y.iloc[:30], X_test)
    _check_predictions(preds, len(X_test))


def test_improved_ann_spec_tiny_epochs(tiny_xy):
    X, y = tiny_xy
    spec = m.improved_ann_spec(seed=0, n_seeds=2, epochs=2)
    X_train, X_test = X.iloc[:30], X.iloc[30:]
    preds = spec.fit_predict(X_train, y.iloc[:30], X_test)
    _check_predictions(preds, len(X_test))


def test_crowd_spec_available_and_fits(tiny_xy):
    # ei_model.models.crowd landed (feat/crowd-ann-model merged): crowd_spec
    # should now import CrowdANN and fit/predict like every other model spec.
    X, y = tiny_xy
    spec = m.crowd_spec(seed=0, replications=1)
    assert spec.available is True
    assert spec.fit_predict is not None
    assert spec.unavailable_reason is None

    X_train, X_test = X.iloc[:30], X.iloc[30:]
    preds = spec.fit_predict(X_train, y.iloc[:30], X_test)
    _check_predictions(preds, len(X_test))


def test_build_model_spec_unknown_key_raises():
    with pytest.raises(KeyError):
        m.build_model_spec("not_a_model")
