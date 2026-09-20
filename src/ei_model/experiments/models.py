"""Model registry: one `ModelSpec` per model family evaluated by the runner.

Every model exposes a `fit_predict(X_train, y_train, X_test) -> np.ndarray`
callable compatible with `ei_model.evaluation.repeated_kfold_cv`, so the same
CV driver scores every model. All preprocessing (imputation, scaling) and,
for XGBoost/LightGBM, hyperparameter search, happens *inside* that callable
so it is refit from scratch on every training fold -- nothing is fit on the
full dataset before splitting.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import RandomizedSearchCV

from . import preprocessing as pp

FitPredict = Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]

RIDGE_ALPHAS = np.logspace(-3, 3, 25)
ENET_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
RF_N_ESTIMATORS = 300
TUNING_N_ITER = 15
TUNING_INNER_CV = 3


@dataclass
class ModelSpec:
    key: str
    label: str
    family: str  # baseline | linear | tree | boosted | ann | ensemble | crowd
    fit_predict: FitPredict | None
    available: bool = True
    unavailable_reason: str | None = None
    extra: dict = field(default_factory=dict)


def _sk_fit_predict(pipeline) -> FitPredict:
    def fit_predict(X_train, y_train, X_test):
        pipeline.fit(X_train, y_train)
        return np.asarray(pipeline.predict(X_test))

    return fit_predict


def mean_baseline_spec(seed: int = 0) -> ModelSpec:
    pipe = pp.unscaled_pipeline(DummyRegressor(strategy="mean"))
    return ModelSpec("mean", "Mean predictor", "baseline", _sk_fit_predict(pipe))


def ridge_spec(seed: int = 0) -> ModelSpec:
    est = RidgeCV(alphas=RIDGE_ALPHAS)
    pipe = pp.standard_pipeline(est)
    return ModelSpec("ridge", "Ridge (CV alpha)", "linear", _sk_fit_predict(pipe))


def elastic_net_spec(seed: int = 0) -> ModelSpec:
    est = ElasticNetCV(l1_ratio=ENET_L1_RATIOS, cv=5, random_state=seed, max_iter=5000)
    pipe = pp.standard_pipeline(est)
    return ModelSpec(
        "elastic_net", "Elastic Net (CV alpha/l1_ratio)", "linear", _sk_fit_predict(pipe)
    )


def random_forest_spec(seed: int = 0) -> ModelSpec:
    est = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    pipe = pp.unscaled_pipeline(est)
    return ModelSpec("random_forest", "Random Forest", "tree", _sk_fit_predict(pipe))


def _xgboost_search(seed: int, n_iter: int, inner_cv: int) -> RandomizedSearchCV:
    from xgboost import XGBRegressor

    est = XGBRegressor(
        n_jobs=-1, random_state=seed, objective="reg:squarederror", tree_method="hist"
    )
    param_dist = {
        "estimator__n_estimators": randint(100, 400),
        "estimator__max_depth": randint(2, 7),
        "estimator__learning_rate": loguniform(0.01, 0.3),
        "estimator__subsample": [0.6, 0.8, 1.0],
        "estimator__colsample_bytree": [0.6, 0.8, 1.0],
        "estimator__reg_lambda": loguniform(0.1, 10.0),
    }
    pipe = pp.unscaled_pipeline(est)
    return RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=n_iter,
        cv=inner_cv,
        scoring="r2",
        random_state=seed,
        n_jobs=-1,
    )


def xgboost_spec(
    seed: int = 0, n_iter: int = TUNING_N_ITER, inner_cv: int = TUNING_INNER_CV
) -> ModelSpec:
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        return ModelSpec("xgboost", "XGBoost (tuned)", "boosted", None, False, str(exc))

    def fit_predict(X_train, y_train, X_test):
        search = _xgboost_search(seed, n_iter, inner_cv)
        search.fit(X_train, y_train)
        return np.asarray(search.predict(X_test))

    return ModelSpec("xgboost", "XGBoost (tuned)", "boosted", fit_predict)


def _lightgbm_search(seed: int, n_iter: int, inner_cv: int) -> RandomizedSearchCV:
    from lightgbm import LGBMRegressor

    est = LGBMRegressor(n_jobs=-1, random_state=seed, verbosity=-1)
    param_dist = {
        "estimator__n_estimators": randint(100, 400),
        "estimator__num_leaves": randint(7, 63),
        "estimator__learning_rate": loguniform(0.01, 0.3),
        "estimator__subsample": [0.6, 0.8, 1.0],
        "estimator__colsample_bytree": [0.6, 0.8, 1.0],
        "estimator__reg_lambda": loguniform(0.1, 10.0),
    }
    pipe = pp.unscaled_pipeline(est)
    return RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=n_iter,
        cv=inner_cv,
        scoring="r2",
        random_state=seed,
        n_jobs=-1,
    )


def lightgbm_spec(
    seed: int = 0, n_iter: int = TUNING_N_ITER, inner_cv: int = TUNING_INNER_CV
) -> ModelSpec:
    try:
        import lightgbm  # noqa: F401
    except ImportError as exc:
        return ModelSpec("lightgbm", "LightGBM (tuned)", "boosted", None, False, str(exc))

    def fit_predict(X_train, y_train, X_test):
        search = _lightgbm_search(seed, n_iter, inner_cv)
        search.fit(X_train, y_train)
        return np.asarray(search.predict(X_test))

    return ModelSpec("lightgbm", "LightGBM (tuned)", "boosted", fit_predict)


def original_ann_spec(seed: int = 42, epochs: int | None = None) -> ModelSpec:
    """`epochs=None` uses the original, unchanged 1000-epoch/early-stopping budget.

    A smaller `epochs` is only for fast tests -- real runs must leave it unset
    per the spec ("the original ANN, unchanged hyperparameters").
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from ei_model.models import original_ann as oann

    run_epochs = epochs if epochs is not None else oann.EPOCHS

    def fit_predict(X_train, y_train, X_test):
        oann.set_random_seed(seed)
        columns = X_train.columns
        imputer = SimpleImputer(strategy="mean")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train), columns=columns, index=X_train.index
        )
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test), columns=columns, index=X_test.index
        )
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train_imp), columns=columns, index=X_train.index
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test_imp), columns=columns, index=X_test.index
        )
        model = oann.build_model(X_train_s.shape[1])
        model.fit(
            X_train_s,
            y_train,
            validation_split=oann.TEST_SIZE,
            epochs=run_epochs,
            batch_size=oann.BATCH_SIZE,
            callbacks=oann.build_callbacks(),
            verbose=0,
        )
        return model.predict(X_test_s, verbose=0).flatten()

    return ModelSpec("original_ann", "Original ANN (unchanged)", "ann", fit_predict)


def improved_ann_spec(seed: int = 0, n_seeds: int = 3, epochs: int | None = None) -> ModelSpec:
    """`epochs=None` uses the improved ANN's own default budget; smaller values are for tests."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import QuantileTransformer

    from ei_model.models import improved_ann as iann

    def fit_predict(X_train, y_train, X_test):
        imputer = SimpleImputer(strategy="median")
        scaler = QuantileTransformer(
            n_quantiles=min(100, len(X_train)), output_distribution="normal", random_state=seed
        )
        X_train_s = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_s = scaler.transform(imputer.transform(X_test))
        X_train_s = pd.DataFrame(X_train_s, columns=X_train.columns, index=X_train.index)
        X_test_s = pd.DataFrame(X_test_s, columns=X_test.columns, index=X_test.index)
        fit_kwargs = {"epochs": epochs} if epochs is not None else {}
        result = iann.fit_predict_ensemble(
            X_train_s,
            y_train.reset_index(drop=True),
            X_test_s,
            n_seeds=n_seeds,
            base_seed=seed,
            **fit_kwargs,
        )
        return result.predictions

    return ModelSpec("improved_ann", f"Improved ANN ({n_seeds}-seed ensemble)", "ann", fit_predict)


def stacking_spec(seed: int = 0) -> ModelSpec:
    """A light stack of the fast base learners (ridge/RF/XGB), each with its own preprocessing."""
    from xgboost import XGBRegressor

    ridge_pipe = pp.standard_pipeline(RidgeCV(alphas=RIDGE_ALPHAS))
    rf_pipe = pp.unscaled_pipeline(
        RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    )
    xgb_pipe = pp.unscaled_pipeline(
        XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=seed,
            objective="reg:squarederror",
            tree_method="hist",
        )
    )
    stack = StackingRegressor(
        estimators=[("ridge", ridge_pipe), ("random_forest", rf_pipe), ("xgboost", xgb_pipe)],
        final_estimator=RidgeCV(alphas=RIDGE_ALPHAS),
        cv=5,
        n_jobs=-1,
    )

    def fit_predict(X_train, y_train, X_test):
        stack.fit(X_train, y_train)
        return np.asarray(stack.predict(X_test))

    return ModelSpec("stacking", "Stacking (ridge + RF + XGBoost)", "ensemble", fit_predict)


def crowd_spec(seed: int = 0, replications: int = 10) -> ModelSpec:
    """Wisdom-of-the-crowd model, wired against `ei_model.models.crowd.CrowdANN`.

    Uses `CrowdANN`'s sklearn-style API (`n_replications`, `random_state`,
    `n_jobs`). `replications` defaults to 10 rather than the MATLAB
    original's 100 to keep this CPU run sane -- pass 100 explicitly once
    timing allows. No external scaling: CrowdANN does its own internal
    MapMinMax normalization, mirroring the MATLAB original's `mapminmax`.

    Caution (measured, see `ei_model.models.crowd`'s module docstring): all
    189 architectures carry a parameter count dominated by the raw feature
    count, not neuron count. At the real EI feature widths (~140 columns),
    each replication of the full architecture sweep took minutes even with
    an analytic Jacobian; running this against the real dataset likely
    wants a low `replications` count and/or a reduced feature set, not the
    full 100-replication ensemble.
    """
    try:
        from ei_model.models.crowd import CrowdANN
    except ImportError as exc:
        return ModelSpec(
            "crowd",
            "Wisdom of the crowd",
            "crowd",
            None,
            available=False,
            unavailable_reason=f"ei_model.models.crowd not importable yet: {exc}",
        )

    def fit_predict(X_train, y_train, X_test):
        est = CrowdANN(n_replications=replications, random_state=seed, n_jobs=-1)
        pipe = pp.unscaled_pipeline(est)
        pipe.fit(X_train, y_train)
        return np.asarray(pipe.predict(X_test))

    return ModelSpec(
        "crowd", "Wisdom of the crowd", "crowd", fit_predict, extra={"replications": replications}
    )


BUILDERS: dict[str, Callable[..., ModelSpec]] = {
    "mean": mean_baseline_spec,
    "ridge": ridge_spec,
    "elastic_net": elastic_net_spec,
    "random_forest": random_forest_spec,
    "xgboost": xgboost_spec,
    "lightgbm": lightgbm_spec,
    "original_ann": original_ann_spec,
    "improved_ann": improved_ann_spec,
    "stacking": stacking_spec,
    "crowd": crowd_spec,
}


def build_model_spec(key: str, **kwargs) -> ModelSpec:
    if key not in BUILDERS:
        raise KeyError(f"Unknown model {key!r}; expected one of {sorted(BUILDERS)}")
    return BUILDERS[key](**kwargs)
