"""Preprocessing helpers, meant to be fit *inside* each CV fold, never before."""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler


def standard_pipeline(estimator, impute_strategy: str = "median") -> Pipeline:
    """Median-impute then z-score, then `estimator`. The default for linear/tree/boosted models."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy=impute_strategy)),
            ("scale", StandardScaler()),
            ("estimator", estimator),
        ]
    )


def unscaled_pipeline(estimator, impute_strategy: str = "median") -> Pipeline:
    """Median-impute only, no scaling. Tree ensembles (RF/XGBoost/LightGBM) don't need scaling."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy=impute_strategy)),
            ("estimator", estimator),
        ]
    )


def quantile_pipeline(estimator, n_quantiles: int = 100, random_state: int = 0) -> Pipeline:
    """Median-impute then quantile-transform to a normal distribution, then `estimator`.

    Used by the improved ANN (spec: "quantile scaling"), which is more robust
    to the heavy-tailed/ordinal-Likert mix of columns than a plain z-score.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "scale",
                QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution="normal",
                    random_state=random_state,
                ),
            ),
            ("estimator", estimator),
        ]
    )
