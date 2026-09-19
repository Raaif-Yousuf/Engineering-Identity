"""Runs one RunConfig across its protocol's datasets and models, timing everything."""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from ei_model.evaluation import FoldScore, repeated_kfold_cv_detailed

from . import protocols as pr
from .config import RunConfig
from .models import build_model_spec
from .stats import paired_r2_comparison


@dataclass
class ModelResult:
    dataset_key: str
    protocol: str
    variant: str
    label: str
    model_key: str
    model_label: str
    family: str
    n_rows: int
    n_features: int
    n_folds: int
    r2_mean: float | None
    r2_std: float | None
    rmse_mean: float | None
    rmse_std: float | None
    mae_mean: float | None
    mae_std: float | None
    fit_seconds: float
    available: bool
    unavailable_reason: str = ""
    vs_baseline_fraction_improved: float | None = None
    vs_baseline_wilcoxon_p: float | None = None
    vs_baseline_n_folds: int = 0


RunLogger = Callable[[str], None]


def _log(logger: RunLogger, message: str) -> None:
    logger(message)


def run_config(
    config: RunConfig,
    data_dir: str | None = None,
    logger: RunLogger = print,
) -> tuple[list[ModelResult], dict[str, dict[str, list[FoldScore]]]]:
    """Run every model in `config.models` on every dataset of `config.protocol`.

    Returns (results, fold_scores_by_dataset_and_model). The latter is used
    both for the baseline-vs-challenger Wilcoxon comparisons here and, by the
    caller, for permutation-importance figures on the best model.
    """
    resolved_data_dir = data_dir or config.data_dir
    datasets = pr.build_all(config.protocol, config.variants, data_dir=resolved_data_dir)

    results: list[ModelResult] = []
    fold_scores: dict[str, dict[str, list[FoldScore]]] = {}

    for dataset in datasets:
        _log(
            logger,
            f"[{config.name}] dataset={dataset.key} protocol={dataset.protocol} "
            f"variant={dataset.variant} n_rows={dataset.n_rows} n_features={dataset.n_features}",
        )
        fold_scores[dataset.key] = {}
        baseline_scores: list[FoldScore] | None = None

        model_keys = list(config.models)
        # Score the paired-comparison baseline first so every other model can
        # be compared against it even if config.models lists it last.
        if config.paired_baseline in model_keys:
            model_keys.remove(config.paired_baseline)
            model_keys.insert(0, config.paired_baseline)

        for model_key in model_keys:
            spec_kwargs: dict = {}
            if model_key in ("xgboost", "lightgbm"):
                spec_kwargs = {"n_iter": config.tuning.n_iter, "inner_cv": config.tuning.inner_cv}
            if model_key == "improved_ann":
                spec_kwargs = {"n_seeds": config.improved_ann_n_seeds}
            spec = build_model_spec(model_key, seed=config.random_state, **spec_kwargs)

            if not spec.available or spec.fit_predict is None:
                _log(logger, f"  {model_key}: unavailable ({spec.unavailable_reason})")
                results.append(
                    ModelResult(
                        dataset_key=dataset.key,
                        protocol=dataset.protocol,
                        variant=dataset.variant,
                        label=dataset.label,
                        model_key=spec.key,
                        model_label=spec.label,
                        family=spec.family,
                        n_rows=dataset.n_rows,
                        n_features=dataset.n_features,
                        n_folds=0,
                        r2_mean=None,
                        r2_std=None,
                        rmse_mean=None,
                        rmse_std=None,
                        mae_mean=None,
                        mae_std=None,
                        fit_seconds=0.0,
                        available=False,
                        unavailable_reason=spec.unavailable_reason or "",
                    )
                )
                continue

            cv = config.effective_cv(spec.family)
            start = time.perf_counter()
            summary, scores = repeated_kfold_cv_detailed(
                dataset.X,
                dataset.y,
                spec.fit_predict,
                n_splits=cv.n_splits,
                n_repeats=cv.n_repeats,
                random_state=cv.random_state,
            )
            elapsed = time.perf_counter() - start
            fold_scores[dataset.key][model_key] = scores

            if model_key == config.paired_baseline:
                baseline_scores = scores

            comparison = None
            if baseline_scores is not None and model_key != config.paired_baseline:
                comparison = paired_r2_comparison(
                    baseline_scores, scores, config.paired_baseline, model_key
                )

            _log(
                logger,
                f"  {model_key}: R2={summary.r2_mean:.3f}+/-{summary.r2_std:.3f} "
                f"({cv.n_splits}x{cv.n_repeats} folds, {elapsed:.1f}s)",
            )

            results.append(
                ModelResult(
                    dataset_key=dataset.key,
                    protocol=dataset.protocol,
                    variant=dataset.variant,
                    label=dataset.label,
                    model_key=spec.key,
                    model_label=spec.label,
                    family=spec.family,
                    n_rows=dataset.n_rows,
                    n_features=dataset.n_features,
                    n_folds=summary.n_folds,
                    r2_mean=summary.r2_mean,
                    r2_std=summary.r2_std,
                    rmse_mean=summary.rmse_mean,
                    rmse_std=summary.rmse_std,
                    mae_mean=summary.mae_mean,
                    mae_std=summary.mae_std,
                    fit_seconds=elapsed,
                    available=True,
                    vs_baseline_fraction_improved=(
                        comparison.fraction_improved if comparison else None
                    ),
                    vs_baseline_wilcoxon_p=(comparison.wilcoxon_p if comparison else None),
                    vs_baseline_n_folds=(comparison.n_matched_folds if comparison else 0),
                )
            )

    return results, fold_scores


def results_to_records(results: list[ModelResult]) -> list[dict]:
    return [asdict(r) for r in results]
