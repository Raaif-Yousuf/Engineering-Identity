"""YAML-driven run configuration for `python -m ei_model.experiments run`."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_PROTOCOLS = ("P1", "P2", "P3", "P3b", "P4")
DEFAULT_VARIANTS = ("after", "before", "diff")


@dataclass
class CvConfig:
    n_splits: int = 5
    n_repeats: int = 5
    random_state: int = 42


@dataclass
class TuningConfig:
    n_iter: int = 15
    inner_cv: int = 3


@dataclass
class RunConfig:
    name: str
    protocol: str
    variants: list[str] = field(default_factory=lambda: list(DEFAULT_VARIANTS))
    models: list[str] = field(
        default_factory=lambda: [
            "mean",
            "ridge",
            "elastic_net",
            "random_forest",
            "xgboost",
            "lightgbm",
            "stacking",
        ]
    )
    cv: CvConfig = field(default_factory=CvConfig)
    # Neural nets are much slower than the tree/linear models above; a
    # protocol can ask for a smaller repeat count for just the ann family to
    # keep wall time sane (see docs/experiments.md "Runtime deviations").
    ann_cv: CvConfig | None = None
    improved_ann_n_seeds: int = 3
    # The crowd reimplementation's cost is dominated by feature count, not
    # neuron count (every cascade layer re-consumes the raw input), so a
    # single replication of all 189 architectures already costs minutes at
    # real feature widths (see src/ei_model/models/crowd.py's docstring: a
    # measured 189x1-replication run on 900 rows took 341s). Default of 1
    # (not the MATLAB original's 100, nor this repo's earlier
    # crowd_spec default of 10) to keep a full 5-dataset-config protocol run
    # affordable -- see docs/experiments.md "Runtime deviations from the
    # spec" for the honest cost/coverage tradeoff this makes.
    crowd_replications: int = 1
    tuning: TuningConfig = field(default_factory=TuningConfig)
    random_state: int = 42
    paired_baseline: str = "original_ann"
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    data_dir: str | None = None
    notes: str = ""
    # If set, also fit this model once per dataset and plot its permutation
    # importance aggregated by feature family (see reporting.py).
    importance_model: str | None = None
    # Hard wall-clock cap (seconds) on one model's *entire* repeated-CV score
    # on one dataset, enforced in its own subprocess (see runner.py
    # `_run_model_scored`). A hang or pathological slowdown (observed
    # 2026-09-19: RandomizedSearchCV(n_jobs=-1) around an estimator that
    # itself used n_jobs=-1, a nested-parallelism oversubscription bug fixed
    # in models.py, but kept as a permanent safety net regardless of cause)
    # is recorded as `available=False, unavailable_reason="timeout: ..."`
    # instead of blocking every later dataset/model in the run.
    model_timeout_seconds: int = 1200

    def effective_cv(self, model_family: str) -> CvConfig:
        # crowd shares the ANN family's reduced schedule for the same
        # reason: it is far more expensive per fold than the tree/linear
        # models, so 25 folds (5x5) is not affordable at this budget.
        if model_family in ("ann", "crowd") and self.ann_cv is not None:
            return self.ann_cv
        return self.cv


def _dataclass_from_dict(cls, data: dict):
    if data is None:
        return cls()
    return cls(**data)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw.get("protocol") not in VALID_PROTOCOLS:
        raise ValueError(
            f"config 'protocol' must be one of {VALID_PROTOCOLS}, got {raw.get('protocol')!r}"
        )

    cv = _dataclass_from_dict(CvConfig, raw.pop("cv", None))
    ann_cv_raw = raw.pop("ann_cv", None)
    ann_cv = _dataclass_from_dict(CvConfig, ann_cv_raw) if ann_cv_raw is not None else None
    tuning = _dataclass_from_dict(TuningConfig, raw.pop("tuning", None))

    return RunConfig(cv=cv, ann_cv=ann_cv, tuning=tuning, **raw)
