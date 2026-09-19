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

    def effective_cv(self, model_family: str) -> CvConfig:
        if model_family == "ann" and self.ann_cv is not None:
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
