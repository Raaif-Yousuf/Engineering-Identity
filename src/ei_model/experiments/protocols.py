"""Dataset construction for protocols P1-P4 (see docs/experiments.md).

- P1 "replication": all features (including the identity-survey block) for
  a given variant. Directly comparable to the original numbers.
- P2 "no proxy items": P1 minus `identity_survey_LEAKAGE_RISK` (single-item
  self-ratings that are one-to-one construct proxies for the composite EI
  score's 4 subscales -- see docs/experiments.md for why this is proxy
  correlation, not literal formula leakage).
- P3 "true prediction": before-semester features (minus the identity block)
  predicting after-semester `ei`, joined on `pid`.
- P3b: P3's features plus the before-semester `ei` as an extra feature
  (a persistence baseline), predicting after `ei`.
- P4 "complete subset": P2 and P3 restricted to the concept-map-complete
  cohort, with and without the concept-map columns.

Every builder returns one or more `ProtocolDataset`s carrying the raw
(unimputed, unscaled) feature matrix -- all preprocessing happens inside
the CV folds in `ei_model.experiments.preprocessing`.
"""

from dataclasses import dataclass, field

import pandas as pd

from ei_model.data import EI_TARGET_COL, MODELING_VARIANTS, PID_COL

from . import data_access as da

REPLICATION_LABEL = "includes identity-subscale proxy items"
NO_LEAKAGE_LABEL = "identity-subscale proxy items excluded"
TRUE_PREDICTION_LABEL = "before -> after, no identity block"
PERSISTENCE_LABEL = "before -> after, + before EI (persistence baseline)"
COMPLETE_WITH_CM_LABEL = "concept-map-complete cohort, with concept-map features"
COMPLETE_NO_CM_LABEL = "concept-map-complete cohort, without concept-map features"


@dataclass
class ProtocolDataset:
    key: str
    protocol: str
    variant: str
    label: str
    X: pd.DataFrame
    y: pd.Series
    feature_families: dict[str, list[str]]
    n_rows: int = field(init=False)
    n_features: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_rows = len(self.X)
        self.n_features = self.X.shape[1]


def _families_present(variant: str, columns: list[str]) -> dict[str, list[str]]:
    from ei_model.data import load_feature_families

    all_families = load_feature_families(variant)
    present = set(columns)
    return {fam: [c for c in cols if c in present] for fam, cols in all_families.items()}


def _split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = table.drop(columns=[PID_COL, EI_TARGET_COL])
    y = table[EI_TARGET_COL]
    return X, y


def build_p1(variant: str, data_dir=None) -> ProtocolDataset:
    """P1: all features (including identity survey), one variant."""
    table = da.load_partial(variant, data_dir=data_dir)
    X, y = _split(table)
    return ProtocolDataset(
        key=f"p1_{variant}",
        protocol="P1",
        variant=variant,
        label=REPLICATION_LABEL,
        X=X,
        y=y,
        feature_families=_families_present(variant, X.columns),
    )


def build_p2(variant: str, data_dir=None) -> ProtocolDataset:
    """P2: P1 minus the identity-survey leakage block."""
    table = da.load_partial(variant, data_dir=data_dir)
    table = da.drop_families(table, variant, [da.LEAKAGE_FAMILY])
    X, y = _split(table)
    return ProtocolDataset(
        key=f"p2_{variant}",
        protocol="P2",
        variant=variant,
        label=NO_LEAKAGE_LABEL,
        X=X,
        y=y,
        feature_families=_families_present(variant, X.columns),
    )


def build_p3(persistence: bool = False, data_dir=None) -> ProtocolDataset:
    """P3 (persistence=False) or P3b (persistence=True): before -> after ei."""
    before = da.load_partial("before", data_dir=data_dir)
    after = da.load_partial("after", data_dir=data_dir)
    before_no_identity = da.drop_families(before, "before", [da.LEAKAGE_FAMILY])

    if persistence:
        # Rename before's own 'ei' first so it survives the merge alongside
        # after's incoming 'ei' instead of colliding with it.
        renamed = before_no_identity.rename(columns={EI_TARGET_COL: "ei_before"})
        joined = da.join_on_pid(renamed, after)
        X = joined.drop(columns=[PID_COL, EI_TARGET_COL])
        y = joined[EI_TARGET_COL]
        key, label = "p3b_before_to_after", PERSISTENCE_LABEL
    else:
        # Drop before's own 'ei' first -- it isn't a P3 feature, and would
        # otherwise collide with (and get suffixed alongside) after's 'ei'.
        before_features_only = before_no_identity.drop(columns=[EI_TARGET_COL])
        joined = da.join_on_pid(before_features_only, after)
        X, y = _split(joined)
        key, label = "p3_before_to_after", TRUE_PREDICTION_LABEL

    families = _families_present("before", [c for c in X.columns if c != "ei_before"])
    if "ei_before" in X.columns:
        families["persistence_ei_before"] = ["ei_before"]
    return ProtocolDataset(
        key=key,
        protocol="P3b" if persistence else "P3",
        variant="before_to_after",
        label=label,
        X=X,
        y=y,
        feature_families=families,
    )


def build_p4_p2(variant: str, data_dir=None) -> tuple[ProtocolDataset, ProtocolDataset]:
    """P4/P2: no-leakage protocol restricted to the concept-map-complete cohort.

    Returns (without_concept_map, with_concept_map).
    """
    complete = da.load_complete(variant, data_dir=data_dir)
    no_identity = da.drop_families(complete, variant, [da.LEAKAGE_FAMILY])
    with_cm_X, y = _split(no_identity)
    without_cm = da.drop_families(no_identity, variant, [da.CONCEPT_MAP_FAMILY])
    without_cm_X, _ = _split(without_cm)

    fams_with = _families_present(variant, with_cm_X.columns)
    fams_without = _families_present(variant, without_cm_X.columns)
    return (
        ProtocolDataset(
            key=f"p4_p2_{variant}_no_cm",
            protocol="P4",
            variant=variant,
            label=f"P2 restricted, {COMPLETE_NO_CM_LABEL}",
            X=without_cm_X,
            y=y,
            feature_families=fams_without,
        ),
        ProtocolDataset(
            key=f"p4_p2_{variant}_with_cm",
            protocol="P4",
            variant=variant,
            label=f"P2 restricted, {COMPLETE_WITH_CM_LABEL}",
            X=with_cm_X,
            y=y,
            feature_families=fams_with,
        ),
    )


def build_p4_p3(data_dir=None) -> tuple[ProtocolDataset, ProtocolDataset]:
    """P4/P3: true-prediction protocol restricted to before's concept-map-complete cohort.

    Returns (without_concept_map, with_concept_map).
    """
    before_complete = da.load_complete("before", data_dir=data_dir)
    after = da.load_partial("after", data_dir=data_dir)
    before_no_identity = da.drop_families(before_complete, "before", [da.LEAKAGE_FAMILY])
    before_features_only = before_no_identity.drop(columns=[EI_TARGET_COL])
    joined_with_cm = da.join_on_pid(before_features_only, after)
    with_cm_X, y = _split(joined_with_cm)

    joined_without_cm = da.drop_families(joined_with_cm, "before", [da.CONCEPT_MAP_FAMILY])
    without_cm_X, _ = _split(joined_without_cm)

    fams_with = _families_present("before", with_cm_X.columns)
    fams_without = _families_present("before", without_cm_X.columns)
    return (
        ProtocolDataset(
            key="p4_p3_before_to_after_no_cm",
            protocol="P4",
            variant="before_to_after",
            label=f"P3 restricted, {COMPLETE_NO_CM_LABEL}",
            X=without_cm_X,
            y=y,
            feature_families=fams_without,
        ),
        ProtocolDataset(
            key="p4_p3_before_to_after_with_cm",
            protocol="P4",
            variant="before_to_after",
            label=f"P3 restricted, {COMPLETE_WITH_CM_LABEL}",
            X=with_cm_X,
            y=y,
            feature_families=fams_with,
        ),
    )


def build_all(protocol: str, variants: list[str] | None = None, data_dir=None):
    """Build every ProtocolDataset for a named protocol ('P1'/'P2'/'P3'/'P3b'/'P4')."""
    variants = variants or list(MODELING_VARIANTS)
    if protocol == "P1":
        return [build_p1(v, data_dir=data_dir) for v in variants]
    if protocol == "P2":
        return [build_p2(v, data_dir=data_dir) for v in variants]
    if protocol == "P3":
        return [build_p3(persistence=False, data_dir=data_dir)]
    if protocol == "P3b":
        return [build_p3(persistence=True, data_dir=data_dir)]
    if protocol == "P4":
        out = []
        for v in variants:
            out.extend(build_p4_p2(v, data_dir=data_dir))
        without_cm, with_cm = build_p4_p3(data_dir=data_dir)
        out.extend([without_cm, with_cm])
        return out
    raise ValueError(f"Unknown protocol {protocol!r}")
