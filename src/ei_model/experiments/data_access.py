"""Protocol-oriented helpers on top of `ei_model.data.load_variant`.

`load_variant(variant, completeness=...)` (added in the concurrent
`feat/concept-map-metrics` work, merged to main as `partial-complete-tables`)
already reads the two completeness levels this module's protocols need:
'partial' (full cohort, no concept-map columns) and 'complete' (only the
concept-map-digitized subset, all 44 concept-map columns populated). This
module only adds the feature-family bookkeeping (which columns belong to
which family, dropping a family, joining before/after on pid) that
`ei_model.data` doesn't itself need to provide.
"""

import pandas as pd

from ei_model.data import EI_TARGET_COL, PID_COL, load_feature_families, load_variant

LEAKAGE_FAMILY = "identity_survey_LEAKAGE_RISK"
SELF_ID_FAMILY = "self_identification"
CONCEPT_MAP_FAMILY = "concept_map"


def load_partial(variant: str, data_dir=None, env_var: str = "EI_DATA_DIR") -> pd.DataFrame:
    """The full-cohort table for `variant` (no concept-map columns)."""
    return load_variant(variant, completeness="partial", data_dir=data_dir, env_var=env_var)


def load_complete(variant: str, data_dir=None, env_var: str = "EI_DATA_DIR") -> pd.DataFrame:
    """The concept-map-complete subset for `variant` (all 44 concept-map columns present)."""
    return load_variant(variant, completeness="complete", data_dir=data_dir, env_var=env_var)


def family_columns(variant: str, family: str, present_columns) -> list[str]:
    """Columns of `family` (per feature_families.yaml) that are actually present."""
    families = load_feature_families(variant)
    wanted = families.get(family, [])
    present = set(present_columns)
    return [c for c in wanted if c in present]


def drop_families(table: pd.DataFrame, variant: str, families: list[str]) -> pd.DataFrame:
    """Drop all columns belonging to any of `families` from `table`."""
    drop_cols: list[str] = []
    for fam in families:
        drop_cols.extend(family_columns(variant, fam, table.columns))
    return table.drop(columns=drop_cols)


def feature_columns(table: pd.DataFrame) -> list[str]:
    """All non-id, non-target columns of a modeling table."""
    return [c for c in table.columns if c not in (PID_COL, EI_TARGET_COL)]


def join_on_pid(
    feature_table: pd.DataFrame,
    target_table: pd.DataFrame,
    target_rename: str | None = None,
) -> pd.DataFrame:
    """Inner-join `feature_table`'s features with `target_table`'s `ei` column, on `pid`.

    Used for P3 (before features -> after ei) and P3b (before features + before
    ei -> after ei). If `target_rename` is set, the joined-in target column is
    renamed to that instead of overwriting `ei` (used when `feature_table`
    already carries its own `ei`, e.g. P3b's before-EI persistence feature).
    """
    target_col = target_rename or EI_TARGET_COL
    right = target_table[[PID_COL, EI_TARGET_COL]].rename(columns={EI_TARGET_COL: target_col})
    if target_col in feature_table.columns and target_col != EI_TARGET_COL:
        raise ValueError(f"'{target_col}' already exists in feature_table.")
    return feature_table.merge(right, on=PID_COL, how="inner")
