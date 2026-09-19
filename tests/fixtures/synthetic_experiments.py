"""Synthetic partial/complete modeling tables for ei_model.experiments tests.

Mirrors the real EI_DATA_DIR layout (see ei_model.experiments.data_access):
`<variant>_partial.parquet` (full cohort, no concept-map columns) and
`<variant>_complete.parquet` (a subset, all concept-map columns populated).
Column *names* are taken from the real, committed `feature_families.yaml`
(safe -- it holds only column names, never data, see docs/data.md); all
values are random noise with no relationship to the real study.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ei_model.data import EI_TARGET_COL, PID_COL, load_feature_families

CONCEPT_MAP_FAMILY = "concept_map"


def _random_columns(columns: list[str], n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({col: rng.normal(size=n_rows) for col in columns})


def generate_variant_tables(
    variant: str,
    pids: list[str],
    n_complete: int,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (partial_df, complete_df) for one variant, real column names, random values.

    `pids` fixes the partial table's participant ids (so callers can overlap
    pids across variants for P3/P3b join tests); `complete` is a random
    subset of `n_complete` of those same pids.
    """
    rng = np.random.default_rng(seed)
    families = load_feature_families(variant)
    non_cm_families = {k: v for k, v in families.items() if k != CONCEPT_MAP_FAMILY}
    cm_columns = families.get(CONCEPT_MAP_FAMILY, [])
    non_cm_columns = [c for cols in non_cm_families.values() for c in cols]

    n_partial = len(pids)
    partial = _random_columns(non_cm_columns, n_partial, rng)
    partial.insert(0, PID_COL, list(pids))
    partial[EI_TARGET_COL] = rng.normal(loc=20.0, scale=15.0, size=n_partial)

    complete_pids = rng.choice(pids, size=min(n_complete, n_partial), replace=False).tolist()
    complete_base = partial[partial[PID_COL].isin(complete_pids)].reset_index(drop=True)
    cm_block = _random_columns(cm_columns, len(complete_base), rng)
    complete = pd.concat([complete_base.drop(columns=[EI_TARGET_COL]), cm_block], axis=1)
    complete[EI_TARGET_COL] = complete_base[EI_TARGET_COL].to_numpy()

    return partial, complete


def write_synthetic_data_dir(
    data_dir: Path, seed: int = 0, n_rows: int = 50, n_overlap: int = 20, n_complete: int = 20
) -> dict[str, tuple[int, int]]:
    """Write all six `<variant>_{partial,complete}.parquet` files under `data_dir`.

    `before` and `after` share the first `n_overlap` pids so P3/P3b joins
    have real rows to work with. Returns variant -> (partial_rows, complete_rows).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    shared_pids = [f"shared{i:05d}" for i in range(n_overlap)]
    shapes: dict[str, tuple[int, int]] = {}

    for i, variant in enumerate(("after", "before", "diff")):
        own_pids = shared_pids + [f"{variant[:2]}{i:05d}" for i in range(n_rows - n_overlap)]
        partial, complete = generate_variant_tables(
            variant, pids=own_pids, n_complete=n_complete, seed=seed + i
        )
        partial.to_parquet(data_dir / f"{variant}_partial.parquet")
        complete.to_parquet(data_dir / f"{variant}_complete.parquet")
        shapes[variant] = (len(partial), len(complete))

    return shapes
