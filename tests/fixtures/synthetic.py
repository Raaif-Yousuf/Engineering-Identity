"""Synthetic modeling-table generator used only by tests.

Produces a table with a shape similar to the real (never-committed)
Engineering Identity dataset: continuous concept-map complexity metrics,
binary classification-and-verification metrics, continuous definition
metrics, ordinal survey items, and a composite identity-score target
clipped to [-50, 50]. All values are random noise and carry no
relationship whatsoever to the real study.
"""

import numpy as np
import pandas as pd

PARTICIPANT_COL = "Participant Code"
TARGET_COL = "Target"


def generate_synthetic_modeling_table(
    n_rows: int = 100,
    seed: int = 42,
    missing_rate: float = 0.05,
) -> pd.DataFrame:
    """Generate a synthetic modeling table for tests.

    Column-name prefixes (cm_, verify_, def_, survey_) intentionally match
    ei_model.data.DEFAULT_FEATURE_FAMILY_PATTERNS so tests can exercise
    feature-family grouping against a realistic layout.
    """
    rng = np.random.default_rng(seed)

    columns: dict[str, object] = {
        PARTICIPANT_COL: [f"P{i:04d}" for i in range(n_rows)],
    }

    for i in range(1, 9):  # concept-map complexity metrics (continuous)
        columns[f"cm_{i:02d}"] = rng.normal(loc=10.0, scale=3.0, size=n_rows)

    for i in range(1, 7):  # binary classification-and-verification metrics
        columns[f"verify_{i:02d}"] = rng.integers(0, 2, size=n_rows).astype(float)

    for i in range(1, 6):  # definition metrics (continuous)
        columns[f"def_{i:02d}"] = rng.normal(loc=5.0, scale=2.0, size=n_rows)

    for i in range(1, 8):  # survey items (5-point Likert, treated as continuous)
        columns[f"survey_{i:02d}"] = rng.integers(1, 6, size=n_rows).astype(float)

    target = np.clip(rng.normal(loc=0.0, scale=15.0, size=n_rows), -50, 50)
    columns[TARGET_COL] = target

    table = pd.DataFrame(columns)

    # Inject missing values into feature columns only (never the ID or target).
    feature_cols = [c for c in table.columns if c not in (PARTICIPANT_COL, TARGET_COL)]
    mask = rng.random((n_rows, len(feature_cols))) < missing_rate
    feature_block = table[feature_cols].to_numpy(copy=True)
    feature_block[mask] = np.nan
    table[feature_cols] = feature_block

    return table
