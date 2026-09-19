"""Deterministic, domain-knowledge data cleaning, applied once at load time.

This is not "fit" from the sample (no means/variances/thresholds are
estimated from the data), only a hard physical bound, so it's safe to apply
identically to every row before any train/test split -- unlike imputation or
scaling, which must be fit inside each CV fold (see preprocessing.py).

Finding (see docs/experiments.md "Data quality: implausible hours values"):
several rows across the self-reported "hours per week" columns hold values
that are not physically possible (a week has 168 hours), up to and including
a literal `123456789` placeholder in one after-wave row. These are treated
as missing (NaN) so the ordinary per-fold imputer handles them like any
other missing value, rather than letting a single row's typo/placeholder
dominate a linear model's fitted coefficients or a neural net's gradients.
"""

import numpy as np
import pandas as pd

MAX_HOURS_PER_WEEK = 168  # 24 * 7 -- a hard physical bound, not a data-fit threshold.

# Every column name, across after/before/diff, that represents a
# self-reported "hours per week" quantity. 'before' spells these
# differently (see feature_families.yaml); its own values are all within
# bounds as of this writing, but the columns are included for robustness.
HOURS_PER_WEEK_COLUMNS: tuple[str, ...] = (
    "Hours on Campus Per Week",
    "Engineering Activity on Campus Per Week",
    "Non-Engineering Activity on Campus Per Week",
    "Total Per Week",
    "Engineering Hours",
    "0n-Engineering Hours",
)


def clean_implausible_hours(
    table: pd.DataFrame, bound: float = MAX_HOURS_PER_WEEK
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Null out `HOURS_PER_WEEK_COLUMNS` values with |value| > `bound`.

    Returns (cleaned_table, {column: n_rows_nulled}) so callers/tests can
    confirm what was touched.
    """
    cleaned = table.copy()
    report: dict[str, int] = {}
    for col in HOURS_PER_WEEK_COLUMNS:
        if col not in cleaned.columns:
            continue
        mask = cleaned[col].abs() > bound
        n_bad = int(mask.sum())
        if n_bad:
            cleaned.loc[mask, col] = np.nan
            report[col] = n_bad
    return cleaned, report
