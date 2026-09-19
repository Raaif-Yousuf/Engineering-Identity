"""Data loading utilities for the engineering identity model.

Real study data is human-subjects research data and never lives in this
repository. It is read at runtime from a directory configured via the
EI_DATA_DIR environment variable (or a small TOML config file), keeping
the actual spreadsheets, model artifacts, and predictions out of git.

The owner's original workbooks store two sheets per file: 'Input Vectors'
(one row per participant, one column per metric) and 'Targets' (participant
code -> composite engineering identity score), matching the shape read by
`load_and_prepare_data` in the original `newANN.py` / `First.py` scripts.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "ei_model" / "config.toml"
DEFAULT_INPUT_SHEET = "Input Vectors"
DEFAULT_TARGET_SHEET = "Targets"
DEFAULT_PARTICIPANT_COL = "Participant Code"
DEFAULT_TARGET_COL = "Target"

# The three canonical, de-identified modeling tables (see docs/data.md). Each
# is a Parquet file living in EI_DATA_DIR, keyed by a salted-hash 'pid'
# column (never the raw Participant Code) with the composite engineering
# identity target in a column named 'ei'.
MODELING_VARIANTS: tuple[str, ...] = ("after", "before", "diff")
PID_COL = "pid"
EI_TARGET_COL = "ei"

# Valid range of the composite EI target, by variant. 'after'/'before' are
# single-timepoint Whole scores; 'diff' is After-minus-Before and so has
# double the range.
VALID_TARGET_RANGES: dict[str, tuple[float, float]] = {
    "after": (-50.0, 50.0),
    "before": (-50.0, 50.0),
    "diff": (-100.0, 100.0),
}

# Committed, column-names-only feature-family map (see FeatureFamilies below
# for the synthetic/regex-based grouping used in tests). Real column names
# never appear anywhere else in this repo.
FEATURE_FAMILIES_PATH = Path(__file__).parent / "feature_families.yaml"


class UnknownVariantError(ValueError):
    """Raised when a variant name outside MODELING_VARIANTS is requested."""


def _check_variant(variant: str) -> None:
    if variant not in MODELING_VARIANTS:
        raise UnknownVariantError(
            f"Unknown variant {variant!r}; expected one of {MODELING_VARIANTS}."
        )

# Column-name patterns used to group features into families. Configurable:
# pass a custom mapping to FeatureFamilies/group_features_by_family to
# override any or all of these, e.g. for a differently-named export.
DEFAULT_FEATURE_FAMILY_PATTERNS: dict[str, str] = {
    "concept_map": r"^cm_",
    "verification": r"^(verify|class)_",
    "definition": r"^def_",
    "survey": r"^survey_",
}


class DataDirNotConfiguredError(RuntimeError):
    """Raised when no data directory can be resolved for the real dataset."""


def get_data_dir(env_var: str = "EI_DATA_DIR", config_path: Path | str | None = None) -> Path:
    """Resolve the directory holding the real (never-committed) modeling data.

    Resolution order:
      1. The `env_var` environment variable (default: EI_DATA_DIR).
      2. A `data_dir = "..."` entry in a TOML config file
         (default: ~/.config/ei_model/config.toml).

    Raises DataDirNotConfiguredError if neither is available.
    """
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser()

    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if path.is_file():
        import tomllib

        with path.open("rb") as fh:
            config = tomllib.load(fh)
        data_dir = config.get("data_dir")
        if data_dir:
            return Path(data_dir).expanduser()

    raise DataDirNotConfiguredError(
        f"No data directory configured. Set the {env_var} environment variable "
        f"or add 'data_dir = \"...\"' to {path}."
    )


def load_modeling_table(
    file_path: Path | str,
    input_sheet: str = DEFAULT_INPUT_SHEET,
    target_sheet: str = DEFAULT_TARGET_SHEET,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    target_col: str = DEFAULT_TARGET_COL,
) -> pd.DataFrame:
    """Load and merge the input-vectors and targets sheets of a modeling workbook.

    Mirrors the merge performed in the owner's original `load_and_prepare_data`
    (see `newANN.py`): an inner join of the 'Input Vectors' and 'Targets'
    sheets on the participant code.
    """
    inputs = pd.read_excel(file_path, sheet_name=input_sheet)
    targets = pd.read_excel(file_path, sheet_name=target_sheet)

    if participant_col not in inputs.columns:
        raise KeyError(f"Missing '{participant_col}' column in '{input_sheet}' sheet.")
    if participant_col not in targets.columns:
        raise KeyError(f"Missing '{participant_col}' column in '{target_sheet}' sheet.")

    return inputs.merge(targets[[participant_col, target_col]], on=participant_col)


def load_variant(
    variant: str,
    data_dir: Path | str | None = None,
    env_var: str = "EI_DATA_DIR",
) -> pd.DataFrame:
    """Load one of the canonical, de-identified modeling tables ('after'/'before'/'diff').

    Reads `<data_dir>/<variant>.parquet`, where `data_dir` defaults to
    `get_data_dir(env_var)`. Each table is keyed by a salted-hash 'pid'
    column (see docs/data.md) and carries the composite EI target in an
    'ei' column; the real Participant Code never appears in these files.
    """
    _check_variant(variant)
    resolved_dir = Path(data_dir).expanduser() if data_dir is not None else get_data_dir(env_var)
    path = resolved_dir / f"{variant}.parquet"
    table = pd.read_parquet(path)

    for required in (PID_COL, EI_TARGET_COL):
        if required not in table.columns:
            raise KeyError(f"'{variant}.parquet' is missing the '{required}' column.")
    return table


def load_feature_families(
    variant: str | None = None,
    path: Path | str | None = None,
) -> dict[str, list[str]] | dict[str, dict[str, list[str]]]:
    """Load the committed column-name-only feature-family map for one/all variants.

    With `variant`, returns that variant's `{family: [column, ...]}` mapping.
    Without it, returns the full `{variant: {family: [column, ...]}}` map.
    """
    import yaml

    families_path = Path(path) if path is not None else FEATURE_FAMILIES_PATH
    with families_path.open("r", encoding="utf-8") as fh:
        all_families = yaml.safe_load(fh)

    if variant is None:
        return all_families
    _check_variant(variant)
    return all_families[variant]


def split_features_target(
    table: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    id_cols: tuple[str, ...] = (DEFAULT_PARTICIPANT_COL,),
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a modeling table into a feature matrix X and target vector y."""
    drop_cols = [c for c in (*id_cols, target_col) if c in table.columns]
    X = table.drop(columns=drop_cols)
    y = table[target_col]
    return X, y


@dataclass(frozen=True)
class FeatureFamilies:
    """Configurable column-name patterns used to group features into families."""

    patterns: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_FAMILY_PATTERNS)
    )

    def group(self, columns: list[str]) -> dict[str, list[str]]:
        """Group `columns` by the first matching family pattern.

        Columns matching no pattern are placed under the 'other' key.
        """
        compiled = [(name, re.compile(pattern)) for name, pattern in self.patterns.items()]
        groups: dict[str, list[str]] = {name: [] for name in self.patterns}
        groups["other"] = []
        for col in columns:
            for name, rx in compiled:
                if rx.search(col):
                    groups[name].append(col)
                    break
            else:
                groups["other"].append(col)
        return groups


def group_features_by_family(
    columns: list[str], patterns: dict[str, str] | None = None
) -> dict[str, list[str]]:
    """Functional convenience wrapper around `FeatureFamilies.group`."""
    families = FeatureFamilies(patterns=dict(patterns)) if patterns else FeatureFamilies()
    return families.group(list(columns))
