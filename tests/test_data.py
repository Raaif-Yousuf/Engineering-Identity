"""Tests for ei_model.data, exercised on the synthetic fixture (no real data)."""

import inspect

import pandas as pd
import pytest

from ei_model.data import (
    DEFAULT_FEATURE_FAMILY_PATTERNS,
    DataDirNotConfiguredError,
    get_data_dir,
    group_features_by_family,
    load_modeling_table,
    split_features_target,
)
from tests.fixtures.synthetic import PARTICIPANT_COL, TARGET_COL


def test_get_data_dir_reads_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("EI_DATA_DIR", str(tmp_path))
    assert get_data_dir() == tmp_path


def test_get_data_dir_reads_config_file(monkeypatch, tmp_path):
    monkeypatch.delenv("EI_DATA_DIR", raising=False)
    config_path = tmp_path / "config.toml"
    data_dir = tmp_path / "real-data"
    config_path.write_text(f'data_dir = "{data_dir.as_posix()}"\n')

    assert get_data_dir(config_path=config_path) == data_dir


def test_get_data_dir_raises_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.delenv("EI_DATA_DIR", raising=False)
    with pytest.raises(DataDirNotConfiguredError):
        get_data_dir(config_path=tmp_path / "does-not-exist.toml")


def test_load_modeling_table_merges_sheets(tmp_path, synthetic_table):
    workbook = tmp_path / "modeling.xlsx"
    inputs = synthetic_table.drop(columns=[TARGET_COL])
    targets = synthetic_table[[PARTICIPANT_COL, TARGET_COL]]

    with pd.ExcelWriter(workbook) as writer:
        inputs.to_excel(writer, sheet_name="Input Vectors", index=False)
        targets.to_excel(writer, sheet_name="Targets", index=False)

    merged = load_modeling_table(workbook)

    assert len(merged) == len(synthetic_table)
    assert TARGET_COL in merged.columns
    assert PARTICIPANT_COL in merged.columns


def test_split_features_target_drops_id_and_target(synthetic_table):
    X, y = split_features_target(synthetic_table)

    assert TARGET_COL not in X.columns
    assert PARTICIPANT_COL not in X.columns
    assert len(X) == len(synthetic_table)
    assert (y == synthetic_table[TARGET_COL]).all()


def test_group_features_by_family_matches_synthetic_prefixes(synthetic_X_y):
    X, _y = synthetic_X_y
    groups = group_features_by_family(X.columns)

    assert set(groups) == set(DEFAULT_FEATURE_FAMILY_PATTERNS) | {"other"}
    assert groups["concept_map"] == [c for c in X.columns if c.startswith("cm_")]
    assert groups["verification"] == [c for c in X.columns if c.startswith("verify_")]
    assert groups["definition"] == [c for c in X.columns if c.startswith("def_")]
    assert groups["survey"] == [c for c in X.columns if c.startswith("survey_")]
    assert groups["other"] == []
    # Every feature column is accounted for exactly once.
    all_grouped = [col for cols in groups.values() for col in cols]
    assert sorted(all_grouped) == sorted(X.columns)


def test_group_features_by_family_accepts_custom_patterns(synthetic_X_y):
    X, _y = synthetic_X_y
    groups = group_features_by_family(X.columns, patterns={"concept_map": r"^cm_"})

    assert set(groups) == {"concept_map", "other"}
    assert groups["concept_map"] == [c for c in X.columns if c.startswith("cm_")]
    assert len(groups["other"]) == len(X.columns) - len(groups["concept_map"])


def test_default_env_var_is_ei_data_dir():
    # Documented contract: the default lookup key is EI_DATA_DIR.
    assert inspect.signature(get_data_dir).parameters["env_var"].default == "EI_DATA_DIR"
