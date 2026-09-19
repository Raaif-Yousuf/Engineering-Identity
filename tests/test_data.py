"""Tests for ei_model.data, exercised on the synthetic fixture (no real data)."""

import inspect

import pandas as pd
import pytest

from ei_model.data import (
    COMPLETENESS_LEVELS,
    DEFAULT_FEATURE_FAMILY_PATTERNS,
    MODELING_VARIANTS,
    VALID_TARGET_RANGES,
    DataDirNotConfiguredError,
    UnknownCompletenessError,
    UnknownVariantError,
    get_data_dir,
    group_features_by_family,
    load_feature_families,
    load_modeling_table,
    load_variant,
    split_features_target,
)
from tests.fixtures.synthetic import PARTICIPANT_COL, TARGET_COL, generate_synthetic_modeling_table


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


def _write_synthetic_variant_parquet(
    tmp_path, variant: str, completeness: str = "partial", n_rows: int = 20
):
    """A tiny synthetic parquet shaped like a real modeling table (pid + ei)."""
    table = generate_synthetic_modeling_table(n_rows=n_rows, seed=1)
    table = table.rename(columns={PARTICIPANT_COL: "pid", TARGET_COL: "ei"})
    table.to_parquet(tmp_path / f"{variant}_{completeness}.parquet", index=False)
    return table


@pytest.mark.parametrize("variant", MODELING_VARIANTS)
@pytest.mark.parametrize("completeness", COMPLETENESS_LEVELS)
def test_load_variant_reads_pid_and_ei_columns(tmp_path, variant, completeness):
    expected = _write_synthetic_variant_parquet(tmp_path, variant, completeness)

    loaded = load_variant(variant, completeness=completeness, data_dir=tmp_path)

    assert "pid" in loaded.columns
    assert "ei" in loaded.columns
    assert len(loaded) == len(expected)


def test_load_variant_defaults_to_partial(tmp_path):
    expected = _write_synthetic_variant_parquet(tmp_path, "after", "partial")

    loaded = load_variant("after", data_dir=tmp_path)

    assert len(loaded) == len(expected)


def test_load_variant_uses_env_var_when_data_dir_omitted(tmp_path, monkeypatch):
    _write_synthetic_variant_parquet(tmp_path, "after", "partial")
    monkeypatch.setenv("EI_DATA_DIR", str(tmp_path))

    loaded = load_variant("after")

    assert "pid" in loaded.columns and "ei" in loaded.columns


def test_load_variant_rejects_unknown_variant(tmp_path):
    with pytest.raises(UnknownVariantError):
        load_variant("nonexistent", data_dir=tmp_path)


def test_load_variant_rejects_unknown_completeness(tmp_path):
    with pytest.raises(UnknownCompletenessError):
        load_variant("after", completeness="nonexistent", data_dir=tmp_path)


def test_load_variant_requires_pid_and_ei_columns(tmp_path):
    pd.DataFrame({"Participant Code": ["P1"], "Target": [1.0]}).to_parquet(
        tmp_path / "after_partial.parquet", index=False
    )
    with pytest.raises(KeyError):
        load_variant("after", data_dir=tmp_path)


def test_valid_target_ranges_cover_every_variant():
    assert set(VALID_TARGET_RANGES) == set(MODELING_VARIANTS)
    for lo, hi in VALID_TARGET_RANGES.values():
        assert lo < hi


def test_valid_target_ranges_diff_is_double_whole():
    lo_after, hi_after = VALID_TARGET_RANGES["after"]
    lo_diff, hi_diff = VALID_TARGET_RANGES["diff"]
    assert (hi_diff - lo_diff) == 2 * (hi_after - lo_after)


def test_load_feature_families_returns_committed_column_names_only():
    all_families = load_feature_families()

    assert set(all_families) == set(MODELING_VARIANTS)
    for variant in MODELING_VARIANTS:
        variant_families = load_feature_families(variant)
        assert variant_families == all_families[variant]
        assert "identity_survey_LEAKAGE_RISK" in variant_families
        for columns in variant_families.values():
            assert all(isinstance(c, str) for c in columns)


def test_load_feature_families_rejects_unknown_variant():
    with pytest.raises(UnknownVariantError):
        load_feature_families("nonexistent")
