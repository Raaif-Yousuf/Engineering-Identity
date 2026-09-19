import pytest

from ei_model.data import EI_TARGET_COL, PID_COL
from ei_model.experiments import data_access as da
from tests.fixtures.synthetic_experiments import write_synthetic_data_dir


@pytest.fixture
def data_dir(tmp_path):
    write_synthetic_data_dir(tmp_path, seed=1)
    return tmp_path


def test_load_partial_has_no_concept_map_columns(data_dir):
    table = da.load_partial("after", data_dir=data_dir)
    cm_cols = da.family_columns("after", da.CONCEPT_MAP_FAMILY, table.columns)
    assert cm_cols == []
    assert PID_COL in table.columns
    assert EI_TARGET_COL in table.columns


def test_load_complete_has_concept_map_columns_and_no_missing(data_dir):
    table = da.load_complete("after", data_dir=data_dir)
    cm_cols = da.family_columns("after", da.CONCEPT_MAP_FAMILY, table.columns)
    assert len(cm_cols) == 44
    assert not table[cm_cols].isna().any().any()


def test_load_complete_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        da.load_complete("after", data_dir=tmp_path)


def test_drop_families_removes_leakage_block(data_dir):
    table = da.load_partial("after", data_dir=data_dir)
    leaked_cols = da.family_columns("after", da.LEAKAGE_FAMILY, table.columns)
    assert leaked_cols  # sanity: the synthetic table does carry this family

    dropped = da.drop_families(table, "after", [da.LEAKAGE_FAMILY])
    assert not any(c in dropped.columns for c in leaked_cols)
    assert len(dropped) == len(table)


def test_load_partial_cleans_implausible_hours_values(tmp_path):
    import numpy as np

    write_synthetic_data_dir(tmp_path, seed=9)
    path = tmp_path / "after_partial.parquet"
    table = da.load_variant("after", completeness="partial", data_dir=tmp_path)
    table.loc[0, "Hours on Campus Per Week"] = 123456789.0
    table.to_parquet(path)

    cleaned = da.load_partial("after", data_dir=tmp_path)

    assert np.isnan(cleaned.loc[0, "Hours on Campus Per Week"])


def test_join_on_pid_is_inner_join(data_dir):
    before = da.load_partial("before", data_dir=data_dir)
    after = da.load_partial("after", data_dir=data_dir)
    joined = da.join_on_pid(before, after)

    expected_overlap = set(before[PID_COL]) & set(after[PID_COL])
    assert len(joined) == len(expected_overlap)
    assert set(joined[PID_COL]) == expected_overlap
