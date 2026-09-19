import pytest

from ei_model.experiments import data_access as da
from ei_model.experiments import protocols as pr
from tests.fixtures.synthetic_experiments import write_synthetic_data_dir


@pytest.fixture
def data_dir(tmp_path):
    write_synthetic_data_dir(tmp_path, seed=3)
    return tmp_path


def test_p1_includes_identity_survey(data_dir):
    ds = pr.build_p1("after", data_dir=data_dir)
    leaked = da.family_columns("after", da.LEAKAGE_FAMILY, ds.X.columns)
    assert leaked
    assert ds.protocol == "P1"
    assert ds.label == pr.REPLICATION_LABEL


def test_p2_excludes_identity_survey(data_dir):
    ds = pr.build_p2("after", data_dir=data_dir)
    leaked = da.family_columns("after", da.LEAKAGE_FAMILY, ds.X.columns)
    assert leaked == []
    assert ds.n_rows == pr.build_p1("after", data_dir=data_dir).n_rows


def test_p3_joins_before_to_after_ei(data_dir):
    ds = pr.build_p3(persistence=False, data_dir=data_dir)
    leaked = da.family_columns("before", da.LEAKAGE_FAMILY, ds.X.columns)
    assert leaked == []
    assert "ei_before" not in ds.X.columns
    assert ds.n_rows > 0
    assert ds.protocol == "P3"


def test_p3b_adds_before_ei_feature(data_dir):
    ds = pr.build_p3(persistence=True, data_dir=data_dir)
    assert "ei_before" in ds.X.columns
    assert ds.protocol == "P3b"
    # Same cohort size as P3 (same join).
    p3 = pr.build_p3(persistence=False, data_dir=data_dir)
    assert ds.n_rows == p3.n_rows


def test_p4_p2_with_and_without_concept_map(data_dir):
    without_cm, with_cm = pr.build_p4_p2("after", data_dir=data_dir)
    assert without_cm.n_rows == with_cm.n_rows
    cm_without = da.family_columns("after", da.CONCEPT_MAP_FAMILY, without_cm.X.columns)
    cm_with = da.family_columns("after", da.CONCEPT_MAP_FAMILY, with_cm.X.columns)
    assert cm_without == []
    assert len(cm_with) == 44
    assert with_cm.n_features == without_cm.n_features + 44
    # Both restricted to the concept-map-complete cohort, not the full partial cohort.
    assert without_cm.n_rows < pr.build_p2("after", data_dir=data_dir).n_rows


def test_p4_p3_with_and_without_concept_map(data_dir):
    without_cm, with_cm = pr.build_p4_p3(data_dir=data_dir)
    assert without_cm.n_rows == with_cm.n_rows
    assert with_cm.n_features == without_cm.n_features + 44


def test_build_all_p1_returns_one_dataset_per_variant(data_dir):
    datasets = pr.build_all("P1", ["after", "before"], data_dir=data_dir)
    assert [d.variant for d in datasets] == ["after", "before"]


def test_build_all_p4_returns_eight_datasets(data_dir):
    datasets = pr.build_all("P4", ["after", "before", "diff"], data_dir=data_dir)
    assert len(datasets) == 3 * 2 + 2
