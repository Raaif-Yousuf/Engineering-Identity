import pytest

from ei_model.experiments.config import CvConfig, RunConfig
from ei_model.experiments.runner import run_config
from tests.fixtures.synthetic_experiments import write_synthetic_data_dir


@pytest.fixture
def data_dir(tmp_path):
    write_synthetic_data_dir(tmp_path, seed=5, n_rows=60, n_overlap=20, n_complete=20)
    return tmp_path


def test_run_config_p2_fast_models(data_dir):
    config = RunConfig(
        name="test_p2",
        protocol="P2",
        variants=["after"],
        models=["mean", "ridge", "random_forest"],
        cv=CvConfig(n_splits=3, n_repeats=1, random_state=0),
        paired_baseline="mean",
    )
    results, fold_scores = run_config(config, data_dir=str(data_dir), logger=lambda _msg: None)

    assert {r.model_key for r in results} == {"mean", "ridge", "random_forest"}
    assert all(r.available for r in results)
    assert all(r.n_folds == 3 for r in results)
    for r in results:
        assert r.r2_mean is not None
        assert r.rmse_mean >= 0
        assert r.mae_mean >= 0

    ridge_result = next(r for r in results if r.model_key == "ridge")
    assert ridge_result.vs_baseline_n_folds == 3
    assert ridge_result.vs_baseline_fraction_improved is not None

    assert set(fold_scores["p2_after"]) == {"mean", "ridge", "random_forest"}
    assert len(fold_scores["p2_after"]["mean"]) == 3


def test_run_config_marks_unavailable_model(data_dir):
    config = RunConfig(
        name="test_crowd",
        protocol="P2",
        variants=["after"],
        models=["mean", "crowd"],
        cv=CvConfig(n_splits=3, n_repeats=1),
        paired_baseline="mean",
    )
    results, _ = run_config(config, data_dir=str(data_dir), logger=lambda _msg: None)

    crowd_result = next(r for r in results if r.model_key == "crowd")
    assert crowd_result.available is False
    assert crowd_result.unavailable_reason
    assert crowd_result.n_folds == 0


def test_run_config_p1_multiple_variants(data_dir):
    config = RunConfig(
        name="test_p1",
        protocol="P1",
        variants=["after", "before"],
        models=["mean"],
        cv=CvConfig(n_splits=3, n_repeats=1),
        paired_baseline="mean",
    )
    results, _ = run_config(config, data_dir=str(data_dir), logger=lambda _msg: None)
    assert {r.dataset_key for r in results} == {"p1_after", "p1_before"}
