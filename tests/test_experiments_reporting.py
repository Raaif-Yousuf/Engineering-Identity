import pandas as pd
import pytest

from ei_model.experiments.config import CvConfig, RunConfig
from ei_model.experiments.reporting import plot_r2_bars, read_results_csv, write_results_csv
from ei_model.experiments.runner import run_config
from tests.fixtures.synthetic_experiments import write_synthetic_data_dir


@pytest.fixture
def sample_results(tmp_path):
    data_dir = tmp_path / "data"
    write_synthetic_data_dir(data_dir, seed=7, n_rows=60, n_overlap=20, n_complete=20)
    config = RunConfig(
        name="test_report",
        protocol="P2",
        variants=["after"],
        models=["mean", "ridge"],
        cv=CvConfig(n_splits=3, n_repeats=1),
        paired_baseline="mean",
    )
    results, _ = run_config(config, data_dir=str(data_dir), logger=lambda _msg: None)
    return results


def test_write_results_csv(tmp_path, sample_results):
    out = tmp_path / "results.csv"
    df = write_results_csv(sample_results, out)

    assert out.is_file()
    reloaded = pd.read_csv(out)
    assert len(reloaded) == len(sample_results) == len(df)
    assert {"model_key", "r2_mean", "r2_std", "protocol"}.issubset(reloaded.columns)


def test_plot_r2_bars_writes_a_png(tmp_path, sample_results):
    out = tmp_path / "figures" / "r2.png"
    plot_r2_bars(sample_results, "P2", out)

    assert out.is_file()
    assert out.stat().st_size > 0


def test_read_results_csv_missing_file_returns_empty_list(tmp_path):
    assert read_results_csv(tmp_path / "does_not_exist.csv") == []


def test_read_results_csv_round_trips_a_written_results_csv(tmp_path, sample_results):
    out = tmp_path / "results.csv"
    write_results_csv(sample_results, out)

    reloaded = read_results_csv(out)

    assert len(reloaded) == len(sample_results)
    by_key = {(r.dataset_key, r.model_key): r for r in reloaded}
    for original in sample_results:
        round_tripped = by_key[(original.dataset_key, original.model_key)]
        assert round_tripped.available == original.available
        assert round_tripped.unavailable_reason == original.unavailable_reason
        assert round_tripped.r2_mean == pytest.approx(original.r2_mean)
        assert round_tripped.n_folds == original.n_folds
