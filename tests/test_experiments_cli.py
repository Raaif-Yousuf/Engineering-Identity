import pandas as pd

from ei_model.experiments.cli import main
from tests.fixtures.synthetic_experiments import write_synthetic_data_dir


def test_cli_run_end_to_end(tmp_path):
    data_dir = tmp_path / "data"
    write_synthetic_data_dir(data_dir, seed=11, n_rows=60, n_overlap=20, n_complete=20)

    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "name: cli_test\n"
        "protocol: P2\n"
        "variants: [after]\n"
        "models: [mean, ridge]\n"
        f"cv: {{n_splits: 3, n_repeats: 1}}\n"
        "paired_baseline: mean\n"
        "importance_model: ridge\n"
        f"results_dir: {results_dir.as_posix()}\n"
        f"figures_dir: {figures_dir.as_posix()}\n",
        encoding="utf-8",
    )

    exit_code = main(["run", "--config", str(config_path), "--data-dir", str(data_dir)])

    assert exit_code == 0
    results_csv = results_dir / "cli_test.csv"
    assert results_csv.is_file()
    df = pd.read_csv(results_csv)
    assert set(df["model_key"]) == {"mean", "ridge"}

    assert (figures_dir / "cli_test_r2.png").is_file()
    assert (figures_dir / "cli_test_importance.png").is_file()
    assert (results_dir / "cli_test_importance.csv").is_file()
