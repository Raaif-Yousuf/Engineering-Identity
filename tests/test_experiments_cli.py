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


def test_cli_run_resume_keeps_existing_rows_and_adds_a_new_model(tmp_path):
    # Regression test for --resume: simulates a config whose model list grew
    # (or whose previous attempt only got partway through) by first running
    # with one model list, then re-running --resume with an extra model
    # added -- the original row must survive untouched and only the new
    # model should actually get scored.
    data_dir = tmp_path / "data"
    write_synthetic_data_dir(data_dir, seed=13, n_rows=60, n_overlap=20, n_complete=20)

    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"

    def _write_config(models: str):
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            "name: cli_resume_test\n"
            "protocol: P2\n"
            "variants: [after]\n"
            f"models: [{models}]\n"
            f"cv: {{n_splits: 3, n_repeats: 1}}\n"
            "paired_baseline: mean\n"
            f"results_dir: {results_dir.as_posix()}\n"
            f"figures_dir: {figures_dir.as_posix()}\n",
            encoding="utf-8",
        )
        return config_path

    config_path = _write_config("mean, ridge")
    assert main(["run", "--config", str(config_path), "--data-dir", str(data_dir)]) == 0

    results_csv = results_dir / "cli_resume_test.csv"
    first_df = pd.read_csv(results_csv).set_index("model_key")

    config_path = _write_config("mean, ridge, random_forest")
    exit_code = main(
        ["run", "--config", str(config_path), "--data-dir", str(data_dir), "--resume"]
    )
    assert exit_code == 0

    second_df = pd.read_csv(results_csv).set_index("model_key")
    assert set(second_df.index) == {"mean", "ridge", "random_forest"}
    # mean/ridge carried over unchanged, not recomputed.
    for model_key in ("mean", "ridge"):
        assert second_df.loc[model_key, "r2_mean"] == first_df.loc[model_key, "r2_mean"]
        assert second_df.loc[model_key, "fit_seconds"] == first_df.loc[model_key, "fit_seconds"]
