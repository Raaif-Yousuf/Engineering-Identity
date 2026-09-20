"""CLI: `python -m ei_model.experiments run --config configs/<name>.yaml`."""

import argparse
import sys
import time
from pathlib import Path

from . import protocols as pr
from .config import load_config
from .reporting import permutation_importance_by_family, plot_r2_bars, write_results_csv
from .runner import run_config


def _raw_estimator(model_key: str, seed: int):
    """A bare (unpipelined) estimator for permutation-importance fitting."""
    if model_key == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    if model_key == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=seed,
            objective="reg:squarederror",
            tree_method="hist",
        )
    if model_key == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
        )
    if model_key in ("ridge",):
        import numpy as np
        from sklearn.linear_model import RidgeCV

        return RidgeCV(alphas=np.logspace(-3, 3, 25))
    raise ValueError(f"No bare estimator available for importance_model={model_key!r}")


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.out:
        results_csv = Path(args.out)
    else:
        results_csv = Path(config.results_dir) / f"{config.name}.csv"

    def logger(message: str) -> None:
        print(message, flush=True)

    def checkpoint(results_so_far) -> None:
        # Writes the whole (small) results-so-far table after every single
        # model finishes, not just once at the end -- so a kill, crash, or
        # a model that hits `model_timeout_seconds` never loses the rows
        # already scored before it (see docs/experiments.md "Runtime
        # stall" for the run this was written in response to).
        write_results_csv(results_so_far, results_csv)

    start = time.perf_counter()
    logger(f"Running config '{config.name}' (protocol {config.protocol})")
    results, fold_scores = run_config(
        config, data_dir=args.data_dir, logger=logger, on_result=checkpoint
    )
    elapsed = time.perf_counter() - start
    logger(f"Done in {elapsed:.1f}s, {len(results)} (dataset, model) results.")

    write_results_csv(results, results_csv)
    logger(f"Wrote {results_csv}")

    figure_path = Path(config.figures_dir) / f"{config.name}_r2.png"
    plot_r2_bars(results, config.protocol, figure_path)
    logger(f"Wrote {figure_path}")

    if config.importance_model:
        resolved_data_dir = args.data_dir or config.data_dir
        datasets = {
            d.key: d
            for d in pr.build_all(config.protocol, config.variants, data_dir=resolved_data_dir)
        }
        available = [r for r in results if r.model_key == config.importance_model and r.available]
        if available:
            best = max(
                available, key=lambda r: (r.r2_mean if r.r2_mean is not None else float("-inf"))
            )
            dataset = datasets[best.dataset_key]
            importance_path = Path(config.figures_dir) / f"{config.name}_importance.png"
            table = permutation_importance_by_family(
                dataset,
                _raw_estimator_factory(config.importance_model),
                importance_path,
            )
            table.to_csv(Path(config.results_dir) / f"{config.name}_importance.csv", index=False)
            logger(f"Wrote {importance_path} (best dataset: {best.dataset_key})")

    return 0


def _raw_estimator_factory(model_key: str):
    def factory(seed: int):
        return _raw_estimator(model_key, seed)

    return factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ei_model.experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one protocol config end to end.")
    run_parser.add_argument("--config", required=True, help="Path to a configs/<name>.yaml file.")
    run_parser.add_argument("--data-dir", default=None, help="Override EI_DATA_DIR for this run.")
    run_parser.add_argument("--out", default=None, help="Override the results CSV output path.")
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
