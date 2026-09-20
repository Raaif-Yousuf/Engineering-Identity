"""Runs one RunConfig across its protocol's datasets and models, timing everything."""

import multiprocessing as mp
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass

from ei_model.evaluation import FoldScore, repeated_kfold_cv_detailed

from . import protocols as pr
from .config import CvConfig, RunConfig
from .models import build_model_spec
from .stats import paired_r2_comparison


@dataclass
class ModelResult:
    dataset_key: str
    protocol: str
    variant: str
    label: str
    model_key: str
    model_label: str
    family: str
    n_rows: int
    n_features: int
    n_folds: int
    r2_mean: float | None
    r2_std: float | None
    rmse_mean: float | None
    rmse_std: float | None
    mae_mean: float | None
    mae_std: float | None
    fit_seconds: float
    available: bool
    unavailable_reason: str = ""
    vs_baseline_fraction_improved: float | None = None
    vs_baseline_wilcoxon_p: float | None = None
    vs_baseline_n_folds: int = 0


RunLogger = Callable[[str], None]
ResultsCallback = Callable[[list["ModelResult"]], None]


def _log(logger: RunLogger, message: str) -> None:
    logger(message)


def _model_worker(conn, X, y, model_key, seed, spec_kwargs, n_splits, n_repeats, random_state):
    """Runs in its own (spawned) subprocess -- see `_run_model_scored`.

    Rebuilds the model spec here (fit_predict closures aren't picklable
    across a process boundary anyway) and sends the outcome back over
    `conn`. Any exception is caught and reported rather than left to kill
    the subprocess silently, so the parent never mistakes a real crash for
    a hang (or vice versa).
    """
    try:
        spec = build_model_spec(model_key, seed=seed, **spec_kwargs)
        if not spec.available or spec.fit_predict is None:
            conn.send({"status": "unavailable", "reason": spec.unavailable_reason or ""})
            return
        summary, scores = repeated_kfold_cv_detailed(
            X,
            y,
            spec.fit_predict,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        conn.send({"status": "ok", "summary": summary, "scores": scores})
    except Exception as exc:  # noqa: BLE001 - report every crash, never let the pipe go silent
        try:
            conn.send({"status": "error", "reason": f"{exc!r}\n{traceback.format_exc()}"})
        except Exception:  # noqa: BLE001 - the pipe itself may already be gone (see below)
            # Observed 2026-09-20: a joblib/loky resource-tracker race deep in
            # a long tuning run (see docs/experiments.md "Resource-tracker
            # crash") can raise a *second*, unrelated exception (e.g.
            # BrokenPipeError) right here, while we're already reporting the
            # first one. There is nothing more this worker can do to tell the
            # parent what happened -- exiting with a non-zero code is enough,
            # since the parent treats "child died without sending a result"
            # as its own error outcome (see `_run_model_scored_once`). Do not
            # let a failed send crash this except-block with an unhandled
            # exception of its own.
            pass
    finally:
        conn.close()


def _kill_process_tree(proc) -> None:
    """`Process.terminate()` only signals the direct child, not any
    grandchildren it spawned (e.g. loky workers from a nested
    RandomizedSearchCV) -- on Windows those would otherwise leak as orphans
    after a timeout. `taskkill /T` kills the whole tree; POSIX processes are
    killed directly (this harness has no grandchild-spawning models outside
    the sklearn n_jobs=-1 path, whose loky workers exit when their parent
    dies on POSIX)."""
    pid = proc.pid
    if pid is not None:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False
            )
        else:
            proc.terminate()
    proc.join(5)
    if proc.is_alive():
        proc.kill()
        proc.join()


def _run_model_scored_once(
    dataset: "pr.ProtocolDataset",
    model_key: str,
    seed: int,
    spec_kwargs: dict,
    cv: CvConfig,
    timeout_seconds: int,
    logger: RunLogger,
) -> dict:
    """Score one model on one dataset in its own subprocess, with a hard timeout.

    Returns a dict with "status" in {"ok", "unavailable", "error", "timeout"}.
    A hang inside `fit_predict` (observed 2026-09-19: a joblib/loky
    nested-n_jobs oversubscription that made LightGBM's tuning search
    non-progressing for over an hour on Windows -- see
    docs/experiments.md) is recorded as "timeout" and the run continues to
    the next model, instead of blocking every later dataset/model.

    This is a single attempt -- see `_run_model_scored` for the retry-once
    wrapper that calls this and handles a crashed attempt.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_model_worker,
        args=(
            child_conn,
            dataset.X,
            dataset.y,
            model_key,
            seed,
            spec_kwargs,
            cv.n_splits,
            cv.n_repeats,
            cv.random_state,
        ),
    )
    proc.start()
    child_conn.close()  # only the child writes
    proc.join(timeout_seconds)
    if proc.is_alive():
        _log(
            logger,
            f"  {model_key}: TIMED OUT after {timeout_seconds}s "
            f"(pid={proc.pid}) -- killing worker tree, recording as unavailable",
        )
        _kill_process_tree(proc)
        return {"status": "timeout", "reason": f"timed out after {timeout_seconds}s"}
    if parent_conn.poll():
        try:
            return parent_conn.recv()
        except Exception as exc:  # noqa: BLE001 - see docstring below for why this must be broad
            # Observed 2026-09-20 (see docs/experiments.md "Resource-tracker
            # crash"): a joblib/loky resource-tracker race deep in a long
            # tuning run can raise a TerminatedWorkerError inside the
            # worker's own except-block, at the exact moment it's trying to
            # `conn.send()` that error back -- so the pipe can close mid
            # write and leave a truncated frame. `parent_conn.recv()` then
            # raises something other than `EOFError` (an `OSError` /
            # `ConnectionResetError` / a pickle error, depending on exactly
            # where the write was cut off). The original code here only
            # caught `EOFError`, so this second, unrelated exception
            # propagated straight out of `_run_model_scored`, out of
            # `run_config`'s loop, and killed the *entire* run (this is
            # exactly what took down `replication.yaml`'s P1 run ~103
            # minutes in, mid `p1_diff`/`lightgbm`, per
            # `results/logs/replication.log`) -- not just this one model.
            # Any receive failure here means the same thing either way: the
            # worker died without giving us a usable result, so treat it
            # like any other crash and let the caller retry/continue.
            return {
                "status": "error",
                "reason": f"failed to read worker result ({exc!r}) -- worker likely crashed",
            }
    return {
        "status": "error",
        "reason": (
            f"worker exited (code={proc.exitcode}) without sending a result -- likely crashed"
        ),
    }


def _run_model_scored(
    dataset: "pr.ProtocolDataset",
    model_key: str,
    seed: int,
    spec_kwargs: dict,
    cv: CvConfig,
    timeout_seconds: int,
    logger: RunLogger,
    max_retries: int = 1,
) -> dict:
    """Score one model on one dataset, retrying once (by default) on a crash.

    Only a "error" outcome (the worker process crashed or its result
    couldn't be read -- see `_run_model_scored_once`) is retried. "timeout"
    and "unavailable" are not: a timeout means the model is genuinely too
    slow for the budget (retrying just burns another `timeout_seconds` for
    no benefit), and "unavailable" means the model spec itself refused to
    build (e.g. a missing optional dependency), which a retry can't fix
    either. Each retry gets a brand-new subprocess (and therefore a brand
    new resource tracker), which is exactly the reset that gives a
    transient loky/resource-tracker race (docs/experiments.md
    "Resource-tracker crash") a real chance of not recurring.
    """
    outcome = _run_model_scored_once(
        dataset, model_key, seed, spec_kwargs, cv, timeout_seconds, logger
    )
    attempt = 1
    while outcome["status"] == "error" and attempt <= max_retries:
        _log(
            logger,
            f"  {model_key}: crashed on attempt {attempt} ({outcome.get('reason', '')[:200]}) "
            f"-- retrying in a fresh subprocess",
        )
        outcome = _run_model_scored_once(
            dataset, model_key, seed, spec_kwargs, cv, timeout_seconds, logger
        )
        attempt += 1
    return outcome


def run_config(
    config: RunConfig,
    data_dir: str | None = None,
    logger: RunLogger = print,
    on_result: ResultsCallback | None = None,
    existing_results: list[ModelResult] | None = None,
) -> tuple[list[ModelResult], dict[str, dict[str, list[FoldScore]]]]:
    """Run every model in `config.models` on every dataset of `config.protocol`.

    Returns (results, fold_scores_by_dataset_and_model). The latter is used
    both for the baseline-vs-challenger Wilcoxon comparisons here and, by the
    caller, for permutation-importance figures on the best model.

    If `on_result` is given, it is called with the full `results` list so
    far after *every* model finishes (available, unavailable, errored, or
    timed out) -- so the caller can checkpoint to disk after each one and a
    kill/crash/timeout never loses already-completed rows (see
    docs/experiments.md "Runtime stall" for why this matters: a run that
    hangs on model N used to lose every model before it too, since the CSV
    was previously written only once at the very end).

    If `existing_results` is given (typically loaded from a previous,
    partial run's checkpointed CSV via `reporting.read_results_csv`), any
    (dataset_key, model_key) pair already present in it is *not*
    recomputed -- its row is carried over as-is and reported again via
    `on_result` so the checkpoint file stays complete. This is what lets a
    crashed or interrupted run resume instead of starting from scratch (see
    docs/experiments.md "Resuming a partial run"). One consequence: if the
    protocol's `paired_baseline` model was itself one of the carried-over
    (not recomputed) rows for a dataset, this run has no fold-level scores
    for it, so `vs_baseline_*` is left at its default (None/0) for every
    *newly* scored model on that dataset -- the aggregate R^2/RMSE/MAE
    numbers are unaffected, only the paired Wilcoxon comparison is skipped
    for that dataset in that resumed run.
    """
    resolved_data_dir = data_dir or config.data_dir
    datasets = pr.build_all(config.protocol, config.variants, data_dir=resolved_data_dir)

    already_done = {(r.dataset_key, r.model_key): r for r in (existing_results or [])}

    results: list[ModelResult] = list(already_done.values())
    fold_scores: dict[str, dict[str, list[FoldScore]]] = {}

    for dataset in datasets:
        _log(
            logger,
            f"[{config.name}] dataset={dataset.key} protocol={dataset.protocol} "
            f"variant={dataset.variant} n_rows={dataset.n_rows} n_features={dataset.n_features}",
        )
        fold_scores[dataset.key] = {}
        baseline_scores: list[FoldScore] | None = None

        model_keys = list(config.models)
        # Score the paired-comparison baseline first so every other model can
        # be compared against it even if config.models lists it last.
        if config.paired_baseline in model_keys:
            model_keys.remove(config.paired_baseline)
            model_keys.insert(0, config.paired_baseline)

        for model_key in model_keys:
            if (dataset.key, model_key) in already_done:
                _log(
                    logger,
                    f"  {model_key}: already recorded in a previous run -- skipping (resume)",
                )
                continue

            spec_kwargs: dict = {}
            if model_key in ("xgboost", "lightgbm"):
                spec_kwargs = {"n_iter": config.tuning.n_iter, "inner_cv": config.tuning.inner_cv}
            if model_key == "improved_ann":
                spec_kwargs = {"n_seeds": config.improved_ann_n_seeds}
            if model_key == "crowd":
                spec_kwargs = {"replications": config.crowd_replications}
            spec = build_model_spec(model_key, seed=config.random_state, **spec_kwargs)

            if not spec.available or spec.fit_predict is None:
                _log(logger, f"  {model_key}: unavailable ({spec.unavailable_reason})")
                results.append(
                    ModelResult(
                        dataset_key=dataset.key,
                        protocol=dataset.protocol,
                        variant=dataset.variant,
                        label=dataset.label,
                        model_key=spec.key,
                        model_label=spec.label,
                        family=spec.family,
                        n_rows=dataset.n_rows,
                        n_features=dataset.n_features,
                        n_folds=0,
                        r2_mean=None,
                        r2_std=None,
                        rmse_mean=None,
                        rmse_std=None,
                        mae_mean=None,
                        mae_std=None,
                        fit_seconds=0.0,
                        available=False,
                        unavailable_reason=spec.unavailable_reason or "",
                    )
                )
                if on_result:
                    on_result(results)
                continue

            cv = config.effective_cv(spec.family)
            start = time.perf_counter()
            try:
                outcome = _run_model_scored(
                    dataset, model_key, config.random_state, spec_kwargs, cv,
                    config.model_timeout_seconds, logger,
                )
            except Exception as exc:  # noqa: BLE001 - a whole protocol run must never die here
                # Final safety net: `_run_model_scored`/`_run_model_scored_once`
                # already handle the specific crash class found 2026-09-20
                # (docs/experiments.md "Resource-tracker crash"), but this
                # catches anything else unforeseen in the parent-side
                # subprocess bookkeeping (Pipe/Process management is OS-level
                # code, not pure Python we fully control) so one bad
                # (dataset, model) pair can never again take the rest of the
                # sequence down with it.
                outcome = {
                    "status": "error",
                    "reason": f"unexpected error managing worker subprocess: {exc!r}",
                }
            elapsed = time.perf_counter() - start

            if outcome["status"] != "ok":
                reason = outcome.get("reason", outcome["status"])
                _log(logger, f"  {model_key}: {outcome['status']} after {elapsed:.1f}s ({reason})")
                results.append(
                    ModelResult(
                        dataset_key=dataset.key,
                        protocol=dataset.protocol,
                        variant=dataset.variant,
                        label=dataset.label,
                        model_key=spec.key,
                        model_label=spec.label,
                        family=spec.family,
                        n_rows=dataset.n_rows,
                        n_features=dataset.n_features,
                        n_folds=0,
                        r2_mean=None,
                        r2_std=None,
                        rmse_mean=None,
                        rmse_std=None,
                        mae_mean=None,
                        mae_std=None,
                        fit_seconds=elapsed,
                        available=False,
                        unavailable_reason=f"{outcome['status']}: {reason}",
                    )
                )
                if on_result:
                    on_result(results)
                continue

            summary, scores = outcome["summary"], outcome["scores"]
            fold_scores[dataset.key][model_key] = scores

            if model_key == config.paired_baseline:
                baseline_scores = scores

            comparison = None
            if baseline_scores is not None and model_key != config.paired_baseline:
                comparison = paired_r2_comparison(
                    baseline_scores, scores, config.paired_baseline, model_key
                )

            _log(
                logger,
                f"  {model_key}: R2={summary.r2_mean:.3f}+/-{summary.r2_std:.3f} "
                f"({cv.n_splits}x{cv.n_repeats} folds, {elapsed:.1f}s)",
            )

            results.append(
                ModelResult(
                    dataset_key=dataset.key,
                    protocol=dataset.protocol,
                    variant=dataset.variant,
                    label=dataset.label,
                    model_key=spec.key,
                    model_label=spec.label,
                    family=spec.family,
                    n_rows=dataset.n_rows,
                    n_features=dataset.n_features,
                    n_folds=summary.n_folds,
                    r2_mean=summary.r2_mean,
                    r2_std=summary.r2_std,
                    rmse_mean=summary.rmse_mean,
                    rmse_std=summary.rmse_std,
                    mae_mean=summary.mae_mean,
                    mae_std=summary.mae_std,
                    fit_seconds=elapsed,
                    available=True,
                    vs_baseline_fraction_improved=(
                        comparison.fraction_improved if comparison else None
                    ),
                    vs_baseline_wilcoxon_p=(comparison.wilcoxon_p if comparison else None),
                    vs_baseline_n_folds=(comparison.n_matched_folds if comparison else 0),
                )
            )
            if on_result:
                on_result(results)

    return results, fold_scores


def results_to_records(results: list[ModelResult]) -> list[dict]:
    return [asdict(r) for r in results]
