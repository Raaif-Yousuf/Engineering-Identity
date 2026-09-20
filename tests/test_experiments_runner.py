from unittest.mock import MagicMock

import pytest

from ei_model.experiments import models as m
from ei_model.experiments import runner as ei_runner
from ei_model.experiments.config import CvConfig, RunConfig
from ei_model.experiments.models import ModelSpec
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


def test_run_config_marks_unavailable_model(data_dir, monkeypatch):
    # ei_model.models.crowd has landed (feat/crowd-ann-model merged), so
    # "crowd" itself is no longer a naturally-unavailable model to exercise
    # this path with -- and running it for real here would mean 189
    # architectures against this fixture's real (~140-column) feature
    # families, which is far too slow for a unit test (see the wall-clock
    # note in ei_model.models.crowd's module docstring). Instead, stand in
    # a builder that reports itself unavailable, to keep testing the
    # runner's graceful-degradation code path in isolation.
    def _unavailable_spec(seed: int = 0) -> ModelSpec:
        return ModelSpec(
            "broken_model",
            "Intentionally broken",
            "baseline",
            None,
            available=False,
            unavailable_reason="simulated import failure",
        )

    monkeypatch.setitem(m.BUILDERS, "broken_model", _unavailable_spec)

    config = RunConfig(
        name="test_broken_model",
        protocol="P2",
        variants=["after"],
        models=["mean", "broken_model"],
        cv=CvConfig(n_splits=3, n_repeats=1),
        paired_baseline="mean",
    )
    results, _ = run_config(config, data_dir=str(data_dir), logger=lambda _msg: None)

    broken_result = next(r for r in results if r.model_key == "broken_model")
    assert broken_result.available is False
    assert broken_result.unavailable_reason
    assert broken_result.n_folds == 0


def test_crowd_is_registered_and_available():
    # Lightweight registration check (the full fit/predict path, including
    # against realistically wide feature sets, is covered on tiny synthetic
    # data in tests/test_experiments_models.py::test_crowd_spec_available_and_fits).
    spec = m.build_model_spec("crowd", seed=0)
    assert spec.available is True
    assert spec.fit_predict is not None


class _FakeProc:
    """Stands in for `multiprocessing.Process`: already finished, never alive."""

    def __init__(self, exitcode: int = 1):
        self.pid = 4242
        self.exitcode = exitcode

    def start(self) -> None:
        pass

    def join(self, timeout=None) -> None:
        pass

    def is_alive(self) -> bool:
        return False


class _FakeConn:
    def __init__(self, recv_exc: Exception | None = None, recv_value=None):
        self._recv_exc = recv_exc
        self._recv_value = recv_value

    def poll(self) -> bool:
        return True

    def recv(self):
        if self._recv_exc is not None:
            raise self._recv_exc
        return self._recv_value

    def close(self) -> None:
        pass


class _FakeCtx:
    """Stands in for `mp.get_context("spawn")` so tests never spawn a real
    subprocess -- fast, deterministic, and lets us simulate the exact
    "child crashed and the pipe read raised something other than
    EOFError" scenario found 2026-09-20 (see docs/experiments.md
    "Resource-tracker crash") without needing a real loky race to occur."""

    def __init__(self, parent_conn):
        self._parent_conn = parent_conn

    def Pipe(self, duplex=False):
        return self._parent_conn, MagicMock()

    def Process(self, target, args):
        return _FakeProc()


def _fake_dataset():
    return MagicMock(X=None, y=None)


def test_run_model_scored_once_survives_non_eof_recv_failure(monkeypatch):
    # Regression test for the crash that killed P1's `replication.yaml` run
    # ~103 minutes in (results/logs/replication.log): a worker crashed
    # while already handling a joblib/loky resource-tracker exception, its
    # own attempt to report that back over the pipe raised a *second*
    # exception (a broken pipe mid-write), and the parent's `recv()` then
    # raised something other than `EOFError` (simulated here directly).
    # The old code only caught `EOFError` there, so this propagated out of
    # the whole run. It must now come back as a plain "error" outcome.
    fake_conn = _FakeConn(recv_exc=ConnectionResetError("simulated broken pipe on recv"))
    monkeypatch.setattr(ei_runner.mp, "get_context", lambda name: _FakeCtx(fake_conn))

    outcome = ei_runner._run_model_scored_once(
        _fake_dataset(), "lightgbm", seed=0, spec_kwargs={},
        cv=CvConfig(n_splits=3, n_repeats=1), timeout_seconds=60, logger=lambda _msg: None,
    )

    assert outcome["status"] == "error"
    assert (
        "ConnectionResetError" in outcome["reason"]
        or "simulated broken pipe" in outcome["reason"]
    )


def test_run_model_scored_retries_once_after_a_crash(monkeypatch):
    # `_run_model_scored` (the retry wrapper) must retry a crashed attempt
    # exactly once, in a fresh subprocess, and succeed if the retry works --
    # this is the "catch, log, retry that model once, continue" behavior
    # requested after the resource-tracker crash (docs/experiments.md).
    calls = {"n": 0}

    def _fake_once(dataset, model_key, seed, spec_kwargs, cv, timeout_seconds, logger):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "error", "reason": "simulated first-attempt crash"}
        return {"status": "ok", "summary": "fake-summary", "scores": []}

    monkeypatch.setattr(ei_runner, "_run_model_scored_once", _fake_once)

    outcome = ei_runner._run_model_scored(
        _fake_dataset(), "lightgbm", seed=0, spec_kwargs={},
        cv=CvConfig(n_splits=3, n_repeats=1), timeout_seconds=60, logger=lambda _msg: None,
    )

    assert calls["n"] == 2
    assert outcome["status"] == "ok"


def test_run_model_scored_gives_up_after_max_retries(monkeypatch):
    calls = {"n": 0}

    def _always_crashes(dataset, model_key, seed, spec_kwargs, cv, timeout_seconds, logger):
        calls["n"] += 1
        return {"status": "error", "reason": "simulated persistent crash"}

    monkeypatch.setattr(ei_runner, "_run_model_scored_once", _always_crashes)

    outcome = ei_runner._run_model_scored(
        _fake_dataset(), "lightgbm", seed=0, spec_kwargs={},
        cv=CvConfig(n_splits=3, n_repeats=1), timeout_seconds=60, logger=lambda _msg: None,
        max_retries=1,
    )

    assert calls["n"] == 2  # one attempt + one retry, not an infinite loop
    assert outcome["status"] == "error"


def test_run_config_survives_a_crashing_model_and_continues(data_dir, monkeypatch):
    # End-to-end version of the same regression: even if `_run_model_scored`
    # itself somehow raises (the belt-and-suspenders case in `run_config`'s
    # loop), the whole protocol run must not die -- it should record that
    # one model as unavailable/error and keep scoring the rest.
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated unexpected subprocess-management failure")

    monkeypatch.setattr(ei_runner, "_run_model_scored", _boom)

    config = RunConfig(
        name="test_crash_resilience",
        protocol="P2",
        variants=["after"],
        models=["mean", "ridge"],
        cv=CvConfig(n_splits=3, n_repeats=1),
        paired_baseline="mean",
    )
    results, _ = run_config(config, data_dir=str(data_dir), logger=lambda _msg: None)

    assert {res.model_key for res in results} == {"mean", "ridge"}
    assert all(res.available is False for res in results)
    assert all("simulated unexpected subprocess-management failure" in res.unavailable_reason
                for res in results)


def test_run_config_resume_skips_already_recorded_pairs(data_dir):
    # Regression test for the resume capability added so a crashed/partial
    # run (e.g. P1's replication.yaml crash, docs/experiments.md
    # "Resource-tracker crash") never has to recompute rows a previous
    # attempt already checkpointed to CSV.
    config = RunConfig(
        name="test_resume",
        protocol="P2",
        variants=["after"],
        models=["mean", "ridge", "random_forest"],
        cv=CvConfig(n_splits=3, n_repeats=1),
        paired_baseline="mean",
    )
    first_results, _ = run_config(config, data_dir=str(data_dir), logger=lambda _msg: None)
    # Simulate "only mean and ridge finished before the crash" by keeping
    # just those two rows as the checkpoint to resume from.
    carried_over = [r for r in first_results if r.model_key in ("mean", "ridge")]

    calls = []
    real_run_model_scored_once = ei_runner._run_model_scored_once

    def _spy_once(dataset, model_key, seed, spec_kwargs, cv, timeout_seconds, logger):
        calls.append(model_key)
        return real_run_model_scored_once(
            dataset, model_key, seed, spec_kwargs, cv, timeout_seconds, logger
        )

    with pytest.MonkeyPatch.context() as mp_ctx:
        mp_ctx.setattr(ei_runner, "_run_model_scored_once", _spy_once)
        results, _ = run_config(
            config,
            data_dir=str(data_dir),
            logger=lambda _msg: None,
            existing_results=carried_over,
        )

    # Only the model missing from the checkpoint should have actually run.
    assert calls == ["random_forest"]
    assert {r.model_key for r in results} == {"mean", "ridge", "random_forest"}
    # The carried-over rows are the exact same objects/values, not recomputed.
    resumed_ridge = next(r for r in results if r.model_key == "ridge")
    original_ridge = next(r for r in first_results if r.model_key == "ridge")
    assert resumed_ridge.r2_mean == original_ridge.r2_mean
    assert resumed_ridge.fit_seconds == original_ridge.fit_seconds


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
