# Experiments: predicting engineering identity, four ways

This repo's real data is human-subjects research data and is never
committed here (see [`docs/data.md`](data.md)). Everything below --
methods, code, and every number in this document -- is safe to read
publicly: results are reported as aggregates (mean +/- sd across
cross-validation folds, counts, and feature-*family* importances), never as
per-participant predictions, residuals, or raw survey/definition/concept-map
values. No figure in `results/figures/` plots one point per student; the
closest thing to a fit-quality plot here is a feature-family importance bar
chart, which is already an aggregate by construction.

The cohort is engineering undergraduates, mostly first-year, not
first-year-only: `Year` runs 1-4 (After 1416/143/84/146, Before
1440/208/128/153, Diff 877/51/41/43 by year -- about 79% first-year). `Year`
is kept as an ordinary `demographics` feature in every protocol below; it is
not stratified on in the CV splits in this pass.

## Why four protocols, not one

The original research question was "can we predict a student's composite
Engineering Identity (EI) score from survey, definition, and concept-map
data." `docs/data.md` flags a specific problem with the most direct version
of that question, which is worth stating precisely because the first version
of this claim (and of this document) overstated it slightly:

The exact composite formula is now recovered (`EI_inventory.md` section G2):
`EI M Norm = SUM(31 Likert items) / (31*6) * 100`, across four subscales
(Interest: 7 items, Performance/Competence: 8, Recognition: 9, Initiative:
7). Checked column-by-column against the modeling tables: **none of those 31
exact items are literal columns of the modeling inputs** -- so this is not
literal formula leakage; no input column *is* a summand of the target.

What *is* true, and still a real concern: the 7-item
`identity_survey_LEAKAGE_RISK` family is a set of separate, single-item
global self-ratings that map one-to-one onto the same 4 subscales the
formula sums -- e.g. "Improved my engineering performance" is a one-item
proxy for the 8-item Performance/Competence subscale, "I was recognized as
an engineer based on my work and actions" for the 9-item Recognition
subscale, and so on for Interest and Initiative. A naive linear fit of just
those 7 proxy items still reaches R^2 = 0.35 against `ei` -- not because
they *are* formula components, but because a single-item global rating of
"how recognized/interested/performant/driven do you feel" correlates
strongly with the multi-item subscale asking the same thing nine ways. This
is construct-level correlated-proxy inflation (shared method variance), not
literal target leakage, but it has the same practical consequence: a model
handed those 7 columns is largely decoding a correlate of the scoring
formula rather than predicting identity from anything genuinely upstream of
it. P1/P2 below are kept (and still worth reporting separately) for exactly
this reason, just described accurately rather than as "leakage."

So instead of one number, this experiment harness reports four, run on
**identical, seeded, repeated 5-fold x 5-repeat cross-validation splits**
(see "Runtime deviations" below for the one place that isn't quite true) so
the four are directly comparable:

| Protocol | Question | Feature set | Cohort |
|---|---|---|---|
| **P1** replication | What did the original setup measure? | Everything, including the identity-survey proxy block | Full (after/before/diff) |
| **P2** no proxy items | What's left once the subscale-proxy items are removed? | P1 minus `identity_survey_LEAKAGE_RISK` | Full |
| **P3** true prediction | Can *before*-semester data predict *after*-semester EI? | before features (minus identity block) | before/after pid overlap |
| **P3b** persistence baseline | Is P3 doing better than "assume nothing changed"? | P3's features + the student's own before EI | Same as P3 |
| **P4** complete subset | What do concept maps add, holding the cohort fixed? | P2 and P3, with/without the 44 concept-map columns | Concept-map-complete subset only |

P1 is labeled **"includes identity-subscale proxy items"** everywhere in the
results CSVs and figures -- it is the only protocol whose R^2 is expected to
be inflated by the proxy-correlation effect described above, and it is
reported precisely so readers can see how much of the original numbers that
effect explains, not because it is a fair estimate of predictive skill.

## Data layer

`ei_model.data.load_variant` reads a single `<variant>.parquet` per
`docs/data.md`. While this experiment harness was being built, `EI_DATA_DIR`
was regenerated one layer more granular by the concurrent
`feat/concept-map-metrics` work: `<variant>_partial.parquet` (the full
cohort, no concept-map columns at all) and `<variant>_complete.parquet`
(only the concept-map-digitized subset, all 44 concept-map columns
populated, never NaN). That split is exactly what P4 needs, so
`ei_model.experiments.data_access` reads it directly instead of waiting on
or duplicating that branch's own changes to `ei_model.data` -- it falls back
to `load_variant` if the split files aren't present, so it still works
against an older `EI_DATA_DIR` layout. See that module's docstring for the
mechanics. **This is a real seam between two in-flight branches, not a
design decision** -- once `feat/concept-map-metrics` merges and updates
`ei_model.data` itself, this fallback module should be revisited (and
possibly retired in favor of whatever it lands on).

Every builder in `ei_model.experiments.protocols` returns a raw (unimputed,
unscaled) feature matrix. All preprocessing -- median imputation, scaling,
and for XGBoost/LightGBM the hyperparameter search itself -- happens *inside*
each cross-validation fold (`ei_model.experiments.preprocessing`,
`ei_model.experiments.models`), never fit on the full dataset first.

## Models

Every model is scored with the same driver
(`ei_model.evaluation.repeated_kfold_cv_detailed`) on the same fold splits,
so the comparison is apples-to-apples:

- **mean**: `DummyRegressor(strategy="mean")` -- the floor every other model
  must beat.
- **ridge** / **elastic_net**: `RidgeCV` / `ElasticNetCV`, alpha (and
  l1_ratio) selected by an inner CV fit fresh inside every training fold.
- **random_forest**: `RandomForestRegressor`, 300 trees, fixed reasonable
  hyperparameters (no tuning needed to be competitive on data this size).
- **xgboost** / **lightgbm**: `RandomizedSearchCV` (not Optuna -- see below)
  over n_estimators/depth/learning_rate/subsample/colsample/reg_lambda, 15
  iterations, 3-fold inner CV, nested inside every outer training fold.
- **original_ann**: `ei_model.models.original_ann`, completely unchanged
  (512-256-128 Keras network, Adam, EarlyStopping/ReduceLROnPlateau/
  ModelCheckpoint, up to 1000 epochs) -- the owner's original architecture,
  just re-fit inside each CV fold instead of a single 90/10 holdout.
- **improved_ann**: `ei_model.models.improved_ann`, a much smaller 64-32-1
  network with L2 weight decay, lighter dropout, early stopping only, fit on
  quantile-scaled (not z-scored) features, as a 3-seed ensemble (average of
  3 independently initialized fits) to damp the variance a single small net
  shows on datasets this size.
- **stacking**: `StackingRegressor` over ridge + random forest + XGBoost
  (each with its own preprocessing), final estimator RidgeCV.
- **crowd**: the MATLAB wisdom-of-the-crowd reimplementation
  (`ei_model.models.crowd.CrowdANN`, merged via PR #13), run at
  `n_replications=1` (`crowd_replications` in each config) instead of the
  MATLAB original's 100, and subject to the same `model_timeout_seconds`
  every model gets. It is included in P1, P2, P3, and P3b (not P4, by
  design). **It only finished on the single smallest dataset it was
  pointed at (`p2_diff`)** -- every other attempt timed out. See "Why
  this harness's own `crowd` model can't be run at MATLAB's scale" below
  for the measured cost and why.

I used `RandomizedSearchCV` rather than Optuna for the tuned tree models --
both were allowed by the spec, and `RandomizedSearchCV` needed no new
dependency-management surface (Optuna's own storage/pruner config) for what
is, at 15 iterations, already a small search.

## Data quality: implausible "hours per week" values

While first running this harness against real data, `ridge` (a plain
regularized linear model, no tuning to blame) scored R^2 in the hundreds of
thousands negative on `after`, and the unchanged `original_ann` scored R^2
in the tens of millions negative -- both wildly outside anything a genuine
model failure produces. The cause was in the data, not the models: three
self-reported "hours per week" columns (`Hours on Campus Per Week`,
`Engineering Activity on Campus Per Week`, `Non-Engineering Activity on
Campus Per Week`) contain values that are not physically possible (a week
has 168 hours), including a literal `123456789` placeholder in one `after`
row. Counted directly from `EI_DATA_DIR`:

| Table | Column | Rows over 168 | Example values |
|---|---|---:|---|
| after_partial (n=1789) | Hours on Campus Per Week | 8 | 300, 2000, 2250, 123456789 |
| after_partial | Engineering Activity on Campus Per Week | 5 | 490, 1000, 123456789 |
| after_partial | Non-Engineering Activity on Campus Per Week | 7 | 500, 1250, 1100000, 123456789 |
| diff_partial (n=1012) | Hours on Campus Per Week | 5 | 300, 2250, 500 |
| diff_partial | Engineering Activity on Campus Per Week | 3 | 490, 1000, 600 |
| diff_partial | Non-Engineering Activity on Campus Per Week | 3 | 200, 1250, 500 |
| before_partial (n=1929) | (same 3, renamed) | 0 | -- clean |

`ei_model.experiments.cleaning.clean_implausible_hours` nulls any value with
`abs(value) > 168` in these columns (a hard physical bound, not a threshold
fit from the sample, so it's applied once at load time rather than inside
each CV fold -- see that module's docstring) before anything else happens to
the data. `data_access.load_partial`/`load_complete` call it automatically.
Concept-map "Sum"/"Max" complexity metrics that also look large (tens of
thousands) were checked and left alone -- those are expected to scale with
graph size for a metric that sums over vertex pairs, not a data error.

This is flagged here, not fixed silently, because it changes what "the
original numbers" even mean: the owner's original single-90/10-holdout runs
almost certainly hit these same rows sometimes and not other times depending
on the random split, which is a very plausible contributor to the wide,
irreproducible swings across saved prediction files that `EI_inventory.md`
documents (R^2 from -0.72 to +0.45 across 41 files). Whoever maintains
`EI_DATA_DIR`'s build script should decide whether to fix this upstream;
this harness works around it defensively so a handful of bad survey
responses can't dominate every model trained on this data.

## Runtime deviations from the spec

Honesty requires flagging every place a CPU/time budget won this session,
not just averaging over it:

1. **The two ANN models run at 1 repeat (5 folds), not 5 repeats (25
   folds), in P1/P2/P3/P3b.** Every tree/linear/boosted model in this repo
   runs the full spec'd 5x5 = 25 folds; a from-scratch Keras fit per fold
   (especially `original_ann`'s up-to-1000-epoch budget) made 25 folds x
   (1 original + 3 improved-ann seeds) infeasible across 8 dataset configs
   on a single CPU machine in this pass. The reduced ANN folds are still a
   genuine subset of the same seeded splits (repeat 0's 5 folds are
   identical whether `n_repeats` is 1 or 5, since `KFold(random_state=...)`
   only depends on sample count), so the paired comparison against
   `original_ann` is exact for the folds it uses -- just 5 folds' worth of
   evidence instead of 25 for the ANN family specifically.
2. **P4 runs a reduced 3-model set (mean, xgboost, original_ann), not the
   full model list.** Revised mid-session (2026-09-20) once P4 was
   reprioritized to run last: with 8 dataset configs already, `ridge`,
   `elastic_net`, `random_forest`, `lightgbm`, `stacking`, and `crowd` are
   not scored for P4 in this pass -- only the floor (`mean`), the best-tuned
   boosted model (`xgboost`; see point 5 below for why `lightgbm` is
   dropped), and `original_ann` (now included, at the same ann_cv-reduced
   5x1 schedule as every other protocol -- this *fixes* an earlier version
   of this deviation list, which had P4 running no ANN at all with
   `random_forest` standing in as `paired_baseline`). `paired_baseline` for
   P4 is `original_ann`, same as every other protocol.
3. **crowd is not run in P4** (dropped in the reprioritization above; it
   *is* run in P1/P2/P3/P3b -- see the next section for why it wasn't run
   at all earlier in this same session).
4. **Tuning budgets are modest by design**, and were cut further before this
   session's runs: `RandomizedSearchCV` at `n_iter=8` (down from an earlier
   planned 15), 3-fold inner CV, for both XGBoost and LightGBM; `improved_ann`
   at a 1-seed "ensemble" (down from a planned 3), to fit five dataset-config
   protocols in one session on a single, shared machine. Both are real
   nested-CV/ensembling procedures, just shallower ones -- see the git
   history of `configs/*.yaml` for the exact prior values.
5. **Protocol run order was reprioritized mid-session** (2026-09-20) to
   surface the headline P1-vs-P2 comparison first: actual order was P3b
   (already in flight under the just-fixed code when the reprioritization
   request arrived -- restarting it to strictly reorder would have thrown
   away real, already-checkpointed progress for a few minutes' difference,
   so it was left to finish), then P1, P2, P3 (re-run), P4. **If `lightgbm`
   is dropped from a protocol's model list below, it's this same wall-clock
   pressure**: `lightgbm` and `xgboost` are both tuned gradient-boosted
   trees answering the same "does a tuned boosted model beat linear/RF"
   question, so keeping only `xgboost` (already the `importance_model`
   everywhere) is the cheaper way to keep that comparison without paying
   for two tuned-boosting searches per dataset. Check each config's
   `models:` list and this document's Results section for exactly which
   protocols this applied to -- it is not applied uniformly by default.

## Runtime stall: nested n_jobs oversubscription (found and fixed 2026-09-19)

The first full run of `configs/true_prediction.yaml` (P3) hung for over an
hour with `results/logs/true_prediction.log` completely silent (last
`R2=...` line was `xgboost`, at 1004.0s; `lightgbm` never printed one) while
the runner process (PID 12480) and its 16 spawned `loky` worker processes
kept consuming real CPU (measured: ~50-65% each over repeated 10-15s
sampling windows, total system CPU ~85%) -- not a classic zero-CPU deadlock,
but a genuine non-progressing hang under real load. Root cause, confirmed by
reading `src/ei_model/experiments/models.py`: `_lightgbm_search` (and
`_xgboost_search`, and `stacking_spec`'s base estimators) wrapped an
estimator constructed with `n_jobs=-1` **inside** a `RandomizedSearchCV(...,
n_jobs=-1)`. `RandomizedSearchCV` at `n_jobs=-1` already spawns one process
per `n_iter x inner_cv` fit (up to 16 on this machine); each of those
processes then also tried to claim all 16 cores for LightGBM's/XGBoost's own
internal threading. That double layer of `n_jobs=-1` oversubscribes the
machine (16 processes x up to 16 threads each, competing for 16 cores),
which manifested as XGBoost merely being ~4.5x slower than a single-layer
run (1004s at the cut `n_iter=8` budget) and LightGBM specifically as a de
facto hang (LightGBM's own threading is known to interact badly with
process-based parallelism on Windows).

**Fix** (`src/ei_model/experiments/models.py`): the inner estimator inside
every `RandomizedSearchCV`/`StackingRegressor` now uses `n_jobs=1`, leaving
exactly one parallelism layer (the outer search/stack). Standalone
`random_forest_spec` is untouched (`n_jobs=-1`, no outer parallel wrapper,
so no oversubscription there).

**Two permanent safety nets added at the same time, not just this one-off
fix**, since a hang or crash of *any* kind should never again take down an
entire config's run:
1. **Per-model wall-clock timeout** (`RunConfig.model_timeout_seconds`,
   default 1200s / 20 min): `runner.py`'s `_run_model_scored` now runs each
   (dataset, model) score in its own subprocess (`multiprocessing`, spawn
   context) and kills the whole process tree (`taskkill /F /T` on Windows,
   so orphaned `loky` grandchildren can't leak) if it doesn't finish in
   time, recording that row as `available=False,
   unavailable_reason="timeout: ..."` and moving on to the next model.
2. **Incremental CSV checkpointing**: `run_config` now takes an `on_result`
   callback invoked after *every* model (available, unavailable, errored, or
   timed out), and `cli.py` wires this to rewrite `results/<name>.csv` after
   each one, not just once at the very end. A kill, crash, or a model that
   hits the timeout above no longer loses every row scored before it.

**Consequence for this session's numbers**: P3 (`true_prediction`) had to be
re-run from scratch after this fix, since the hung run produced no CSV
(nothing had been checkpointed yet -- it predates fix #2). Every other
protocol in the Results section below ran under the fixed code.

**Also enabled by the same session's `feat/crowd-ann-model` merge (PR
#13)**: `crowd` is now a real, importable model (`ei_model.models.crowd.
CrowdANN`), not a permanent `available=False` stub. Its cost is dominated by
feature count, not neuron count (every cascade layer re-consumes the raw
input) -- the module's own docstring records a measured 189-architecture x
1-replication run at 341s on 900 rows. Given that, `crowd` now: (a) defaults
to `n_replications=1`, down from this harness's earlier planned default of
10 (`RunConfig.crowd_replications`); (b) shares the ANN family's reduced
5-fold x 1-repeat schedule via `effective_cv`, instead of the full 5x5; (c)
is still subject to the 1200s per-dataset timeout above, so on the larger
variants (`after`/`before`, ~1800-1900 rows) it may still time out and show
`available=False, unavailable_reason="timeout: ..."` for some datasets --
that itself is reported as a real, honest finding below rather than hidden.

## Resource-tracker crash: unhandled parent-side exception on a crashed model worker (found and fixed 2026-09-20)

`replication.yaml` (P1) died about 103 minutes into its run, mid `p1_diff`
right after `lightgbm`'s tuning search, with the top-level
`python -m ei_model.experiments run` process itself exiting non-zero --
not just one model marked unavailable, the *whole* config run stopped,
losing P1's remaining models for `p1_diff` (`improved_ann`, `stacking`,
`crowd` never ran for that variant; `p1_after`/`p1_before` were unaffected,
since incremental checkpointing had already written their rows).
`results/logs/replication.log` shows the immediate trigger:

```
joblib.externals.loky.process_executor.TerminatedWorkerError: A worker
process managed by the executor was unexpectedly terminated. ...
...
File ".../ei_model/experiments/runner.py", line 77, in _model_worker
    conn.send({"status": "error", "reason": f"{exc!r}\n{traceback.format_exc()}"})
File ".../multiprocessing/connection.py", line 289, in _send_bytes
    ov, err = _winapi.WriteFile(self._handle, buf, overlapped=True)
BrokenPipeError: [WinError 232] The pipe is being closed
```

followed by thousands of lines of

```
File ".../joblib/externals/loky/backend/resource_tracker.py", line 386, in main
    del cache[rtype][name]
KeyError: 'C:\\Users\\...\\Temp\\joblib_memmapping_folder_5948_..._...'
```

**Root cause, in two layers:**

1. **The underlying trigger** is a known class of joblib/loky bug on
   long-running Windows processes that create and tear down many
   `Parallel(n_jobs=-1)` executors in the same process lifetime.
   `lightgbm`'s tuning is `repeated_kfold_cv_detailed` (25 outer folds under
   P1's full 5x5 CV) each doing its own `RandomizedSearchCV(n_jobs=-1)` fit
   -- so a single (dataset, model) subprocess can spin up ~25+ separate
   loky executors and memmapping temp folders back to back. Eventually the
   background resource-tracker process's bookkeeping for these desyncs
   (two cleanup paths racing to remove the same temp-folder cache entry) and
   it raises an uncaught `KeyError` deleting an entry that's already gone.
   That, in turn, made one of the in-flight `RandomizedSearchCV` fits see a
   `TerminatedWorkerError` (its executor's worker was killed as a side
   effect). This part is upstream joblib/loky behavior, not a bug unique to
   this repo, and is not fully fixed here -- see point 2 for why it no longer
   matters as much.
2. **The bug that let it take down the entire run** was ours, in
   `src/ei_model/experiments/runner.py`. Every model already runs in its
   own subprocess with a hard timeout specifically so a hang or crash can't
   block the rest of the sequence (see "Runtime stall" above) -- but the
   crash-reporting path had a gap. `_model_worker` catches the
   `TerminatedWorkerError` and tries to report it back over the pipe with
   `conn.send(...)`; here, that `send` itself raised a *second*, unrelated
   `BrokenPipeError` (the pipe was already closing). That secondary
   exception was never caught, so the child process died with a truncated,
   partially-written frame still in the pipe buffer. Back in the parent,
   `_run_model_scored`'s pipe-read logic was:
   ```python
   if parent_conn.poll():
       try:
           return parent_conn.recv()
       except EOFError:
           return {"status": "error", "reason": "worker closed its pipe without a result"}
   ```
   `parent_conn.recv()` on a truncated frame does not necessarily raise
   `EOFError` -- it can raise other exceptions depending on exactly where
   the write was cut off (a `ConnectionResetError`/`OSError`, or a pickle
   error). Since only `EOFError` was caught, that exception propagated
   straight out of `_run_model_scored`, out of `run_config`'s per-model
   loop, and killed the entire protocol run. This is the actual, precise,
   code-verifiable cause of "one model's crash took down the whole 103-minute
   run" -- the timeout/subprocess isolation was already correct in spirit,
   this one `except` clause just wasn't broad enough to catch every way a
   crashed child can fail to hand back a clean result.

**Fix** (`src/ei_model/experiments/runner.py`):
- `_model_worker`'s own `conn.send(...)` inside its except-block is now
  itself wrapped in a `try/except: pass` -- if the pipe is already gone,
  there is nothing more the worker can do; exiting non-zero is enough,
  since the parent already treats "child died without sending a result" as
  its own error outcome.
- The parent-side read (now `_run_model_scored_once`) catches any
  `Exception` from `parent_conn.recv()`, not just `EOFError`, and reports
  it as a normal `{"status": "error", ...}` outcome instead of letting it
  propagate.
- **New retry-once wrapper** `_run_model_scored` calls
  `_run_model_scored_once` and, if the outcome is `"error"` (a crash --
  not `"timeout"` or `"unavailable"`, which retrying can't fix), retries
  exactly once in a brand-new subprocess before giving up. A fresh
  subprocess means a fresh resource tracker, which is exactly the reset
  that gives this class of transient loky race a real chance of not
  recurring. `max_retries=1` by default.
- **Belt-and-suspenders**: `run_config`'s per-model loop now also wraps the
  call to `_run_model_scored` itself in a `try/except Exception`, so any
  *other*, still-unforeseen failure in the parent-side subprocess/pipe
  bookkeeping (which is OS-level code, not something this harness fully
  controls) can never again take down the rest of a multi-hour run -- it's
  recorded as an `unavailable`/error row for that one model and the
  sequence continues.
- Regression tests added in `tests/test_experiments_runner.py`: a fake
  `mp.get_context`/`Process`/`Connection` reproduces the exact "child
  crashed, `recv()` raises something other than `EOFError`" scenario
  without needing a real loky race, plus tests for the retry-once behavior,
  giving up after the retry, and the `run_config`-level safety net.

**Consequence for P1's numbers**: `p1_diff`'s `lightgbm`, `improved_ann`,
`stacking`, and `crowd` rows were never scored in the run that crashed;
`p1_after`/`p1_before` (fully scored) and `p1_diff`'s first 6 models
(already checkpointed) survived. P1 was re-run in full under the fixed
code to fill in the rest -- see the Results section for the final numbers
and whether the crash recurred (it should now show as at most a logged
retry, never a dead run).

## The MATLAB crowd baseline, for context

Independent of this harness's own `crowd` model hook (not runnable yet, see
above), `EI_inventory.md` section G1 reconstructed a real, held-out-scored
number for the original MATLAB "wisdom of the crowd" model, on its 44-feature
(29 complexity + 15 C&V keyword) concept-map-only schema, EI Diff target,
using the original's own fixed (non-random) first-130-rows test split:

| Split | R^2 | n |
|---|---:|---:|
| Train (in-sample) | 0.140 | 521 |
| **Test (held-out)** | **-0.075** | 130 |

The crowd fit its own training data poorly (R^2 = 0.14) and failed to
generalize at all to held-out students (R^2 = -0.075, worse than predicting
the mean). **Only the test number is a fair comparison point against this
repo's CV results** -- every R^2 reported below is a held-out-fold average,
never an in-sample fit, so the right comparison is test-to-test:
MATLAB test R^2 -0.075 vs. this repo's protocol R^2s. The paper draft's own
unconfirmed recollection of "R^2 approx. 0.2" for the full-feature MATLAB
model is not supported by this or any other recovered artifact (see
`EI_inventory.md` sections 1 and G1) -- treat "~0.2" as unverified, and
-0.075 as the best-evidenced number for what the original MATLAB approach
actually achieved on data of this kind.

## Why this harness's own `crowd` model can't be run at MATLAB's scale

`ei_model.models.crowd.CrowdANN` is a real, importable, tested
reimplementation (PR #13) of the MATLAB architecture-sweep-and-average
design (189 cascade-forward architectures x N replications, averaged). It
is wired into every protocol that includes it (P1, P2, P3, P3b -- not P4,
by design) at a deliberately reduced `n_replications=1`
(`RunConfig.crowd_replications`), one-tenth of the harness's own originally
planned default of 10, and 1/100th of the MATLAB original's 100
replications, specifically to see whether it can finish at all within the
`model_timeout_seconds=1200` (20 min) budget every model gets.

Measured, not estimated, directly from the real production runs
(`results/replication.csv`, `results/no_leakage.csv`,
`results/true_prediction.csv`, `results/persistence_baseline.csv`):

| Dataset | n_rows | n_features | crowd result at n_replications=1 (189 nets) |
|---|---:|---:|---|
| p1_after | 1789 | 144 | timed out (killed at 1204.2s) |
| p1_before | 1929 | 198 | timed out (killed at 1202.2s) |
| p1_diff | 1012 | 149 | timed out (killed at 1204.0s) |
| p2_after | 1789 | 137 | timed out (killed at 1202.2s) |
| p2_before | 1929 | 191 | timed out (killed at 1202.0s) |
| **p2_diff** | **1012** | **142** | **finished: R^2 = -0.178 +/- 0.091, 963.3s** |
| **p3_before_to_after** | **1012** | **191** | **finished: R^2 = -0.052 +/- 0.085, 1139.5s** |
| **p3b_before_to_after** | **1012** | **192** | **finished: R^2 = 0.121 +/- 0.069, 1116.9s** |

The pattern is clear once all three completions are in the same table:
`crowd` only ever finished on the three ~1012-row datasets (the `diff` /
`before_to_after` variants); every ~1789-1929-row dataset (`after`,
`before`) timed out regardless of feature count (137-198 features across
both groups). Row count, not feature count alone, is the dominant driver
of wall-clock cost here -- though the module's own docstring is still
correct that parameter count (and therefore cost) scales with feature
count, not neuron count, since every cascade layer re-consumes the raw
input; both effects are real, and row count is the one that happens to
separate "finishes in ~950-1150s" from "times out at 1200s" for the
datasets actually tried.

**Extrapolated cost of a MATLAB-parity run (100 replications instead of
1).** Scaling the slowest of the three real completions (`p3_before_to_after`,
1139.5s) linearly with replication count is justified here because 189
architectures already exceeds this machine's core count at `n_jobs=-1`, so
adding replications queues more of the same work rather than changing the
parallelism regime. That gives 1139.5 x 100 = 113,950s, **about 31.7 hours
for one dataset** -- and it's one of the cheapest three tried. The other
five (larger) datasets already time out at *one* replication, so each
would cost more than that, not less. Running `crowd` at MATLAB parity (100
replications) across all 8 P1/P2/P3/P3b dataset variants would cost on the
order of 250+ hours (more than 10 days) of continuous single-machine
compute: outside any practical budget for this project.

**Conclusion, stated plainly.** The crowd model cannot be evaluated on
equal terms with every other model in this harness (all of which finish in
seconds to a few hundred seconds) within a practical time budget. The
three honest numbers available (`p2_diff` -0.178, `p3_before_to_after`
-0.052, `p3b_before_to_after` +0.121, all at 1/100th the original
replication count) are inconsistent in sign and don't support a single
summary claim about how the reimplementation performs -- they are real
measurements, but only three, and only at a fraction of the intended
ensemble size. They are broadly consistent with the MATLAB original's own
poor held-out performance (R^2 = -0.075) in the sense that none of the four
numbers together suggest a strong, reliable model, but this is not a fair
apples-to-apples comparison to either MATLAB (different replication count,
different, much narrower, feature set) or to this harness's other models (a
tiny fraction of their compute budget). Every other `crowd` row in the
tables below is `available=False, unavailable_reason="timeout: ..."` and
should be read as "not evaluable within budget," not as "performs badly
there."

## Results

Full per-protocol tables, from `results/*.csv` (nothing below is
hand-typed or estimated). "vs original_ann" is the paired comparison over
the 5 folds `original_ann` itself was scored on (`ann_cv`'s reduced
schedule -- see "Runtime deviations" above): fraction of those 5 folds
where the challenger's R^2 beat `original_ann`'s, and the two-sided
Wilcoxon signed-rank p-value on the paired per-fold differences. At n=5
matched folds, p=0.0625 is the smallest p-value the test can produce (all 5
folds agree), so it recurs often below; it is not a computation error.
Comparisons left blank happened when `original_ann`'s own row for that
dataset was a *carried-over* (resumed, not freshly scored) row, so no
fold-level scores existed to pair against in that run -- documented in
"Resuming a partial run" below, and visible directly in `p1_diff` (P1,
crashed mid-run and resumed) and every non-`original_ann`/`mean`/`ridge`/
`elastic_net`/`random_forest` model in P3b (5 rows carried over from an
earlier session, only `xgboost`/`improved_ann`/`stacking`/`crowd` freshly
scored this run).

### P1: replication (includes identity-subscale proxy items)

| Model | p1_after (n=1789, 144 feat) | p1_before (n=1929, 198 feat) | p1_diff (n=1012, 149 feat) |
|---|---:|---:|---:|
| mean | -0.004 +/- 0.005 | -0.002 +/- 0.002 | -0.008 +/- 0.009 |
| ridge | 0.401 +/- 0.035 | 0.380 +/- 0.041 | 0.033 +/- 0.036 |
| elastic_net | 0.404 +/- 0.034 | 0.392 +/- 0.041 | 0.033 +/- 0.027 |
| random_forest | 0.400 +/- 0.040 | 0.352 +/- 0.034 | 0.022 +/- 0.036 |
| xgboost | 0.400 +/- 0.041 | 0.372 +/- 0.039 | 0.005 +/- 0.050 |
| lightgbm | 0.385 +/- 0.040 | 0.351 +/- 0.039 | -0.002 +/- 0.038 |
| original_ann | 0.281 +/- 0.027 | 0.252 +/- 0.076 | -0.054 +/- 0.069 |
| improved_ann | 0.340 +/- 0.038 | 0.326 +/- 0.079 | -0.030 +/- 0.061 |
| **stacking** | **0.417 +/- 0.040** | **0.387 +/- 0.042** | **0.029 +/- 0.026** |
| crowd | unavailable (timeout) | unavailable (timeout) | unavailable (timeout) |

vs original_ann: every challenger beats it in 5/5 matched folds
(p=0.0625) on `p1_after`/`p1_before`. On `p1_diff`, only `ridge`/
`elastic_net`/`random_forest` (4/5 folds, p=0.125-0.1875) and `xgboost`
(3/5, p=0.3125) have a comparison at all -- `lightgbm`/`improved_ann`/
`stacking`/`crowd` were scored in the post-crash resumed run and
`original_ann`'s fold scores from the pre-crash run weren't available to
pair against (the blank-comparison caveat above, concretely).

### P2: no proxy items (identity-subscale proxy items excluded -- the honest, leakage-free number)

| Model | p2_after (n=1789, 137 feat) | p2_before (n=1929, 191 feat) | p2_diff (n=1012, 142 feat) |
|---|---:|---:|---:|
| mean | -0.004 +/- 0.005 | -0.002 +/- 0.002 | -0.008 +/- 0.009 |
| ridge | 0.274 +/- 0.030 | 0.267 +/- 0.042 | 0.013 +/- 0.031 |
| elastic_net | 0.281 +/- 0.029 | 0.276 +/- 0.043 | 0.014 +/- 0.020 |
| random_forest | 0.269 +/- 0.031 | 0.241 +/- 0.046 | 0.004 +/- 0.035 |
| xgboost | 0.283 +/- 0.031 | 0.266 +/- 0.046 | -0.015 +/- 0.049 |
| original_ann | 0.186 +/- 0.029 | 0.153 +/- 0.066 | -0.100 +/- 0.067 |
| improved_ann | 0.242 +/- 0.034 | 0.228 +/- 0.085 | -0.033 +/- 0.032 |
| **stacking** | **0.294 +/- 0.030** | **0.278 +/- 0.050** | 0.012 +/- 0.023 |
| crowd | unavailable (timeout) | unavailable (timeout) | -0.178 +/- 0.091 |

`crowd`'s only completion anywhere in P1/P2 was `p2_diff` (963.3s, just
inside the 1200s budget) -- see the crowd-cost section above for the full
comparison across every dataset it was pointed at, including its two other
completions in P3/P3b. vs original_ann: every challenger beats it 5/5
(p=0.0625) on `p2_after`/`p2_before`. On `p2_diff`, all challengers except
`xgboost` beat it in 5/5 folds (p=0.0625); `xgboost` and `crowd` beat it
4/5 (p=0.0625) and `improved_ann` 3/5 (p=0.3125) -- `p2_diff` is uniformly
the hardest dataset in this protocol.

**Headline finding: P1 -> P2.** Removing the 7-item proxy block drops the
best model's R^2 from 0.417 to 0.294 on the `after` variant (0.387 ->
0.278 on `before`) -- a real, substantial, expected drop. This is the
single largest effect in this entire study, larger than any
model-architecture choice.

### P3: true prediction (before -> after, no identity block; n=1012, 191 features)

| Model | R^2 mean +/- sd | vs original_ann (frac. improved / Wilcoxon p, n=5) |
|---|---:|---|
| mean | -0.008 +/- 0.010 | 0.4 / 0.8125 |
| ridge | 0.106 +/- 0.041 | 1.0 / 0.0625 |
| elastic_net | 0.106 +/- 0.039 | 1.0 / 0.0625 |
| random_forest | 0.106 +/- 0.049 | 1.0 / 0.0625 |
| xgboost | 0.109 +/- 0.054 | 1.0 / 0.0625 |
| improved_ann | 0.062 +/- 0.062 | 1.0 / 0.0625 |
| **stacking** | **0.115 +/- 0.051** | 1.0 / 0.0625 |
| crowd | -0.052 +/- 0.085 | 0.4 / 0.625 |
| original_ann | -0.018 +/- 0.079 | (baseline) |

This is the only protocol here with a genuine temporal gap between
features and target, and it shows: even the best model (`stacking`,
R^2=0.115) explains a small fraction of end-of-semester identity from
start-of-semester data alone. `original_ann` is negative (worse than
predicting the mean). `crowd` finished this time (1139.5s, just inside the
1200s budget) at R^2=-0.052 -- worse than every non-baseline model, and
its comparison to `original_ann` (0.4/0.625) is not significant.

### P3b: persistence baseline (P3's features + before-semester ei; n=1012, 192 features)

| Model | R^2 mean +/- sd | vs original_ann |
|---|---:|---|
| mean | -0.008 +/- 0.010 | 0.0 / 0.0625 |
| ridge | 0.222 +/- 0.047 | 1.0 / 0.0625 |
| **elastic_net** | **0.288 +/- 0.049** | 1.0 / 0.0625 |
| random_forest | 0.281 +/- 0.057 | 1.0 / 0.0625 |
| xgboost | 0.273 +/- 0.056 | blank (see caveat) |
| improved_ann | 0.080 +/- 0.059 | blank (see caveat) |
| stacking | 0.284 +/- 0.057 | blank (see caveat) |
| crowd | 0.121 +/- 0.069 | blank (see caveat) |
| original_ann | 0.136 +/- 0.065 | (baseline; a carried-over row this run) |

**P3 vs. P3b, the "does knowing where a student started beat genuine
prediction" question**: yes, clearly. Adding one column (the student's own
before-semester EI score) takes the best model from R^2=0.115 (P3) to
R^2=0.288 (P3b, `elastic_net`, narrowly ahead of `stacking`'s 0.284) --
more than double. Most of what looks like "predicting identity change"
in P3 is not upstream signal about *why* identity changes; a large part of
it is well explained by "the student already had a stable identity level
at the start of the semester," a persistence effect, not a genuinely new
prediction. `xgboost`/`improved_ann`/`stacking`/`crowd`'s comparisons
against `original_ann` are blank because `original_ann`'s row here was
carried over from an earlier session (5/10 rows were already checkpointed
before this session's run resumed the remaining `xgboost`/`improved_ann`/
`stacking`/`crowd`) -- a concrete instance of the caveat under "Resuming a
partial run" below, not a missing computation.

### P4: complete subset (concept-map-complete cohort, with/without concept-map columns; reduced model set)

| Dataset | n | mean | xgboost | original_ann | xgboost vs original_ann |
|---|---:|---:|---:|---:|---|
| p4_p2_after_no_cm | 1074 | -0.006 +/- 0.008 | 0.272 +/- 0.055 | 0.150 +/- 0.042 | 1.0 / 0.0625 |
| p4_p2_after_with_cm | 1074 | -0.006 +/- 0.008 | 0.275 +/- 0.050 | 0.151 +/- 0.017 | 1.0 / 0.0625 |
| p4_p2_before_no_cm | 1284 | -0.004 +/- 0.005 | 0.259 +/- 0.046 | 0.173 +/- 0.071 | 1.0 / 0.0625 |
| p4_p2_before_with_cm | 1284 | -0.004 +/- 0.005 | 0.256 +/- 0.053 | 0.134 +/- 0.098 | 1.0 / 0.0625 |
| p4_p2_diff_no_cm | 479 | -0.012 +/- 0.017 | -0.040 +/- 0.066 | -0.086 +/- 0.094 | 0.8 / 0.4375 |
| p4_p2_diff_with_cm | 479 | -0.012 +/- 0.017 | -0.050 +/- 0.056 | -0.076 +/- 0.101 | 0.4 / 1.0 |
| p4_p3_before_to_after_no_cm | 714 | -0.007 +/- 0.006 | 0.133 +/- 0.045 | -0.011 +/- 0.059 | 1.0 / 0.0625 |
| p4_p3_before_to_after_with_cm | 714 | -0.007 +/- 0.006 | 0.122 +/- 0.059 | -0.070 +/- 0.075 | 1.0 / 0.0625 |

**Do the 44 concept-map columns help, holding the cohort fixed?** No,
not measurably. `xgboost` with concept-map features beats without by
+0.003 on `after`, but *loses* by -0.003 on `before`, -0.010 on `diff`, and
-0.011 on `before_to_after` -- differences on the order of the run-to-run
noise (r2_std is 0.05-0.06 on every one of these datasets), not a
consistent signal in either direction. `original_ann` shows the same
pattern (mixed sign, small magnitude). The concept-map complexity metrics
do not add usable predictive signal beyond what the survey/definition
features already provide, on this cohort and this feature encoding.

### Overall best model and its feature-family importance

`stacking` (ridge + random forest + XGBoost, RidgeCV final estimator) is
the best or tied-for-best model in P1, P2, and P3, and second-best by a
margin smaller than one standard deviation in P3b (`elastic_net` 0.288 vs.
`stacking` 0.284). It is the single most consistent top performer across
this study. Its permutation-importance-by-feature-family breakdown is
reported on `p2_after` (the best-scoring *honest*, leakage-free dataset it
was run on, R^2=0.294) rather than `p1_after` (higher, R^2=0.417, but
inflated by the identity-survey proxy block, which would trivially
dominate the chart the same way it dominates `replication_importance.csv`)
-- see `results/best_model_stacking_importance.csv` /
`results/figures/best_model_stacking_importance.png`:

| Feature family | Importance (sum of permutation R^2 drop) |
|---|---:|
| **self_identification** | **0.177** |
| experiences (curricular/co-curricular/extracurricular) | 0.056 |
| career_achievements | 0.019 |
| demographics | 0.014 |
| influences | 0.010 |
| definitions_ed | 0.003 |
| keywords_skey | 0.002 |
| role_er | -0.002 |

`self_identification` dominates by an order of magnitude over every other
family -- consistent with the same ordering `xgboost`'s own per-protocol
importance chart shows for P2 (`no_leakage_importance.csv`). The
concept-map/keyword/definition families contribute little to nothing
(`role_er` is slightly negative, i.e. permuting it very slightly *helped*,
within noise of zero).

See `results/figures/summary_r2_by_protocol.png` /
`results/summary_r2_by_protocol.csv` for the cross-protocol view: each
protocol's best model vs. `original_ann`, averaged across that protocol's
datasets, with error bars.

## Limitations not addressed in this pass

- **No per-Year stratification or breakdown.** The cohort spans Years 1-4
  (about 79% first-year); `Year` is included as an ordinary feature in every
  protocol, but CV folds are not stratified by it, and no per-year R^2
  breakdown is reported. Left for a follow-up rather than done partially.
- **Predicting *change* in identity over a semester is close to
  unpredictable for every model tried, not just the weak ones.** P1's
  `diff` variant and P2's `diff` variant (identity `after - before`) are
  the hardest dataset in every protocol that includes them: best R^2 is
  0.029 (P1) and 0.014 (P2) -- both barely above the `mean` floor's
  -0.008, and every model including `stacking` clusters within about 0.05
  R^2 of zero. P3 (genuine before -> after prediction, no `ei_before`
  feature) tops out at R^2=0.115. Only P3b, which is handed the student's
  own before-semester score directly, reaches a clearly non-trivial R^2
  (0.288) -- and that is a persistence effect ("this student already had a
  stable identity level"), not evidence that semester-over-semester *change*
  itself is predictable from the available features. Taken together: this
  cohort's engineering-identity *change* has little to no signal in any
  survey/definition/concept-map feature set tried here, for any of the ten
  model architectures tried, tuned or not. This is a property of the
  prediction target and this feature set, not a failure of any one model.
- **Concept-map features (44 columns) show no measurable contribution**
  once the cohort is held fixed (P4, "Do the 44 concept-map columns help"
  above) -- differences between with/without runs are within run-to-run
  noise in both directions.
- **The `crowd` model could only be evaluated on 3 of the 8 dataset
  variants it was pointed at**, each at 1/100th its intended replication
  count, and cannot be run at MATLAB parity within any practical time
  budget on this hardware (see "Why this harness's own `crowd` model can't
  be run at MATLAB's scale" above). Its three real numbers should not be
  read as a confident characterization of the reimplementation's quality.

## Reproducing this

```bash
export EI_DATA_DIR=/path/to/Engineering-Identity-data
python -m ei_model.experiments run --config configs/replication.yaml          # P1
python -m ei_model.experiments run --config configs/no_leakage.yaml           # P2
python -m ei_model.experiments run --config configs/true_prediction.yaml      # P3
python -m ei_model.experiments run --config configs/persistence_baseline.yaml # P3b
python -m ei_model.experiments run --config configs/complete_subset.yaml      # P4
```

Each run writes `results/<name>.csv` (one row per dataset x model),
`results/figures/<name>_r2.png` (grouped bar chart, R^2 mean +/- sd), and,
for the configured `importance_model`, `results/<name>_importance.csv` /
`results/figures/<name>_importance.png` (permutation importance summed by
feature family, on the best-scoring dataset for that model). Every one of
those artifacts is an aggregate; see the top of this document.

## Resuming a partial run

Add `--resume` to any of the commands above (e.g.
`python -m ei_model.experiments run --config configs/replication.yaml --resume`)
to skip any `(dataset, model)` pair already present in that config's
`results/<name>.csv` instead of recomputing it -- exactly what a crashed or
interrupted run (like the one in "Resource-tracker crash" above) needs to
finish without burning CPU on models that already have a real, checkpointed
score. Without `--resume` (the default), a re-run always starts from
scratch and overwrites the existing CSV, same as before this flag existed.

One accuracy trade-off: the paired baseline-vs-challenger comparison
(`vs_baseline_*` columns) needs the baseline model's *fold-level* scores,
which aren't saved in the CSV (only the aggregate mean/sd is) -- so if a
dataset's baseline row (usually `original_ann`) was itself one of the
carried-over, not-recomputed rows, newly scored models for that dataset get
`vs_baseline_*` left blank rather than a real paired comparison. The R^2 /
RMSE / MAE numbers themselves are unaffected either way. This is checked
directly in `tests/test_experiments_runner.py::test_run_config_resume_skips_already_recorded_pairs`
and `tests/test_experiments_cli.py::test_cli_run_resume_keeps_existing_rows_and_adds_a_new_model`.
