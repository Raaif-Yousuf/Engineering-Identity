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
  (`ei_model.models.crowd`), owned by a parallel worker and not edited here.
  **Not included in any of the run results below** -- as of this writing it
  had not merged to `main`. The registry (`ei_model.experiments.models.crowd_spec`)
  already wires it in behind a try/import, defaulting to 10 replications
  instead of the MATLAB original's 100 to keep a CPU run sane; re-run with
  `replications=100` once timing allows, and say so if it still isn't
  affordable.

I used `RandomizedSearchCV` rather than Optuna for the tuned tree models --
both were allowed by the spec, and `RandomizedSearchCV` needed no new
dependency-management surface (Optuna's own storage/pruner config) for what
is, at 15 iterations, already a small search.

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
2. **P4 runs no ANN models at all.** P4 already fans out to 8 dataset
   configs; adding two Keras models on top was not affordable here. P4's
   paired-comparison baseline is `random_forest`, not `original_ann` --
   there is no original-ANN comparison for P4 in this pass.
3. **crowd is not run** (see above) -- not merged yet, not a budget choice.
4. **Tuning budgets are modest by design**: 15 RandomizedSearchCV
   iterations, 3-fold inner CV, for both XGBoost and LightGBM. This is a
   real nested-CV tuning loop, just a shallow one.

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

## Results

<!-- Filled in from results/*.csv after each protocol's run. See that
directory for the exact numbers this table summarizes; nothing here is
hand-typed. -->

*(placeholder -- populated once `results/*.csv` exist for all four
protocols; see the PR that adds them.)*

## Limitations not addressed in this pass

- **No per-Year stratification or breakdown.** The cohort spans Years 1-4
  (about 79% first-year); `Year` is included as an ordinary feature in every
  protocol, but CV folds are not stratified by it, and no per-year R^2
  breakdown is reported. Left for a follow-up rather than done partially.

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
