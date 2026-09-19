# Data card: canonical EI modeling tables

Aggregates only, computed from the real (never-committed) tables. No
per-participant values, free text, or demographics values appear below.

## Tables

Each of the three variants (after/before/diff) ships as two completeness
levels, loaded via `ei_model.data.load_variant(variant, completeness)` from
`EI_DATA_DIR/<variant>_<completeness>.parquet`:

- **partial**: the full survey cohort, no concept-map columns (matches the
  source paper's 144/198/149-metric "partial" tables).
- **complete**: only participants whose hand-drawn concept map has been
  digitized/coded, with the 44 extra concept-map columns added (matches the
  paper's 188-metric "complete" tables).

Each row is one participant, keyed by a salted-hash `pid` (SHA-256 of
`salt + Participant Code`, codes normalized by stripping whitespace and
uppercasing before hashing; the salt lives only in `EI_DATA_DIR/salt.txt`,
never in git). The composite engineering identity target is column `ei`.

| Variant | Rows (partial) | Rows (complete) | Feature cols (partial) | Feature cols (complete) | `ei` mean | `ei` sd | `ei` observed min/max | Valid range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| after  | 1789 | 1074 | 144 | 188 | 21.61 | 15.04 | -46.24 / 50.00 | -50..50 |
| before | 1929 | 1284 | 198 | 242 | 20.93 | 13.74 | -31.72 / 50.00 | -50..50 |
| diff   | 1012 |  479 | 149 | 193 |  1.11 | 13.15 | -55.91 / 68.82 | -100..100 |

(`ei` mean/sd/range above are for the `partial` table; `complete` values are
close but not identical, since it's a different, smaller cohort -- e.g.
after/complete mean=22.40, sd=14.47.) `ei` has 0% missingness in all six
tables, and every other feature column also has ~0% missingness within a
given completeness level (concept_map columns are simply absent from
`partial`, not sparse within it).

Source: UTD-Research `Engineering Identitiy/files/` canonical workbooks
(`EIAfter ... Whole.xlsx`, `EIBefore ... Whole_Final Dup Cleaned 1.xlsx`,
`EIAfter ... Diff.xlsx`), each with `Input Vectors`/`Targets` sheets merged
on Participant Code (verified 1:1, zero duplicates, identical participant
sets between sheets before merging -- never assumed from row order).

### Why two files per variant, not one

The 44 concept-map columns (15 keyword-presence-in-map indicators + 29
MATLAB complexity metrics) come from a separate, newer "+CM" export
(2026-09-19 OneDrive dump, `ANN EI Prediction Code\Input\All Factors\*+CM.xlsx`)
matched by Participant Code. Its `*-CM.xlsx` siblings in that same dump are
byte-for-byte participant-identical to the UTD-Research canonical tables
above (same row counts, same participant sets), confirming both trees agree
on the base (non-concept-map) cohort. Concept-map digitization was not
completed for every participant, so the "+CM" file only covers a strict
subset:

| Variant | `-CM` rows (== canonical `partial`) | `+CM` source rows | Participant-code mismatches |
|---|---:|---:|---|
| after  | 1789 | 1076 | 1 code (`LL04LO12`) appears twice in `+CM` with conflicting values (different `Year`); both rows dropped as ambiguous -> 1074 usable |
| before | 1929 | 1284 | none |
| diff   | 1012 |  479 | 1 code (`on18na61`) only matched after case-normalizing to `ON18NA61`; recovered, 0 rows dropped -> 479 usable |

Rather than left-joining concept-map columns into the full cohort (which
would make `concept_map` ~35-53% NaN inside a single table), each variant is
split: `<variant>_partial.parquet` keeps the full cohort with no
concept-map columns, and `<variant>_complete.parquet` keeps only the
concept-map-covered rows with all 188/242/193 columns populated.

## Feature families

See `src/ei_model/feature_families.yaml` (committed, column names only) for
the full per-variant column lists; the same mapping applies to both
completeness levels (`concept_map` columns simply aren't present in
`partial` tables). Family sizes (feature columns, excluding `pid`/`ei`):

| Family | after | before | diff |
|---|---:|---:|---:|
| demographics | 14 | 14 | 14 |
| influences | 17 | 43 | 19 |
| experiences (curricular/co-curricular/extra-curricular) | 36 | 37 | 39 |
| self_identification | 5 | 4 | 5 |
| identity_survey (**leakage risk**, see below) | 7 | 7 | 7 |
| career_achievements | 20 | 22 | 20 |
| skills_self_assessment | -- | 26 | -- |
| definitions_ed | 15 | 15 | 15 |
| role_er | 15 | 15 | 15 |
| keywords_skey | 15 | 15 | 15 |
| concept_map (complete tables only) | 44 | 44 | 44 |

`before` has no `experiences`-only difference of note beyond an extra
program/school-choice sub-block folded into `influences`, plus the 26-item
`skills_self_assessment` block that has no analog in `after`/`diff`.

## Target leakage: identity survey items

The 7-column `identity_survey_LEAKAGE_RISK` family (`"I felt strong ties to
other engineers..."`, `"Being an engineer is an important reflection of who
I am"`, `"Improved my engineering performance"`, etc.) are the Recognition /
Interest / Performance items the source paper drafts describe as the three
dimensions the composite `ei` score is built from
(`Using an Artificial Neural Network to Predict Engineering Identity Shifts
in ECS Students.docx`, section 2.1). A naive linear fit of just these 7
columns against `ei` (after/partial, n=1778 complete cases) reaches
**R^2 = 0.35**; adding the 5-column `self_identification` family raises it
to **R^2 = 0.39**. No exact (R^2=1.0) reconstruction of `ei` from any subset
of visible columns was found, so the precise scoring formula is not fully
recoverable from these tables alone, and the drafts do not spell it out
either. Treat `identity_survey_LEAKAGE_RISK` (and, to a lesser extent,
`self_identification`) as excluded-by-default inputs for any model meant to
*predict* `ei` from upstream/antecedent factors rather than from its own
survey components.

## Private, never-committed files (in `EI_DATA_DIR` only)

- `after_partial.parquet`, `after_complete.parquet`, `before_partial.parquet`,
  `before_complete.parquet`, `diff_partial.parquet`, `diff_complete.parquet`
  -- the six modeling tables above.
- `salt.txt` -- the pid-hash salt.
- `pid_mapping.parquet` -- `pid <-> Participant Code` (plus `variant` and
  `completeness`), so concept-map/other metrics can be joined in later by
  code. Never committed.
- `target_ranges.json` -- machine-readable copy of the valid-range table above.
- `build_report.json` -- the row-count/mismatch report summarized above.
