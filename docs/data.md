# Data card: canonical EI modeling tables

Aggregates only, computed from the real (never-committed) tables. No
per-participant values, free text, or demographics values appear below.

## Tables

Three variants, loaded via `ei_model.data.load_variant("after" | "before" | "diff")`
from `EI_DATA_DIR/<variant>.parquet`. Each row is one participant, keyed by a
salted-hash `pid` (SHA-256 of `salt + Participant Code`; the salt lives only
in `EI_DATA_DIR/salt.txt`, never in git). The composite engineering identity
target is column `ei`.

| Variant | Rows | Feature columns | `ei` mean | `ei` sd | `ei` observed min/max | Valid range |
|---|---:|---:|---:|---:|---:|---:|
| after  | 1789 | 188 | 21.61 | 15.04 | -46.24 / 50.00 | -50..50 |
| before | 1929 | 242 | 20.93 | 13.74 | -31.72 / 50.00 | -50..50 |
| diff   | 1012 | 193 |  1.11 | 13.15 | -55.91 / 68.82 | -100..100 |

`ei` has 0% missingness in all three tables. Valid ranges are stored as
`ei_model.data.VALID_TARGET_RANGES` and reflect the instrument's theoretical
range, not the observed sample (observed values fall inside it in all three
tables).

Source: UTD-Research `Engineering Identitiy/files/` canonical workbooks
(`EIAfter ... Whole.xlsx`, `EIBefore ... Whole_Final Dup Cleaned 1.xlsx`,
`EIAfter ... Diff.xlsx`), each with `Input Vectors`/`Targets` sheets merged
on Participant Code (verified 1:1, zero duplicates, identical participant
sets between sheets before merging -- never assumed from row order).

### Concept-map columns are a left-join, not a full column

44 concept-map columns (15 keyword-presence-in-map indicators + 29 MATLAB
complexity metrics) were left-joined in per participant from a newer,
independently-discovered "+CM" export (2026-09-19 OneDrive dump), matched by
Participant Code. Digitized/coded concept maps only exist for a subset of
participants, so this family has much higher missingness than the rest of
the table:

| Variant | concept_map missing % | other families missing % (typical) |
|---|---:|---:|
| after  | 39.97 | 0.00 (0.16 for identity_survey) |
| before | 33.44 | 0.00-0.01 |
| diff   | 52.77 | 0.00 (0.04 for identity_survey) |

Row counts (1789/1929/1012) were kept at the full non-concept-map cohort
size on purpose -- the "+CM" source files alone only cover 1076/1284/479
rows, a strict subset of the same participants, because concept-map
digitization was not completed for everyone. Overall feature missingness
(all families, all columns) is 9.36% (after), 6.08% (before), 12.03%
(diff), almost entirely attributable to the concept_map family above.

One participant code (`after` variant) appears twice in the "+CM" source
file with conflicting values (different `Year`); both rows were dropped as
ambiguous, so that one participant's concept-map columns are NaN rather
than guessed at.

## Feature families

See `src/ei_model/feature_families.yaml` (committed, column names only) for
the full per-variant column lists. Family sizes (feature columns, excluding
`pid`/`ei`):

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
| concept_map | 44 | 44 | 44 |

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
columns against `ei` (after/whole, n=1778 complete cases) reaches
**R^2 = 0.35**; adding the 5-column `self_identification` family raises it
to **R^2 = 0.39**. No exact (R^2=1.0) reconstruction of `ei` from any subset
of visible columns was found, so the precise scoring formula is not fully
recoverable from these tables alone, and the drafts do not spell it out
either. Treat `identity_survey_LEAKAGE_RISK` (and, to a lesser extent,
`self_identification`) as excluded-by-default inputs for any model meant to
*predict* `ei` from upstream/antecedent factors rather than from its own
survey components.

## Private, never-committed files (in `EI_DATA_DIR` only)

- `after.parquet`, `before.parquet`, `diff.parquet` -- the modeling tables above.
- `salt.txt` -- the pid-hash salt.
- `pid_mapping.parquet` -- `pid <-> Participant Code` (plus `variant`), so
  concept-map/other metrics can be joined in later by code. Never committed.
- `target_ranges.json` -- machine-readable copy of the valid-range table above.
