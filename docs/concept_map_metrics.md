# Concept-map complexity metrics (cMet_01..cMet_29)

`src/ei_model/concept_maps.py` computes the 29 concept-map complexity
metrics used as the ANN model's concept-map input feature family (`cm_*` /
`concept_map` in `docs/data.md`), ported from the owner's original MATLAB
toolbox rather than re-derived from scratch.

## Source

Found in the lab's MATLAB archive (UGR SP25 data dump; not present in this
repo). `ANN_Prediction_v3.m` confirms `cMet_01..cMet_29` as the model's
concept-map input vector name. The call chain that produces it from a
concept-map workbook:

- `generate_complexity.m` reads a folder of workbooks, calls `importxls.m`
  (which requires a `Sources`/`Source`, a `Sinks`/`Sink`, and reads the
  participant code from a `Name` sheet) and `compag.m` per workbook.
- `compag.m` / `condition_assembly.m` / `map_degrees.m` build a directed,
  weighted "assembly" graph from the Source -> Sink relation list and
  compute all 29 values, in this order:

| # | Metric | Definition |
|---|---|---|
| 1 | E | number of distinct concept nodes |
| 2 | R | number of relations (Source -> Sink rows) |
| 3 | DOF | count of distinct node-pairs connected by at least one relation |
| 4 | Conn | sum of the per-node self-loop bookkeeping counts (`trace` of the raw degree matrix; not a real self-loop, an artifact of the relation-bookkeeping the toolbox inherited from its assembly-time-prediction origin) |
| 5-8 | TSPL, MSPL, ASPL, SPLD | total/max/average shortest-path length and average-over-R, from a level-by-level BFS (`iterative_spath.m`) |
| 9-12 | TFR, MFR, AFR, FRD | total/max/average max-flow (all ordered node pairs, including i==i) and average-over-R |
| 13-16 | TBC, MBC, ABC, BCD | total/max/average unweighted directed betweenness centrality and average-over-R |
| 17-20 | TCC, MCC, ACC, CCD | total/max/average directed clustering coefficient (Fagiolo 2007) and average-over-R |
| 21 | ASA | Ameri-Summers connective-complexity score (`ASAcomp.m`) |
| 22-25 | TCNi, MCNi, ACNi, CNDi | total/max/average in-degree core number (Batagelj-Zaversnik) and average-over-R |
| 26-29 | TCNo, MCNo, ACNo, CNDo | same, out-degree core number |

`libmbgl/statistics.cc` (the toolbox's bundled BGL mex C++ source, also in
the zip) confirmed the exact directed clustering-coefficient formula
(Fagiolo 2007: cycle / middleman / in / out triangle motifs), and
`libmbgl/yasmic/boost_mod/core_numbers.hpp` confirmed the in-degree
core-number peeling algorithm (Batagelj & Zaversnik, 2002).

## Validation method

No MATLAB or Octave installation was available in this environment. The
port was validated two ways, both **against real MATLAB-computed
output**, not just hand-derived cases:

1. **Real MATLAB output recovered from the data dump.** One class-level
   aggregate output workbook (the toolbox's own `format_complexity_output`
   export: one header row of participant codes, 29 metric rows) was found
   alongside the raw per-student concept-map workbooks it was computed
   from, letting every one of its students' 29 metrics be checked against
   this port's output, cell by cell, on a real graph structure (never
   reproduced here -- see `data/` policy below).

   Result across every student in that file: **286 of 290 metric cells
   matched exactly** (to floating-point precision). All 28 metrics other
   than `cMet_21` (ASA) matched exactly for every student. `cMet_21`
   matched for 6 of 10 students.

   Two behaviors were confirmed this way that are easy to get wrong by
   reading the MATLAB alone (both are now covered by hand-derived unit
   tests too):
   - `iterative_spath.m` never special-cases the start vertex, so a
     directed cycle back to vertex `i` gives it a nonzero *self*-distance
     (the cycle length) instead of 0.
   - `max_flow(A, i, i)` (used for the diagonal of the flow matrix) is not
     a real graph max-flow; it always equals vertex `i`'s total
     out-capacity, regardless of whether that capacity can be routed back
     to `i`, which fits a push-relabel implementation whose
     preflow-saturation step already "arrives" at the sink because
     sink == source.

2. **Hand-derived small graphs** (a single edge, and a 3-node directed
   cycle) worked through by hand against the same algorithm definitions,
   pinned as regression tests in `tests/test_concept_maps.py`.

### Known gap: `cMet_21` (ASA)

`ASAcomp.m` contains a MATLAB linear-index dedup loop over a shrinking
matrix (removing "already claimed" relations as connectivity levels are
peeled off) that could not be reproduced bit-for-bit from static reading
alone. This port implements the most defensible reading of the
algorithm's intent -- at each level, elements at the current minimum
connection frequency claim their relations in first-appearance order, and
a relation already claimed by an earlier element is not claimed again --
which reproduces real MATLAB `cMet_21` values exactly for maps with
tree/DAG-like structure, and under-counts on maps with many reciprocal
(bidirectional) edge pairs. `ei_model.concept_maps.ASA_APPROXIMATE` flags
this in code. If a working MATLAB/Octave install becomes available, this
is the one metric worth re-deriving directly from `ASAcomp.m`'s execution
trace rather than from output values alone.

## Data policy

No real concept-map content, participant codes, or per-participant metric
values appear in this repository -- only the aggregate match-rate numbers
above. Tests use synthetic workbooks generated by
`tests/fixtures/synthetic_concept_maps.py`. `scripts/build_concept_map_metrics.py`
is a generic, path-agnostic CLI meant to be pointed at the owner's local,
never-committed UGR data dump; it is not run by CI and takes no
credentials or embedded paths.
