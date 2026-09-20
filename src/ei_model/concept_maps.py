"""Concept-map complexity metrics (cMet_01..cMet_29).

Faithful Python (networkx/numpy) port of the MATLAB toolbox found in the
owner's UGR SP25 data dump, the lab's MATLAB archive, specifically:

  - `generate_complexity.m` / `compag.m`: reads a per-participant concept-map
    workbook (a directed "assembly" of Source -> Sink relations between
    named concept nodes) and produces the 29-value `cMet_01..cMet_29`
    complexity vector confirmed as the ANN model's concept-map input in
    `ANN_Prediction_v3.m`.
  - `map_degrees.m`: builds the weighted degree-of-freedom / adjacency
    matrices from the relation list.
  - `iterative_spath.m`: a level-by-level BFS. It has a quirk this port
    reproduces exactly: because it never special-cases the start vertex,
    a directed cycle back to vertex i gives spath[i, i] a nonzero
    "self distance" instead of 0.
  - `max_flow.m` (self-flow, i==j): real MATLAB self-flow values always
    equal vertex i's total out-capacity, regardless of whether that
    capacity can be routed back to i -- consistent with a push-relabel
    max-flow whose preflow-saturation step already "arrives" at the sink
    because sink == source (see docs/concept_map_metrics.md).
  - `libmbgl/statistics.cc` (`directed_clustering_coefficients`): the
    Fagiolo (2007) directed clustering coefficient, which is also what
    `networkx.clustering` implements for directed graphs.
  - `libmbgl/yasmic/boost_mod/core_numbers.hpp`: the Batagelj-Zaversnik
    O(m) in-degree core-number peeling algorithm.
  - `ASAcomp.m`: the Ameri-Summers connective-complexity score. This port
    matches most (but not all) real MATLAB outputs -- see
    `ASA_APPROXIMATE` below and docs/concept_map_metrics.md.

Workbook format (confirmed against real participant files): a `Sources`
(or `Source`) sheet and a `Sinks` (or `Sink`) sheet, each a single column
of concept-node labels, row-aligned so that row i is one directed relation
`Sources[i] -> Sinks[i]`; trailing blank rows are padding. A `Name` sheet
holds the participant code in cell A1 (falls back to the file stem).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import openpyxl

METRIC_NAMES: tuple[str, ...] = tuple(f"cMet_{i:02d}" for i in range(1, 30))

# ASAcomp.m (Ameri-Summers connective complexity) contains index-arithmetic
# tie-breaking logic (a MATLAB linear-index dedup loop over a shrinking
# matrix) that could not be reproduced bit-for-bit from static reading alone
# without a MATLAB/Octave install. This port implements the most defensible
# reading of the algorithm's intent and reproduces real MATLAB-computed
# cMet_21 values exactly for maps with tree/DAG-like structure. It
# under-counts on maps containing many reciprocal (bidirectional) edge
# pairs (a known, documented gap -- see docs/concept_map_metrics.md for the
# measured match rate).
ASA_APPROXIMATE = True


class ConceptMapError(ValueError):
    """Raised when a workbook cannot be parsed as a concept map."""


@dataclass(frozen=True)
class ConceptMap:
    """A parsed concept-map workbook: a directed multigraph of relations."""

    participant_code: str
    rows: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    source_path: str | None = None


def _read_label_rows(ws) -> list[tuple[str, ...]]:
    """Read a Sources/Sinks worksheet into one tuple of labels per row.

    Mirrors `parseTextdata.m`: within a row, values are read left to right
    until the first blank cell (supporting multi-column hyperedge rows,
    though real concept-map workbooks observed so far use a single column).
    """
    out = []
    for row in ws.iter_rows(values_only=True):
        cells: list[str] = []
        for v in row:
            if v is None or (isinstance(v, str) and v.strip() == ""):
                break
            cells.append(str(v).strip())
        out.append(tuple(cells))
    return out


def parse_workbook(path: Path | str) -> ConceptMap:
    """Parse a concept-map Excel workbook into a `ConceptMap`.

    Raises `ConceptMapError` for anything that isn't a readable workbook
    with both a Sources/Source and a Sinks/Sink sheet.
    """
    path = Path(path)
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as exc:  # noqa: BLE001 - re-raise as a domain error
        raise ConceptMapError(f"{path}: not a readable Excel workbook ({exc})") from exc

    sheets = set(wb.sheetnames)
    src_name = "Sources" if "Sources" in sheets else ("Source" if "Source" in sheets else None)
    snk_name = "Sinks" if "Sinks" in sheets else ("Sink" if "Sink" in sheets else None)
    if src_name is None or snk_name is None:
        raise ConceptMapError(f"{path}: missing a Sources/Source and/or Sinks/Sink sheet")

    sources = _read_label_rows(wb[src_name])
    sinks = _read_label_rows(wb[snk_name])
    n = max(len(sources), len(sinks))
    rows: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for i in range(n):
        s = sources[i] if i < len(sources) else ()
        t = sinks[i] if i < len(sinks) else ()
        if not s and not t:
            continue
        rows.append((s, t))

    code = None
    if "Name" in sheets:
        try:
            first_row = next(iter(wb["Name"].iter_rows(values_only=True)), None)
        except StopIteration:
            first_row = None
        if first_row and first_row[0] is not None:
            code = str(first_row[0]).strip()
    if not code:
        code = path.stem

    return ConceptMap(participant_code=code, rows=tuple(rows), source_path=str(path))


def _map_degrees(
    rows: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
) -> tuple[list[str], np.ndarray, float, int]:
    """Port of `condition_assembly.m` + `map_degrees.m`.

    Returns (element_list, DOF_mat, DOF, R). `DOF_mat[i, j]` is the number
    of relations connecting element i (as a source) to element j (as a
    sink); its diagonal holds each element's self-loop count (an element
    appearing as a source in a relation with no matching sink, or vice
    versa).
    """
    elements: list[str] = []
    seen: set[str] = set()
    for s, t in rows:
        for lab in (*s, *t):
            if lab not in seen:
                seen.add(lab)
                elements.append(lab)
    idx = {lab: i for i, lab in enumerate(elements)}
    e = len(elements)
    r = len(rows)

    dof_mat = np.zeros((e, e))
    para_mat = np.zeros((e, e))
    for s, t in rows:
        s_ids = [idx[x] for x in s]
        t_ids = [idx[x] for x in t]
        rel_mat = np.zeros((e, e))
        for si in s_ids:
            if si not in t_ids:
                rel_mat[si, si] += 1
            for ti in t_ids:
                rel_mat[si, ti] += 1
        for ti in t_ids:
            if ti not in s_ids:
                rel_mat[ti, ti] += 1
        dof_mat += rel_mat
        para_mat += (np.triu(rel_mat + rel_mat.T, 1) > 0).astype(float)

    dof = float(para_mat.sum())
    return elements, dof_mat, dof, r


def _iterative_spath(dsm: np.ndarray) -> np.ndarray:
    """Literal port of `iterative_spath.m`'s level-by-level BFS.

    Deliberately does not special-case the start vertex: if a directed
    cycle leads back to vertex i, `spath[i, i]` ends up nonzero (the cycle
    length), matching the MATLAB behavior and real MATLAB-computed
    cMet_05/06/07/08 values exactly.
    """
    e = dsm.shape[0]
    succ = [set(np.nonzero(dsm[i])[0].tolist()) for i in range(e)]
    spath_mat = np.zeros((e, e))
    for i in range(e):
        level = set(succ[i])
        found: set[int] = set()
        level_number = 0
        spath = np.zeros(e)
        while level:
            level_number += 1
            found |= level
            for v in level:
                spath[v] = level_number
            level_next: set[int] = set()
            for v in level:
                level_next |= succ[v]
            level_next -= found
            level = level_next
        spath_mat[i, :] = spath
    return spath_mat


def _self_flow(nodiag: np.ndarray, v: int) -> float:
    """Max-flow from v to v, matching `max_flow(A, i, i)` in this MATLAB
    toolbox's BGL wrapper.

    A standard max-flow library has no real notion of source == sink; the
    observed MATLAB self-flow values are always exactly v's total
    out-capacity (the row sum of the capacity matrix, ignoring whether
    that capacity can actually be routed back to v), consistent with a
    push-relabel implementation whose preflow-saturation step already
    "arrives" at the sink because sink == source, before any augmenting
    work happens. Validated exactly against real MATLAB output on both
    tree-shaped and cyclic maps (see docs/concept_map_metrics.md).
    """
    return float(nodiag[v, :].sum())


def _in_degree_core_numbers(adj: np.ndarray) -> list[float]:
    """Batagelj-Zaversnik in-degree core-number peeling (core_numbers.hpp).

    `adj[i, j] == 1` means a directed edge i -> j. Returns the in-degree
    core number of each vertex (call with `adj.T` for out-degree cores, as
    the MATLAB docstring for `core_numbers` instructs).
    """
    e = adj.shape[0]
    indeg = adj.sum(axis=0).astype(int).tolist()
    succ = [set(np.nonzero(adj[i])[0].tolist()) for i in range(e)]
    deg = list(indeg)
    core = [0.0] * e
    removed = [False] * e
    for _ in range(e):
        v = min((i for i in range(e) if not removed[i]), key=lambda i: deg[i])
        core[v] = float(deg[v])
        removed[v] = True
        for w in succ[v]:
            if not removed[w] and deg[w] > deg[v]:
                deg[w] -= 1
    return core


def _asa_score(rows: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]) -> float:
    """Port of `ASAcomp.m` (Ameri-Summers connective complexity).

    At each level, `ASAcomp.m` finds the elements with the current minimum
    "connection frequency" (stepsize) and removes every relation touching
    any of them, in element-processing order; MATLAB's linear-index dedup
    loop means a relation already claimed by an earlier element in that
    order is not claimed again by a later one. This port reproduces that
    element-processing-order, first-claim-wins behavior, using first
    appearance order (matching MATLAB's `unique(..., 'first')` element
    numbering) to break ties among elements sharing the current stepsize.

    See `ASA_APPROXIMATE` docstring above for the known fidelity gap on
    maps with many reciprocal (bidirectional) edge pairs.
    """
    order: list[str] = []
    seen: set[str] = set()
    for s, t in rows:
        for lab in (*s, *t):
            if lab not in seen:
                seen.add(lab)
                order.append(lab)
    order_index = {lab: i for i, lab in enumerate(order)}

    active_rows: list[tuple[str, ...]] = []
    for s, t in rows:
        elems = tuple(sorted(set(s) | set(t), key=lambda x: order_index[x]))
        if elems:
            active_rows.append(elems)

    score = 0
    level = 1
    while active_rows:
        freq: Counter[str] = Counter()
        for r in active_rows:
            freq.update(r)
        if not freq:
            break
        stepsize = min(freq.values())
        target = sorted(
            (e for e, c in freq.items() if c == stepsize), key=lambda x: order_index[x]
        )

        # rems = number of elements (in first-appearance order) whose
        # relation-group is not already fully claimed by an earlier
        # element's group at this stepsize.
        used_rows: set[int] = set()
        rems = 0
        for e in target:
            group = [i for i, r in enumerate(active_rows) if e in r]
            if any(i in used_rows for i in group):
                continue
            rems += 1
            used_rows.update(group)

        score += level * stepsize * rems
        active_rows = [r for i, r in enumerate(active_rows) if i not in used_rows]
        level += 1
    return float(score)


def compute_metrics(cm: ConceptMap) -> dict[str, float]:
    """Compute the 29 concept-map complexity metrics for a parsed workbook.

    Raises `ConceptMapError` if the map has no valid relations.
    """
    elements, dof_mat, dof, r = _map_degrees(cm.rows)
    e = len(elements)
    if e == 0 or r == 0:
        raise ConceptMapError(
            f"{cm.source_path or cm.participant_code}: no valid Source->Sink relations"
        )

    m: dict[str, float] = {}
    m["cMet_01"] = float(e)
    m["cMet_02"] = float(r)
    m["cMet_03"] = dof
    m["cMet_04"] = float(np.trace(dof_mat))

    nodiag = dof_mat.copy()
    np.fill_diagonal(nodiag, 0)
    dsm = (nodiag > 0).astype(int)

    graph = nx.DiGraph()
    graph.add_nodes_from(range(e))
    for i in range(e):
        for j in range(e):
            if dsm[i, j]:
                graph.add_edge(i, j)

    spath = _iterative_spath(dsm)
    tspl = float(spath.sum())
    mspl = float(spath.max())
    denom = e * e - e
    aspl = tspl / denom if denom else 0.0
    m["cMet_05"], m["cMet_06"], m["cMet_07"], m["cMet_08"] = tspl, mspl, aspl, aspl / r

    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from(range(e))
    for i in range(e):
        for j in range(e):
            if nodiag[i, j] > 0:
                flow_graph.add_edge(i, j, capacity=float(nodiag[i, j]))
    fr = np.zeros((e, e))
    for i in range(e):
        for j in range(e):
            if i == j:
                fr[i, j] = _self_flow(nodiag, i)
            elif flow_graph.has_node(i) and flow_graph.has_node(j):
                try:
                    fr[i, j] = nx.maximum_flow_value(flow_graph, i, j, capacity="capacity")
                except nx.NetworkXError:
                    fr[i, j] = 0.0
    tfr, mfr, afr = float(fr.sum()), float(fr.max()), float(fr.mean())
    m["cMet_09"], m["cMet_10"], m["cMet_11"], m["cMet_12"] = tfr, mfr, afr, afr / r

    bc = nx.betweenness_centrality(graph, normalized=False)
    bc_vec = np.array([bc[i] for i in range(e)])
    tbc, mbc, abc = float(bc_vec.sum()), float(bc_vec.max()), float(bc_vec.mean())
    m["cMet_13"], m["cMet_14"], m["cMet_15"], m["cMet_16"] = tbc, mbc, abc, abc / r

    cc = nx.clustering(graph)
    cc_vec = np.array([cc[i] for i in range(e)])
    tcc, mcc, acc = float(cc_vec.sum()), float(cc_vec.max()), float(cc_vec.mean())
    m["cMet_17"], m["cMet_18"], m["cMet_19"], m["cMet_20"] = tcc, mcc, acc, acc / r

    m["cMet_21"] = _asa_score(cm.rows)

    cni = np.array(_in_degree_core_numbers(dsm))
    cno = np.array(_in_degree_core_numbers(dsm.T))
    tcni, mcni, acni = float(cni.sum()), float(cni.max()), float(cni.mean())
    tcno, mcno, acno = float(cno.sum()), float(cno.max()), float(cno.mean())
    m["cMet_22"], m["cMet_23"], m["cMet_24"], m["cMet_25"] = tcni, mcni, acni, acni / r
    m["cMet_26"], m["cMet_27"], m["cMet_28"], m["cMet_29"] = tcno, mcno, acno, acno / r

    return m


def compute_metrics_for_workbook(path: Path | str) -> tuple[str, dict[str, float]]:
    """Convenience wrapper: parse a workbook and compute its 29 metrics.

    Returns (participant_code, metrics).
    """
    cm = parse_workbook(path)
    return cm.participant_code, compute_metrics(cm)
