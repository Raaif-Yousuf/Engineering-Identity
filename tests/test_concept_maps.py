"""Tests for ei_model.concept_maps, exercised only on synthetic workbooks.

Two independent kinds of check back this module (see
docs/concept_map_metrics.md for the full validation report):
  1. Real MATLAB-computed cMet_01..cMet_29 values recovered from the
     owner's data dump matched this port exactly for 28 of 29 metrics
     across every real participant checked (286/290 metric cells); this
     is *not* re-checked here since it requires real, never-committed data.
  2. The small graphs below, whose expected values were independently
     hand-derived (see PR description) from the same algorithm definitions
     this module implements, and cross-checked against this module's
     output before being pinned as test expectations.
"""

import math

import pytest

from ei_model.concept_maps import (
    METRIC_NAMES,
    ConceptMap,
    ConceptMapError,
    compute_metrics,
    compute_metrics_for_workbook,
    parse_workbook,
)
from tests.fixtures.synthetic_concept_maps import write_concept_map_workbook


def test_metric_names_are_29_in_order():
    assert len(METRIC_NAMES) == 29
    assert METRIC_NAMES[0] == "cMet_01"
    assert METRIC_NAMES[-1] == "cMet_29"
    assert list(METRIC_NAMES) == [f"cMet_{i:02d}" for i in range(1, 30)]


class TestParseWorkbook:
    def test_parses_participant_code_and_edges(self, tmp_path):
        path = write_concept_map_workbook(
            tmp_path / "FAKE01.xlsx", "FA01KE01", [("engineer", "problem_solver")]
        )
        cm = parse_workbook(path)
        assert cm.participant_code == "FA01KE01"
        assert cm.rows == ((("engineer",), ("problem_solver",)),)

    def test_trailing_blank_rows_are_dropped(self, tmp_path):
        path = write_concept_map_workbook(
            tmp_path / "FAKE02.xlsx",
            "FA02KE02",
            [("A", "B"), ("B", "C")],
            trailing_blank_rows=5,
        )
        cm = parse_workbook(path)
        assert len(cm.rows) == 2

    def test_falls_back_to_filename_when_no_name_sheet(self, tmp_path):
        path = write_concept_map_workbook(
            tmp_path / "FAKE03.xlsx", "unused", [("A", "B")], include_name_sheet=False
        )
        cm = parse_workbook(path)
        assert cm.participant_code == "FAKE03"

    def test_accepts_singular_source_sink_sheet_names(self, tmp_path):
        path = write_concept_map_workbook(
            tmp_path / "FAKE04.xlsx",
            "FA04KE04",
            [("A", "B")],
            source_sheet_name="Source",
            sink_sheet_name="Sink",
        )
        cm = parse_workbook(path)
        assert cm.rows == ((("A",), ("B",)),)

    def test_missing_sheets_raises(self, tmp_path):
        import openpyxl

        path = tmp_path / "NOTAMAP.xlsx"
        wb = openpyxl.Workbook()
        wb.save(path)

        with pytest.raises(ConceptMapError):
            parse_workbook(path)

    def test_unreadable_file_raises(self, tmp_path):
        path = tmp_path / "CORRUPT.xlsx"
        path.write_text("not actually an xlsx file")

        with pytest.raises(ConceptMapError):
            parse_workbook(path)


class TestComputeMetrics:
    def test_empty_map_raises(self):
        cm = ConceptMap(participant_code="EMPTY", rows=())
        with pytest.raises(ConceptMapError):
            compute_metrics(cm)

    def test_single_edge_hand_derived(self):
        """A -> B. Hand-derived expected values (see module docstring)."""
        cm = ConceptMap(participant_code="X", rows=((("A",), ("B",)),))
        m = compute_metrics(cm)

        expected = {
            "cMet_01": 2.0,  # E
            "cMet_02": 1.0,  # R
            "cMet_03": 1.0,  # DOF
            "cMet_04": 2.0,  # Conn (2 self-loop bookkeeping increments)
            "cMet_05": 1.0,  # TSPL
            "cMet_06": 1.0,  # MSPL
            "cMet_07": 0.5,  # ASPL = 1 / (2^2 - 2)
            "cMet_08": 0.5,  # SPLD = ASPL / R
            "cMet_09": 2.0,  # TFR (direct flow 1 + self-flow(A)=1)
            "cMet_10": 1.0,  # MFR
            "cMet_11": 0.5,  # AFR = 2 / 2^2
            "cMet_12": 0.5,  # FRD = AFR / R
            "cMet_13": 0.0,  # TBC (no 3rd vertex to sit between)
            "cMet_14": 0.0,
            "cMet_15": 0.0,
            "cMet_16": 0.0,
            "cMet_17": 0.0,  # TCC (no triangles possible)
            "cMet_18": 0.0,
            "cMet_19": 0.0,
            "cMet_20": 0.0,
            "cMet_21": 1.0,  # ASA: one relation, one level, stepsize 1
            "cMet_22": 0.0,  # in-degree cores (0 for a 2-node path)
            "cMet_23": 0.0,
            "cMet_24": 0.0,
            "cMet_25": 0.0,
            "cMet_26": 0.0,  # out-degree cores
            "cMet_27": 0.0,
            "cMet_28": 0.0,
            "cMet_29": 0.0,
        }
        assert set(m) == set(METRIC_NAMES)
        for name, value in expected.items():
            assert m[name] == pytest.approx(value), name

    def test_three_cycle_hand_derived(self):
        """A -> B -> C -> A. Hand-derived expected values (see module docstring).

        Every vertex lies on the cycle, so `iterative_spath`'s
        no-self-case quirk gives each vertex a nonzero self-distance
        (the cycle length, 3) -- this is intentional, matching real
        MATLAB output, not a bug in the test.
        """
        cm = ConceptMap(
            participant_code="Y",
            rows=((("A",), ("B",)), (("B",), ("C",)), (("C",), ("A",))),
        )
        m = compute_metrics(cm)

        expected = {
            "cMet_01": 3.0,  # E
            "cMet_02": 3.0,  # R
            "cMet_03": 3.0,  # DOF (3 distinct pairs)
            "cMet_04": 6.0,  # Conn = 2 * R
            "cMet_05": 18.0,  # TSPL (each vertex has distances {1,2,3-self})
            "cMet_06": 3.0,  # MSPL
            "cMet_07": 3.0,  # ASPL = 18 / (9 - 3)
            "cMet_08": 1.0,  # SPLD = ASPL / R
            "cMet_09": 9.0,  # TFR (every pair, including self, has flow 1)
            "cMet_10": 1.0,  # MFR
            "cMet_11": 1.0,  # AFR = 9 / 9
            "cMet_12": 1.0 / 3.0,  # FRD = AFR / R
            "cMet_13": 3.0,  # TBC (each vertex sits on exactly 1 shortest path)
            "cMet_14": 1.0,  # MBC
            "cMet_15": 1.0,  # ABC
            "cMet_16": 1.0 / 3.0,  # BCD
            "cMet_17": 1.5,  # TCC (Fagiolo directed CC = 0.5 per vertex)
            "cMet_18": 0.5,  # MCC
            "cMet_19": 0.5,  # ACC
            "cMet_20": 1.0 / 6.0,  # CCD
            "cMet_21": 4.0,  # ASA (see docs/concept_map_metrics.md derivation)
            "cMet_22": 3.0,  # TCNi (each vertex has in-degree core 1)
            "cMet_23": 1.0,  # MCNi
            "cMet_24": 1.0,  # ACNi
            "cMet_25": 1.0 / 3.0,  # CNDi
            "cMet_26": 3.0,  # TCNo
            "cMet_27": 1.0,  # MCNo
            "cMet_28": 1.0,  # ACNo
            "cMet_29": 1.0 / 3.0,  # CNDo
        }
        for name, value in expected.items():
            assert m[name] == pytest.approx(value), name

    def test_larger_random_map_does_not_crash_and_stays_finite(self):
        """A denser synthetic map should run end to end and yield finite metrics."""
        import random

        rng = random.Random(7)
        labels = [f"N{i}" for i in range(12)]
        edges = []
        for _ in range(25):
            s, t = rng.sample(labels, 2)
            edges.append((s, t))
        cm = ConceptMap(participant_code="BIG", rows=tuple(((s,), (t,)) for s, t in edges))
        m = compute_metrics(cm)
        assert len(m) == 29
        for name, value in m.items():
            assert math.isfinite(value), name
            assert value >= 0.0, name


class TestComputeMetricsForWorkbook:
    def test_end_to_end_wrapper(self, tmp_path):
        path = write_concept_map_workbook(
            tmp_path / "FAKE05.xlsx", "FA05KE05", [("A", "B"), ("B", "C"), ("C", "A")]
        )
        code, metrics = compute_metrics_for_workbook(path)
        assert code == "FA05KE05"
        assert metrics["cMet_01"] == 3.0
