#!/usr/bin/env python3
"""Batch-run `ei_model.concept_maps` over real concept-map workbooks.

This script is generic and path-agnostic on purpose: it takes the roots to
scan as command-line arguments, so no real data-directory paths are ever
committed to this repository. It is meant to be run by hand (or by a
private, uncommitted wrapper) against the owner's local UGR data dump.

Usage:
    python scripts/build_concept_map_metrics.py \\
        --root lake=/path/to/_Lake \\
        --root onedrive1=/path/to/OneDrive_1_2-9-2025 \\
        --output /path/to/Engineering-Identity-data/concept_map_metrics.parquet

For each `--root name=path`, every `*.xlsx` under `path` is attempted as a
concept-map workbook. Output is one row per (participant_code, distinct map
content); exact re-encounters of the same participant + identical
Source/Sink content (duplicate copies across overlapping OneDrive exports)
are deduplicated to a single row. A participant with genuinely different
map content across roots/folders (e.g. a before/after pair) keeps one row
per distinct map, distinguished by `map_index` and `source_relpath`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ei_model.concept_maps import (  # noqa: E402
    METRIC_NAMES,
    ConceptMapError,
    compute_metrics,
    parse_workbook,
)


@dataclass
class ParseFailure:
    root_name: str
    relpath: str
    reason: str


def _content_hash(cm) -> str:
    payload = repr(cm.rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scan_roots(roots: dict[str, Path]) -> tuple[list[dict], list[ParseFailure]]:
    """Walk every root, parse+score every .xlsx, dedupe exact repeats.

    Returns (records, failures). `records` is a list of row-dicts ready for
    a DataFrame; `failures` records every file that could not be parsed or
    scored, with a human-readable reason, for the coverage report.
    """
    # code -> content_hash -> first record seen (for dedup)
    seen: dict[str, dict[str, dict]] = defaultdict(dict)
    failures: list[ParseFailure] = []

    for root_name, root_path in roots.items():
        if not root_path.is_dir():
            failures.append(ParseFailure(root_name, str(root_path), "root does not exist"))
            continue
        for xlsx_path in sorted(root_path.rglob("*.xlsx")):
            if xlsx_path.name.startswith("~$"):
                continue  # Excel lock file
            relpath = str(xlsx_path.relative_to(root_path))
            try:
                cm = parse_workbook(xlsx_path)
            except ConceptMapError as exc:
                failures.append(ParseFailure(root_name, relpath, f"parse error: {exc}"))
                continue
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    ParseFailure(root_name, relpath, f"unexpected error: {exc!r}")
                )
                continue

            try:
                metrics = compute_metrics(cm)
            except ConceptMapError as exc:
                failures.append(ParseFailure(root_name, relpath, f"metrics error: {exc}"))
                continue
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    ParseFailure(
                        root_name,
                        relpath,
                        f"unexpected metrics error: {exc!r}\n{traceback.format_exc()}",
                    )
                )
                continue

            code = cm.participant_code
            chash = _content_hash(cm)
            bucket = seen[code]
            if chash in bucket:
                bucket[chash]["duplicate_count"] += 1
                bucket[chash]["duplicate_sources"].append(f"{root_name}/{relpath}")
                continue
            bucket[chash] = {
                "participant_code": code,
                "source_root": root_name,
                "source_relpath": relpath,
                "n_relations": len(cm.rows),
                "duplicate_count": 1,
                "duplicate_sources": [],
                **metrics,
            }

    records: list[dict] = []
    for _code, bucket in seen.items():
        for map_index, (_chash, rec) in enumerate(
            sorted(bucket.items(), key=lambda kv: kv[1]["source_relpath"])
        ):
            rec = dict(rec)
            rec["map_index"] = map_index
            rec["duplicate_sources"] = ";".join(rec["duplicate_sources"])
            records.append(rec)
    return records, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="A labeled root directory to scan recursively for *.xlsx. Repeatable.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output .parquet path")
    parser.add_argument(
        "--failures-log",
        type=Path,
        default=None,
        help="Optional path to write the full parse-failure log (one line each).",
    )
    args = parser.parse_args()

    roots: dict[str, Path] = {}
    for spec in args.root:
        if "=" not in spec:
            parser.error(f"--root must be NAME=PATH, got {spec!r}")
        name, _, path = spec.partition("=")
        roots[name] = Path(path)

    records, failures = scan_roots(roots)

    columns = [
        "participant_code",
        "map_index",
        "source_root",
        "source_relpath",
        "n_relations",
        "duplicate_count",
        "duplicate_sources",
        *METRIC_NAMES,
    ]
    df = pd.DataFrame.from_records(records, columns=columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    unique_participants = df["participant_code"].nunique()
    total_maps = len(df)
    total_dupes_skipped = int(df["duplicate_count"].sum()) - total_maps if total_maps else 0

    print(f"Wrote {total_maps} map row(s) for {unique_participants} unique participant(s)")
    print(f"  to {args.output}")
    print(f"Duplicate workbook copies skipped (identical content): {total_dupes_skipped}")
    print(f"Parse/score failures: {len(failures)}")
    reasons: dict[str, int] = defaultdict(int)
    for f in failures:
        head = f.reason.split("\n", 1)[0]
        reasons[head.split(":", 1)[0]] += 1
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")

    if args.failures_log:
        with args.failures_log.open("w", encoding="utf-8") as fh:
            for f in failures:
                fh.write(f"[{f.root_name}] {f.relpath}: {f.reason}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
