#!/usr/bin/env python3
"""
Batch-convert all IFC models referenced in the IFC-Bench question datasets
to filesystem format using ifc2fs.py.

Output goes to data/conversions/ifc_filesys/{project}/{ifc_model}_fs/ (repo
root) for each unique (project, ifc_model) pair found in the CSV files.

Usage:
    uv run python convert_all.py [--force]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

import pandas as pd

from ifc2fs import convert

QUESTIONS_DIR = _REPO_ROOT / "data" / "questions"
# Source IFC models: $IFC_BENCH_DIR if set (see scripts/download_data.py),
# otherwise data/models/<project>/<ifc_model>.ifc at the repo root.
_bench_dir = os.environ.get("IFC_BENCH_DIR")
if _bench_dir:
    MODELS_DIR = Path(_bench_dir).expanduser()
    if not MODELS_DIR.is_absolute():
        MODELS_DIR = _REPO_ROOT / MODELS_DIR  # relative paths anchor at the repo root
    MODELS_DIR = MODELS_DIR.resolve()
else:
    MODELS_DIR = _REPO_ROOT / "data" / "models"
CONVERSIONS_DIR = _REPO_ROOT / "data" / "conversions" / "ifc_filesys"

CSV_FILES = [
    QUESTIONS_DIR / "ifc-bench-v2.csv",
]


def collect_pairs() -> list[tuple[str, str]]:
    """Collect unique (project, ifc_model) pairs from all CSV files."""
    pairs: set[tuple[str, str]] = set()
    for csv_path in CSV_FILES:
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            pairs.add((str(row["project"]), str(row["ifc_model"])))
    return sorted(pairs)


def main():
    ap = argparse.ArgumentParser(description="Batch-convert IFC models to filesystem format")
    ap.add_argument(
        "--force", action="store_true",
        help="Re-convert even if output directory already exists",
    )
    args = ap.parse_args()

    pairs = collect_pairs()
    print(f"Found {len(pairs)} unique (project, ifc_model) pairs\n")

    converted, skipped, failed = 0, 0, 0

    for project, ifc_model in pairs:
        ifc_path = MODELS_DIR / project / f"{ifc_model}.ifc"
        output_dir = CONVERSIONS_DIR / project / f"{ifc_model}_fs"

        if not ifc_path.exists():
            print(f"  SKIP {project}/{ifc_model}: IFC file not found at {ifc_path}")
            failed += 1
            continue

        if output_dir.exists() and not args.force:
            print(f"  SKIP {project}/{ifc_model}: already converted ({output_dir})")
            skipped += 1
            continue

        if output_dir.exists() and args.force:
            shutil.rmtree(output_dir)

        print(f"{'='*60}")
        print(f"  CONVERTING {project}/{ifc_model}")
        print(f"    input:  {ifc_path}")
        print(f"    output: {output_dir}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            convert(str(ifc_path), str(output_dir))
            elapsed = time.time() - t0
            print(f"  DONE in {elapsed:.1f}s\n")
            converted += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.1f}s: {e}\n", file=sys.stderr)
            failed += 1

    print(f"\n{'='*60}")
    print(f"Summary: {converted} converted, {skipped} skipped, {failed} failed")
    print(f"Total pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
