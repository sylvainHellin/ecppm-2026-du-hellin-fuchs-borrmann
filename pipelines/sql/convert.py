"""Convert IFC files to SQLite databases using ifcpatch Ifc2Sql.

Iterates over projects in IFC_BENCH_DIR, converts each .ifc file to a SQLite
database in data/conversions/sql/<project>/<model>.sqlite. Existing files are
skipped unless --force is passed.

Usage:
    uv run python convert.py                       # convert all projects
    uv run python convert.py --project dental_clinic
    uv run python convert.py --force               # re-convert even if .sqlite exists
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import ifcopenshell
import ifcpatch
from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_OUTPUT_DIR = _REPO_ROOT / "data" / "conversions" / "sql"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert IFC files to SQLite via Ifc2Sql.")
    parser.add_argument("--project", type=str, default=None, help="Convert only this project.")
    parser.add_argument("--force", action="store_true", help="Re-convert even if .sqlite already exists.")
    return parser.parse_args()


def _find_ifc_files(bench_dir: Path, project_filter: str | None) -> list[tuple[str, Path]]:
    """Return (project_name, ifc_path) pairs from the bench directory."""
    pairs: list[tuple[str, Path]] = []
    for project_dir in sorted(bench_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        if project_filter is not None and project_dir.name != project_filter:
            continue
        for ifc_file in sorted(project_dir.glob("*.ifc")):
            pairs.append((project_dir.name, ifc_file))
    return pairs


def convert_one(ifc_path: Path, sqlite_path: Path, *, should_get_psets: bool = True) -> None:
    """Convert a single IFC file to SQLite."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    model = ifcopenshell.open(str(ifc_path))

    # ifcpatch.execute unpacks arguments positionally after (file, logger):
    #   sql_type, host, username, password, database,
    #   full_schema, is_strict, should_expand, should_get_inverses,
    #   should_get_psets, should_get_geometry, should_skip_geometry_data
    result_path = ifcpatch.execute({
        "input": str(ifc_path),
        "file": model,
        "recipe": "Ifc2Sql",
        "arguments": [
            "sqlite",           # sql_type
            "localhost",        # host (unused for sqlite)
            "root",             # username (unused for sqlite)
            "pass",             # password (unused for sqlite)
            str(sqlite_path),   # database path
            False,              # full_schema -- only tables for classes present in the model
            False,              # is_strict
            False,              # should_expand
            False,              # should_get_inverses -- skip to keep DB lean
            should_get_psets,   # should_get_psets
            False,              # should_get_geometry -- no geometry needed for Q&A
            False,              # should_skip_geometry_data
        ],
    })

    # ifcpatch may append .sqlite if not present; move to our desired path
    if result_path and Path(result_path).resolve() != sqlite_path.resolve():
        shutil.move(result_path, sqlite_path)


def main() -> None:
    load_dotenv(_REPO_ROOT / ".env")
    args = _parse_args()

    bench_dir_env = os.environ.get("IFC_BENCH_DIR")
    if not bench_dir_env:
        sys.exit("error: IFC_BENCH_DIR env var not set (path to ifc-bench projects/ dir).")
    bench_dir = Path(bench_dir_env).expanduser()
    if not bench_dir.is_absolute():
        bench_dir = _REPO_ROOT / bench_dir  # relative paths anchor at the repo root
    bench_dir = bench_dir.resolve()
    if not bench_dir.is_dir():
        sys.exit(f"error: IFC_BENCH_DIR does not exist: {bench_dir}")

    pairs = _find_ifc_files(bench_dir, args.project)
    if not pairs:
        sys.exit("error: no IFC files found matching the given filters.")

    print(f"> found {len(pairs)} IFC file(s) to convert")
    print(f"> output: {_OUTPUT_DIR}\n")

    converted = 0
    skipped = 0
    errors = 0

    for project, ifc_path in pairs:
        stem = ifc_path.stem
        sqlite_path = _OUTPUT_DIR / project / f"{stem}.sqlite"

        if sqlite_path.exists() and not args.force:
            print(f"  SKIP {project}/{stem} (exists)")
            skipped += 1
            continue
        # Remove existing file before re-converting to avoid ifcpatch reuse issues
        if sqlite_path.exists():
            sqlite_path.unlink()

        print(f"  CONVERT {project}/{stem} ...", end=" ", flush=True)
        t0 = time.monotonic()
        try:
            convert_one(ifc_path, sqlite_path)
            elapsed = time.monotonic() - t0
            print(f"ok ({elapsed:.1f}s)")
            converted += 1
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"FAILED ({elapsed:.1f}s): {exc}")
            # Remove partial/broken .sqlite so it does not fool run.py
            if sqlite_path.exists():
                sqlite_path.unlink()
            # Retry without psets -- recovers entity + relationship tables
            print(f"  RETRY  {project}/{stem} (no psets) ...", end=" ", flush=True)
            t1 = time.monotonic()
            try:
                convert_one(ifc_path, sqlite_path, should_get_psets=False)
                elapsed2 = time.monotonic() - t1
                print(f"ok ({elapsed2:.1f}s, no psets)")
                converted += 1
            except Exception as exc2:
                elapsed2 = time.monotonic() - t1
                print(f"FAILED ({elapsed2:.1f}s): {exc2}")
                if sqlite_path.exists():
                    sqlite_path.unlink()
                errors += 1

    print(f"\ndone -- converted: {converted}, skipped: {skipped}, errors: {errors}")


if __name__ == "__main__":
    main()
