#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub>=0.26",
# ]
# ///
"""Download the ifc-bench dataset from Hugging Face.

Always fetches the questions CSV to data/questions/ifc-bench-v2.csv.
With --with-models it also downloads the IFC models (projects/, several GB)
and links them under data/ifc-bench/projects.

The dataset is public (https://huggingface.co/datasets/sylvainHellin/ifc-bench),
no authentication token is needed.

Usage:
    uv run scripts/download_data.py
    uv run scripts/download_data.py --with-models
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "sylvainHellin/ifc-bench"
REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_CSV = REPO_ROOT / "data" / "questions" / "ifc-bench-v2.csv"
PROJECTS_DIR = REPO_ROOT / "data" / "ifc-bench" / "projects"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--with-models",
        action="store_true",
        help="also download the IFC models (projects/, several GB)",
    )
    args = parser.parse_args()

    patterns = ["questions/ifc-bench-v2.csv"]
    if args.with_models:
        patterns.append("projects/**")

    local = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=patterns,
        )
    )

    QUESTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local / "questions" / "ifc-bench-v2.csv", QUESTIONS_CSV)
    print(f"Questions CSV: {QUESTIONS_CSV}")

    if args.with_models:
        source_projects = local / "projects"
        if PROJECTS_DIR.is_symlink() or PROJECTS_DIR.exists():
            print(f"IFC models already present at {PROJECTS_DIR}, leaving as-is.")
        else:
            PROJECTS_DIR.parent.mkdir(parents=True, exist_ok=True)
            PROJECTS_DIR.symlink_to(source_projects)
            print(f"IFC models linked at {PROJECTS_DIR} -> {source_projects}")
        print("Point the pipelines at the models by setting in your .env:")
        print(f"    IFC_BENCH_DIR={PROJECTS_DIR}")


if __name__ == "__main__":
    main()
