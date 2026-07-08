#!/usr/bin/env python3
"""Re-judge error rows in a judged run directory.

Reads <run>/judged.csv (produced by shared/eval/evaluate.py), finds rows where
classification == "error", re-runs only the BAML judge (skipping the agent),
and updates judged.csv and judged_summary.json in place.

Rows whose predicted answer is empty (agent errors) are skipped -- re-judging
cannot fix those; re-run the agent with the pipeline's run.py --resume instead.

Run it with the shared/eval uv environment so baml_client resolves:

Usage:
    uv run --project shared/eval python scripts/rejudge_errors.py --run results/filesystem/<run>
    uv run --project shared/eval python scripts/rejudge_errors.py --run results/cypher/<run> --judge gemini
    uv run --project shared/eval python scripts/rejudge_errors.py --run results/ifc/<run> --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent

# shared/eval/evaluate.py is a script, not an installed package -- import it by path.
sys.path.insert(0, str(_REPO_ROOT / "shared" / "eval"))
from evaluate import (  # noqa: E402
    JUDGE_CHOICES,
    calculate_metrics,
    judge_single,
    print_report,
)


def main() -> int:
    load_dotenv(_REPO_ROOT / ".env")

    ap = argparse.ArgumentParser(description="Re-judge error rows in a judged run")
    ap.add_argument("--run", required=True, type=Path,
                    help="Run directory containing judged.csv")
    ap.add_argument("--judge", default="minimax", choices=JUDGE_CHOICES,
                    help="BAML judge backend (default: minimax)")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="Max retries per judge call (default: 5)")
    ap.add_argument("--retry-delay", type=float, default=60.0,
                    help="Delay between retries in seconds (default: 60)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be re-judged without calling the judge")
    args = ap.parse_args()

    run_dir: Path = args.run.expanduser().resolve()
    judged_csv = run_dir / "judged.csv"
    summary_json = run_dir / "judged_summary.json"
    if not judged_csv.is_file():
        sys.exit(f"error: judged.csv not found in {run_dir} (run shared/eval/evaluate.py first)")

    df = pd.read_csv(judged_csv)
    if "classification" not in df.columns:
        sys.exit("error: judged.csv has no 'classification' column -- is this a judged run?")

    error_mask = df["classification"] == "error"
    error_indices = list(df.index[error_mask])

    print(f"Loaded {len(df)} rows from {judged_csv}")
    print(f"Found {len(error_indices)} error rows to re-judge")

    if not error_indices:
        print("Nothing to do.")
        return 0

    if "project" in df.columns:
        by_project = df.loc[error_indices, "project"].astype(str).value_counts()
        print("Error distribution by project:")
        for p, cnt in sorted(by_project.items()):
            print(f"  {p}: {cnt}")
        print()

    if args.dry_run:
        print("Dry run -- exiting without changes.")
        return 0

    # Backup originals
    shutil.copy2(judged_csv, judged_csv.with_suffix(".csv.bak"))
    print(f"Backup saved to {judged_csv.name}.bak")
    if summary_json.exists():
        shutil.copy2(summary_json, summary_json.with_suffix(".json.bak"))

    succeeded = 0
    skipped = 0
    failed = 0

    for seq, idx in enumerate(error_indices):
        row = df.loc[idx]
        question = str(row["question"])
        predicted = row.get("predicted", "")

        print(f"[{seq + 1}/{len(error_indices)}] "
              f"{row.get('project', '?')}/{row.get('ifc_model', '?')}: {question[:70]}...")

        # Agent-side errors have an empty predicted answer -- nothing to judge.
        if not isinstance(predicted, str) or not predicted.strip():
            print("  SKIP: empty predicted answer (agent error -- re-run with run.py --resume)")
            skipped += 1
            continue

        try:
            category = int(row["category"])
        except (TypeError, ValueError):
            print(f"  SKIP: invalid category: {row.get('category')}")
            skipped += 1
            continue

        out = judge_single(
            question=question,
            ground_truth=str(row["ground_truth"]),
            predicted=predicted,
            category=category,
            judge=args.judge,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

        if out.classification == "error":
            print(f"  STILL ERROR: {out.error_message}")
            failed += 1
            continue

        print(f"  => {out.classification} "
              f"(faith={out.faithfulness}, comp={out.completeness}, "
              f"trans={out.transparency}, rel={out.relevance})")

        for col, val in asdict(out).items():
            df.loc[idx, col] = val
        df.loc[idx, "judge"] = args.judge
        succeeded += 1

    print(f"\nRe-judged: {succeeded} succeeded, {failed} still failed, {skipped} skipped "
          f"(out of {len(error_indices)} errors)")

    if succeeded == 0:
        print("No rows updated -- skipping file writes.")
        return 0

    df.to_csv(judged_csv, index=False)
    print(f"Updated CSV: {judged_csv}")

    metrics = calculate_metrics(df)
    print_report(metrics)

    judge = args.judge
    if summary_json.exists():
        try:
            judge = json.loads(summary_json.read_text()).get("judge", args.judge)
        except Exception:
            pass
    summary_json.write_text(json.dumps({
        "run_dir": str(run_dir),
        "judge": judge,
        "metrics": asdict(metrics),
    }, indent=2))
    print(f"\n> summary: {summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
