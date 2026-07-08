#!/usr/bin/env python3
"""
Merge results from parallel evaluation runs.

When running a pipeline's run.py in multiple sessions (split by --project,
--projects, --offset, or --category), each session produces its own CSV.
This script merges them into a single result file with regenerated summary.

Usage:
    # Merge by glob (run dirs under results/<pipeline>/)
    python scripts/merge_results.py results/filesystem/openai_gpt-4.1_*/results.csv

    # Merge specific files
    python scripts/merge_results.py results/filesystem/run1/results.csv results/filesystem/run2/results.csv

    # Custom output path
    python scripts/merge_results.py results/cypher/*/results.csv -o results/cypher/merged.csv

    # Also merge trace files
    python scripts/merge_results.py results/filesystem/*/results.csv --merge-traces
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def detect_judge_type(df: pd.DataFrame) -> str:
    if "classification" in df.columns:
        return "baml"
    return "langchain"


def merge_csvs(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate CSVs, deduplicating by (question, project, ifc_model)."""
    frames = []
    for p in paths:
        try:
            frame = pd.read_csv(p)
            print(f"  Loaded {len(frame):>4d} rows from {p.name}")
            frames.append(frame)
        except Exception as e:
            print(f"  SKIP {p.name}: {e}", file=sys.stderr)

    if not frames:
        print("ERROR: No valid CSV files to merge.", file=sys.stderr)
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)

    dedup_keys = ["question", "project", "ifc_model"]
    available_keys = [k for k in dedup_keys if k in merged.columns]
    if available_keys:
        merged = merged.drop_duplicates(subset=available_keys, keep="last")

    after = len(merged)
    if before != after:
        print(f"  Removed {before - after} duplicate rows")

    return merged


def generate_langchain_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {"total": 0}

    correct_count = int(df["correct"].sum())
    avg_score = float(df["score"].mean())
    avg_elapsed = float(df["elapsed_s"].mean()) if "elapsed_s" in df.columns else 0

    summary = {
        "merged": True,
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "correct_count": correct_count,
        "accuracy": round(correct_count / total, 4),
        "average_score": round(avg_score, 3),
        "average_elapsed_s": round(avg_elapsed, 2),
        "score_distribution": {
            str(s): int(c) for s, c in sorted(df["score"].value_counts().items())
        },
    }

    if "category" in df.columns and df["category"].notna().any():
        cat_stats = {}
        for cat, group in df.groupby("category"):
            cat_total = len(group)
            cat_correct = int(group["correct"].sum())
            cat_stats[str(int(cat))] = {
                "total": cat_total,
                "correct": cat_correct,
                "accuracy": round(cat_correct / cat_total, 4) if cat_total > 0 else 0,
                "average_score": round(float(group["score"].mean()), 3),
            }
        summary["by_category"] = cat_stats

    if "project" in df.columns:
        proj_stats = {}
        for proj, group in df.groupby("project"):
            proj_total = len(group)
            proj_correct = int(group["correct"].sum())
            proj_stats[str(proj)] = {
                "total": proj_total,
                "correct": proj_correct,
                "accuracy": round(proj_correct / proj_total, 4) if proj_total > 0 else 0,
                "average_score": round(float(group["score"].mean()), 3),
            }
        summary["by_project"] = proj_stats

    return summary


def generate_baml_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {"total": 0}

    correct = int((df["classification"] == "correct").sum())
    wrong = int((df["classification"] == "wrong").sum())
    abstained = int((df["classification"] == "abstained").sum())
    errors = int((df["classification"] == "error").sum())
    evaluated = correct + wrong
    total_eval = correct + wrong + abstained

    def _criterion_rate(col: str) -> float:
        if col not in df.columns:
            return 0.0
        yes = int((df[col] == "Yes").sum())
        no = int((df[col] == "No").sum())
        return yes / (yes + no) if (yes + no) > 0 else 0.0

    summary = {
        "merged": True,
        "judge_type": "baml",
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "correct_count": correct,
        "wrong_count": wrong,
        "abstained_count": abstained,
        "error_count": errors,
        "accuracy": round(correct / evaluated, 4) if evaluated > 0 else 0,
        "abstention_rate": round(abstained / total_eval, 4) if total_eval > 0 else 0,
        "faithfulness_rate": round(_criterion_rate("faithfulness"), 4),
        "completeness_rate": round(_criterion_rate("completeness"), 4),
        "transparency_rate": round(_criterion_rate("transparency"), 4),
        "relevance_rate": round(_criterion_rate("relevance"), 4),
        "classification_distribution": {
            "correct": correct,
            "wrong": wrong,
            "abstained": abstained,
            "error": errors,
        },
    }

    if "elapsed_s" in df.columns and len(df) > 0:
        summary["average_elapsed_s"] = round(float(df["elapsed_s"].mean()), 2)

    if "input_tokens" in df.columns:
        summary["total_input_tokens"] = int(df["input_tokens"].sum())
    if "output_tokens" in df.columns:
        summary["total_output_tokens"] = int(df["output_tokens"].sum())

    if "category" in df.columns and df["category"].notna().any():
        cat_stats = {}
        for cat, group in df.groupby("category"):
            cat_total = len(group)
            cat_correct = int((group["classification"] == "correct").sum())
            cat_wrong = int((group["classification"] == "wrong").sum())
            cat_abstained = int((group["classification"] == "abstained").sum())
            cat_evaluated = cat_correct + cat_wrong
            cat_stats[str(int(cat))] = {
                "total": cat_total,
                "correct": cat_correct,
                "wrong": cat_wrong,
                "abstained": cat_abstained,
                "accuracy": round(cat_correct / cat_evaluated, 4) if cat_evaluated > 0 else 0,
            }
        summary["by_category"] = cat_stats

    if "project" in df.columns:
        proj_stats = {}
        for proj, group in df.groupby("project"):
            proj_total = len(group)
            proj_correct = int(group["correct"].sum())
            proj_wrong = int((group["classification"] == "wrong").sum())
            proj_abstained = int((group["classification"] == "abstained").sum())
            proj_evaluated = proj_correct + proj_wrong
            proj_stats[str(proj)] = {
                "total": proj_total,
                "correct": proj_correct,
                "wrong": proj_wrong,
                "abstained": proj_abstained,
                "accuracy": round(proj_correct / proj_total, 4) if proj_total > 0 else 0,
                "accuracy_evaluated": round(proj_correct / proj_evaluated, 4) if proj_evaluated > 0 else 0,
            }
        summary["by_project"] = proj_stats

    return summary


def merge_trace_files(csv_paths: list[Path], output_path: Path):
    """Merge trace JSON files that correspond to the given CSV result files."""
    merged_traces = {}
    found = 0
    for csv_path in csv_paths:
        trace_path = csv_path.with_name(csv_path.stem + "_traces.json")
        if not trace_path.exists():
            trace_path = csv_path.with_name(
                csv_path.name.replace(".csv", "_traces.json")
            )
        if not trace_path.exists():
            continue
        try:
            with open(trace_path) as f:
                traces = json.load(f)
            for key, entries in traces.items():
                merged_traces.setdefault(key, []).extend(entries)
            found += 1
            print(f"  Loaded traces from {trace_path.name}")
        except Exception as e:
            print(f"  SKIP traces {trace_path.name}: {e}", file=sys.stderr)

    if found == 0:
        print("  No trace files found to merge.")
        return

    trace_output = output_path.with_name(output_path.stem + "_traces.json")
    with open(trace_output, "w") as f:
        json.dump(merged_traces, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Merged traces saved to {trace_output}")


def print_baml_summary(s: dict):
    print(f"\n{'='*60}")
    print("MERGED EVALUATION SUMMARY (BAML)")
    print(f"{'='*60}")
    print(f"  Total:        {s.get('total_questions')}")
    print(f"  Correct:      {s.get('correct_count')}")
    print(f"  Wrong:        {s.get('wrong_count')}")
    print(f"  Abstained:    {s.get('abstained_count')}")
    print(f"  Errors:       {s.get('error_count')}")
    print(f"  Accuracy:     {s.get('accuracy', 0):.1%}")
    print(f"  Abstention:   {s.get('abstention_rate', 0):.1%}")
    print(f"  Avg time:     {s.get('average_elapsed_s', 0):.1f}s per question")
    print(f"\n  Criterion rates:")
    print(f"    Faithfulness:  {s.get('faithfulness_rate', 0):.1%}")
    print(f"    Completeness:  {s.get('completeness_rate', 0):.1%}")
    print(f"    Transparency:  {s.get('transparency_rate', 0):.1%}")
    print(f"    Relevance:     {s.get('relevance_rate', 0):.1%}")

    if "by_category" in s:
        print(f"\n  By category:")
        print(f"    {'Cat':>4s}  {'Total':>5s}  {'Correct':>7s}  {'Wrong':>5s}  {'Abst':>5s}  {'Accuracy':>8s}")
        print(f"    {'----':>4s}  {'-----':>5s}  {'-------':>7s}  {'-----':>5s}  {'-----':>5s}  {'--------':>8s}")
        for cat in sorted(s["by_category"].keys()):
            c = s["by_category"][cat]
            print(f"    {cat:>4s}  {c['total']:>5d}  {c['correct']:>7d}  {c['wrong']:>5d}  {c['abstained']:>5d}  {c['accuracy']:>8.1%}")

    if "by_project" in s:
        print(f"\n  By project:")
        print(f"    {'Project':<35s}  {'Total':>5s}  {'Correct':>7s}  {'Accuracy':>8s}")
        print(f"    {'-------':<35s}  {'-----':>5s}  {'-------':>7s}  {'--------':>8s}")
        for proj in sorted(s["by_project"].keys()):
            p = s["by_project"][proj]
            print(f"    {proj:<35s}  {p['total']:>5d}  {p['correct']:>7d}  {p['accuracy']:>8.1%}")

    print(f"\n{'='*60}\n")


def print_langchain_summary(s: dict):
    print(f"\n{'='*60}")
    print("MERGED EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:        {s.get('total_questions')}")
    print(f"  Correct:      {s.get('correct_count')}")
    print(f"  Accuracy:     {s.get('accuracy', 0):.1%}")
    print(f"  Avg score:    {s.get('average_score', 0):.2f} / 5")
    print(f"  Avg time:     {s.get('average_elapsed_s', 0):.1f}s per question")

    dist = s.get("score_distribution", {})
    if dist:
        print(f"\n  Score distribution:")
        for score in sorted(dist.keys()):
            count = dist[score]
            bar = "#" * count
            print(f"    {score}: {count:4d}  {bar}")

    if "by_category" in s:
        print(f"\n  By category:")
        print(f"    {'Cat':>4s}  {'Total':>5s}  {'Correct':>7s}  {'Accuracy':>8s}  {'Avg Score':>9s}")
        for cat in sorted(s["by_category"].keys()):
            c = s["by_category"][cat]
            print(f"    {cat:>4s}  {c['total']:>5d}  {c['correct']:>7d}  {c['accuracy']:>8.1%}  {c['average_score']:>9.2f}")

    if "by_project" in s:
        print(f"\n  By project:")
        print(f"    {'Project':<35s}  {'Total':>5s}  {'Correct':>7s}  {'Accuracy':>8s}")
        for proj in sorted(s["by_project"].keys()):
            p = s["by_project"][proj]
            print(f"    {proj:<35s}  {p['total']:>5d}  {p['correct']:>7d}  {p['accuracy']:>8.1%}")

    print(f"\n{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Merge CSV results from parallel evaluate.py runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/merge_results.py results/filesystem/openai_gpt-4.1_*/results.csv
  python scripts/merge_results.py results/filesystem/run1/results.csv results/filesystem/run2/results.csv -o results/filesystem/merged.csv
  python scripts/merge_results.py results/cypher/*/results.csv --merge-traces
""",
    )
    ap.add_argument("csvs", nargs="+", help="CSV result files to merge")
    ap.add_argument("-o", "--output", default=None, help="Output CSV path (default: auto-generated)")
    ap.add_argument("--merge-traces", action="store_true", help="Also merge corresponding trace JSON files")
    args = ap.parse_args()

    csv_paths = [Path(p) for p in args.csvs]
    csv_paths = [p for p in csv_paths if p.exists() and p.suffix == ".csv"]
    # Exclude files that are themselves merge outputs (contain "_merged")
    csv_paths = [p for p in csv_paths if "_merged" not in p.stem]

    if not csv_paths:
        print("ERROR: No valid CSV files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(csv_paths)} result files...")
    merged = merge_csvs(csv_paths)
    judge_type = detect_judge_type(merged)
    print(f"  Judge type: {judge_type}")
    print(f"  Total rows: {len(merged)}")

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent = csv_paths[0].parent
        # Try to extract a common prefix from filenames
        stems = [p.stem for p in csv_paths]
        prefix = stems[0].rsplit("_", 1)[0] if len(stems) == 1 else _common_prefix(stems)
        output_path = parent / f"{prefix}_merged_{timestamp}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\nMerged CSV saved to {output_path}")

    if judge_type == "baml":
        summary = generate_baml_summary(merged)
        print_baml_summary(summary)
    else:
        summary = generate_langchain_summary(merged)
        print_langchain_summary(summary)

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")

    if args.merge_traces:
        merge_trace_files(csv_paths, output_path)


def _common_prefix(strings: list[str]) -> str:
    """Find the longest common prefix of a list of strings, trimmed to last '_'."""
    if not strings:
        return "merged"
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return "merged"
    # Trim to last underscore for clean naming
    idx = prefix.rfind("_")
    return prefix[:idx] if idx > 0 else prefix or "merged"


if __name__ == "__main__":
    main()
