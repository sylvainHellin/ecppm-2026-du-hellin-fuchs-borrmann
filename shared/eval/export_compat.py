"""Export a judged run into the suite CSV format used for cross-pipeline comparison.

Read-only converter. Does NOT touch the existing run/eval workflow -- it just
reads a finished run directory (the one produced by run.py + evaluate.py) and
writes two extra files in the flat CSV + summary layout the analysis scripts
(analysis/*.py) consume, so runs from all four pipelines can be compared
side-by-side.

Input  (a run directory):
    <run>/judged.csv          (required -- produced by evaluate.py)
    <run>/config.json         (optional -- agent model, split, dataset path)

Output (written into --out, default <run>/export_compat):
    <name>.csv                suite CSV columns (see CSV_COLUMNS below)
    <name>_summary.json       summary JSON in the matching shape

Usage:
    # default: writes alongside the run, name derived from the run dir
    uv run python shared/eval/export_compat.py --run results/ifc/<run>

    # custom output dir / base filename
    uv run python shared/eval/export_compat.py \
        --run results/sql/<run> \
        --out /tmp/exports/sql \
        --name v2_minimax_MiniMax-M2.7_sql

Note: per-question execution traces are copied only when the run dir contains a
traces.json (written by pipelines/{filesystem,cypher}/run.py and by
scripts/run_export.py). Runs made with pipelines/{ifc,sql}/run.py keep their
traces in Phoenix, so only the CSV + summary are emitted for those.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Suite CSV column order, consumed by the analysis scripts.
CSV_COLUMNS = [
    "question", "ground_truth", "predicted", "project", "ifc_model", "category",
    "classification", "correct", "abstention",
    "faithfulness", "completeness", "transparency", "relevance", "justification",
    "input_tokens", "output_tokens", "elapsed_s",
]


def _as_bool(val) -> bool:
    """Coerce a CSV cell (bool / 'True' / 'False' / NaN) to a plain bool."""
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return False
    return str(val).strip().lower() in ("true", "1", "yes")


def _round(x: float, n: int = 4) -> float:
    return round(float(x), n)


def _criterion_rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    yes = int((df[col] == "Yes").sum())
    no = int((df[col] == "No").sum())
    return _round(yes / (yes + no)) if (yes + no) > 0 else 0.0


def _group_stats(df: pd.DataFrame, *, evaluated_key: bool = False) -> dict:
    """Counts + accuracy for one group of rows."""
    total = len(df)
    correct = int((df["classification"] == "correct").sum())
    wrong = int((df["classification"] == "wrong").sum())
    abstained = int((df["classification"] == "abstained").sum())
    out = {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "accuracy": _round(correct / (correct + wrong)) if (correct + wrong) > 0 else 0.0,
    }
    if evaluated_key:
        out["accuracy_evaluated"] = out["accuracy"]
    return out


def build_csv(judged: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["question"] = judged.get("question", "")
    out["ground_truth"] = judged.get("ground_truth", "")
    out["predicted"] = judged.get("predicted", "")
    out["project"] = judged.get("project", "")
    out["ifc_model"] = judged.get("ifc_model", "")
    out["category"] = judged.get("category", "")
    out["classification"] = judged.get("classification", "")
    out["correct"] = judged["classification"].eq("correct")
    out["abstention"] = judged.get("abstention", False).map(_as_bool) \
        if "abstention" in judged.columns else False
    out["faithfulness"] = judged.get("faithfulness", "")
    out["completeness"] = judged.get("completeness", "")
    out["transparency"] = judged.get("transparency", "")
    out["relevance"] = judged.get("relevance", "")
    out["justification"] = judged.get("justification", "")
    out["input_tokens"] = judged.get("input_tokens", 0)
    out["output_tokens"] = judged.get("output_tokens", 0)
    out["elapsed_s"] = judged.get("elapsed_s", 0.0)
    return out[CSV_COLUMNS]


def build_summary(judged: pd.DataFrame, config: dict, *, dataset: str) -> dict:
    total = len(judged)
    cls = judged["classification"]
    correct = int((cls == "correct").sum())
    wrong = int((cls == "wrong").sum())
    abstained = int((cls == "abstained").sum())
    error = int((cls == "error").sum())

    evaluated = correct + wrong
    answered = correct + wrong + abstained
    non_error = judged[cls != "error"]

    judge = str(judged.get("judge", pd.Series(["minimax"])).iloc[0]) if "judge" in judged.columns else "minimax"

    return {
        "dataset": dataset,
        "agent_model": config.get("model", ""),
        "judge_model": f"baml:{judge}",
        "judge_type": "baml",
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "correct_count": correct,
        "wrong_count": wrong,
        "abstained_count": abstained,
        "error_count": error,
        "accuracy": _round(correct / evaluated) if evaluated > 0 else 0.0,
        "abstention_rate": _round(abstained / answered) if answered > 0 else 0.0,
        "faithfulness_rate": _criterion_rate(non_error, "faithfulness"),
        "completeness_rate": _criterion_rate(non_error, "completeness"),
        "transparency_rate": _criterion_rate(non_error, "transparency"),
        "relevance_rate": _criterion_rate(non_error, "relevance"),
        "classification_distribution": {
            "correct": correct, "wrong": wrong,
            "abstained": abstained, "error": error,
        },
        "total_input_tokens": int(judged.get("input_tokens", pd.Series([0])).fillna(0).sum()),
        "total_output_tokens": int(judged.get("output_tokens", pd.Series([0])).fillna(0).sum()),
        "total_judge_duration_s": _round(judged.get("judge_duration", pd.Series([0.0])).fillna(0).sum(), 2),
        "average_elapsed_s": _round(judged.get("elapsed_s", pd.Series([0.0])).fillna(0).mean(), 2),
        "by_category": {
            str(int(cat)): _group_stats(judged[judged["category"] == cat])
            for cat in sorted(judged["category"].dropna().unique())
        },
        "by_project": {
            str(proj): _group_stats(judged[judged["project"].astype(str) == str(proj)], evaluated_key=True)
            for proj in sorted(judged["project"].astype(str).dropna().unique())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, type=Path,
                        help="Run directory containing judged.csv (+ optional config.json).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: <run>/export_compat).")
    parser.add_argument("--name", type=str, default=None,
                        help="Base filename for outputs (default: v2_<run dir name>).")
    parser.add_argument("--dataset", type=str, default="v2",
                        help="Dataset tag written into the summary (default: v2).")
    args = parser.parse_args()

    run_dir: Path = args.run.expanduser().resolve()
    judged_csv = run_dir / "judged.csv"
    if not judged_csv.is_file():
        sys.exit(f"error: judged.csv not found in {run_dir} (run evaluate.py first).")

    config = {}
    config_json = run_dir / "config.json"
    if config_json.is_file():
        config = json.loads(config_json.read_text())

    judged = pd.read_csv(judged_csv)
    if "classification" not in judged.columns:
        sys.exit("error: judged.csv has no 'classification' column -- is this an evaluated run?")

    out_dir: Path = (args.out.expanduser().resolve() if args.out else run_dir / "export_compat")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{args.dataset}_{run_dir.name}"

    csv_df = build_csv(judged)
    summary = build_summary(judged, config, dataset=args.dataset)

    csv_path = out_dir / f"{name}.csv"
    summary_path = out_dir / f"{name}_summary.json"
    csv_df.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"> wrote {csv_path}")
    print(f"> wrote {summary_path}")

    # Runs from pipelines/{filesystem,cypher}/run.py and scripts/run_export.py
    # leave a traces.json in the run dir; copy it into <name>_traces.json.
    # (pipelines/{ifc,sql}/run.py runs send traces to Phoenix instead, so this
    # file simply won't exist for those -- skipped.)
    traces_src = run_dir / "traces.json"
    if traces_src.is_file():
        traces_path = out_dir / f"{name}_traces.json"
        traces_path.write_text(traces_src.read_text())
        print(f"> wrote {traces_path}")
    else:
        print("> no traces.json in run dir (run.py run? traces are in Phoenix) -- skipping traces export")
    print(f"  {summary['total_questions']} questions | "
          f"accuracy {summary['accuracy']:.2%} | "
          f"{summary['correct_count']} correct / {summary['wrong_count']} wrong / "
          f"{summary['abstained_count']} abstained / {summary['error_count']} error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
