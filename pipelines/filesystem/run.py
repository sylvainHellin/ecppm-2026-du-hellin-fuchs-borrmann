#!/usr/bin/env python3
"""CLI: run the filesystem DeepAgent against the ifc-bench-v2 dataset.

The agent answers each question by exploring a JSON filesystem representation
of the IFC model (produced by ifc2fs.py / convert_all.py) with read-only file
tools plus shell execution.

Usage:
    uv run python run.py --model minimax:MiniMax-M2.7 --split test
    uv run python run.py --model openai:gpt-4.1 --limit 5 --category 1
    uv run python run.py --model openai:gpt-4.1 --resume <run_dir_name>

Outputs:
    results/filesystem/<model_tag>_<timestamp>/
        results.csv    -- one row per question (predicted answer, elapsed, tool calls)
        config.json    -- run metadata (model, filters, run_id)
        traces.json    -- full execution traces grouped by <project>/<ifc_model>
        traces.json.gz -- compressed copy, written at the end of the run

Judging happens offline with the shared BAML judge (MiniMax M2.7):
    uv run --project ../../shared/eval python ../../shared/eval/evaluate.py \
        --run ../../results/filesystem/<run>
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

import argparse
import gzip
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from agent import answer_question, create_ifc_agent


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_QUESTIONS_CSV = _REPO_ROOT / "data" / "questions" / "ifc-bench-v2.csv"
_SPLITS_JSON = _REPO_ROOT / "data" / "questions" / "splits.json"
_CONVERSIONS_DIR = _REPO_ROOT / "data" / "conversions" / "ifc_filesys"
_RESULTS_DIR = _REPO_ROOT / "results" / "filesystem"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the filesystem DeepAgent against the ifc-bench-v2 dataset.",
    )
    parser.add_argument("--model", default="minimax:MiniMax-M2.7",
                        help="Prefixed model id (default: minimax:MiniMax-M2.7).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N questions.")
    parser.add_argument("--category", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Filter by category (1-4).")
    parser.add_argument("--project", type=str, default=None, help="Filter by project id (single).")
    parser.add_argument("--projects", nargs="+", default=None,
                        help="Filter by multiple project ids (for parallel splits).")
    parser.add_argument("--split", choices=["dev", "test"], default=None,
                        help="Restrict to projects in the given split (see data/questions/splits.json).")
    parser.add_argument("--projects-dir", type=str, default=None,
                        help="Override conversions directory containing <project>/<ifc_model>_fs "
                             f"(default: {_CONVERSIONS_DIR}).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print agent tool calls in real time.")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between questions.")
    parser.add_argument("--agent-retries", type=int, default=3,
                        help="Max retry attempts per question on agent error (default 3).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume a previous run. Pass the run directory name "
                             "(e.g. minimax_MiniMax-M2.7_20260419_212612).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_tag(model: str) -> str:
    """Convert 'openai:gpt-4.1' -> 'openai_gpt-4.1' (filesystem-safe)."""
    return model.replace(":", "_").replace("/", "-").replace(" ", "-")


def _load_split_projects(split: str) -> list[str]:
    if not _SPLITS_JSON.exists():
        sys.exit(f"error: splits file not found at {_SPLITS_JSON}")
    with _SPLITS_JSON.open() as f:
        splits = json.load(f)
    if split not in splits:
        sys.exit(f"error: split '{split}' not in {_SPLITS_JSON}. Available: "
                 f"{[k for k in splits if not k.startswith('_')]}")
    return list(splits[split])


def _load_questions(args: argparse.Namespace) -> pd.DataFrame:
    if not _QUESTIONS_CSV.exists():
        sys.exit(f"error: questions file not found at {_QUESTIONS_CSV} "
                 "(run scripts/download_data.py first).")
    df = pd.read_csv(_QUESTIONS_CSV)
    df = df.reset_index().rename(columns={"index": "question_id"})
    if args.split is not None:
        projects = _load_split_projects(args.split)
        df = df[df["project"].astype(str).isin(projects)]
    if args.category is not None:
        df = df[df["category"] == args.category]
    if args.project is not None:
        df = df[df["project"].astype(str) == str(args.project)]
    elif args.projects is not None:
        df = df[df["project"].astype(str).isin([str(p) for p in args.projects])]
    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit is not None:
        df = df.iloc[: args.limit]
    return df.reset_index(drop=True)


def _append_result(results_csv: Path, row: dict) -> None:
    """Append a row to results.csv, creating header on first write."""
    df_row = pd.DataFrame([row])
    header = not results_csv.exists()
    df_row.to_csv(results_csv, mode="a", header=header, index=False)


def _load_traces(run_dir: Path) -> dict[str, list]:
    """Load existing traces from a resumed run (json or json.gz)."""
    for name in ("traces.json", "traces.json.gz"):
        tp = run_dir / name
        if tp.exists():
            try:
                if tp.suffix == ".gz":
                    with gzip.open(tp, "rt") as f:
                        return json.load(f)
                with open(tp) as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _wait_for_glm_quota_reset(error_text: str) -> bool:
    """Parse GLM quota-type 429 reset time and sleep until it passes.

    Returns True if a quota reset was detected and waited, False otherwise.
    """
    match = re.search(
        r"Your limit will reset at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
        error_text,
    )
    if not match:
        return False
    reset_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    wait_seconds = (reset_time - datetime.now()).total_seconds()
    if wait_seconds > 0:
        print(f"\n  GLM quota reached. Waiting until {reset_time} "
              f"({wait_seconds:.0f}s / {wait_seconds / 60:.1f}min)...")
        time.sleep(wait_seconds + 10)
    else:
        print(f"\n  GLM quota reset time {reset_time} already passed, retrying...")
        time.sleep(5)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv(_REPO_ROOT / ".env")

    args = _parse_args()
    if args.project and args.projects:
        sys.exit("error: --project and --projects are mutually exclusive")

    conversions_dir = Path(args.projects_dir).expanduser().resolve() if args.projects_dir \
        else _CONVERSIONS_DIR

    questions = _load_questions(args)
    if questions.empty:
        sys.exit("error: no questions match the given filters.")

    # Output dir
    answered_qids: set[int] = set()
    all_traces: dict[str, list] = {}
    if args.resume:
        run_dir = _RESULTS_DIR / args.resume
        if not run_dir.is_dir():
            sys.exit(f"error: resume directory not found: {run_dir}")
        run_id = args.resume
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            answered_qids = set(pd.read_csv(results_csv)["question_id"].astype(int))
        all_traces = _load_traces(run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{_model_tag(args.model)}_{timestamp}"
        run_dir = _RESULTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        results_csv = run_dir / "results.csv"

        (run_dir / "config.json").write_text(json.dumps({
            "model": args.model,
            "timestamp": timestamp,
            "run_id": run_id,
            "questions_csv": str(_QUESTIONS_CSV),
            "conversions_dir": str(conversions_dir),
            "num_questions": int(len(questions)),
            "filters": {"category": args.category, "project": args.project,
                        "projects": args.projects, "split": args.split,
                        "offset": args.offset, "limit": args.limit},
            "agent_retries": args.agent_retries,
        }, indent=2))

    trace_file = run_dir / "traces.json"
    is_glm_agent = args.model.startswith("glm:")

    print(f"> model: {args.model}")
    print(f"> questions: {len(questions)}")
    if answered_qids:
        print(f"> resuming: {len(answered_qids)} already answered, skipping those")
    print(f"> output: {run_dir}")
    print()

    # Cache agents per (project, ifc_model) pair
    agent_cache: dict[tuple[str, str], object] = {}

    total = len(questions)
    for i, row in questions.iterrows():
        qid = int(row["question_id"])
        project = str(row["project"])
        ifc_model = str(row["ifc_model"])
        question = str(row["question"])
        ground_truth = str(row["ground_truth"])
        category = int(row["category"]) if pd.notna(row.get("category")) else None

        if qid in answered_qids:
            continue

        fs_root = conversions_dir / project / f"{ifc_model}_fs"
        print(f"[{i + 1}/{total}] qid={qid} {project}/{ifc_model}: {question[:80]}...")

        # Check filesystem conversion exists
        if not fs_root.exists():
            print(f"  ERROR: Filesystem not found at {fs_root} -- run convert_all.py first")
            _append_result(results_csv, {
                "question_id": qid, "project": project, "ifc_model": ifc_model,
                "category": category, "question": question, "ground_truth": ground_truth,
                "predicted": "", "elapsed_s": 0.0, "num_tool_calls": 0,
                "error": f"Filesystem not found at {fs_root}",
            })
            continue

        # Get or create agent for this (project, model) pair
        cache_key = (project, ifc_model)
        if cache_key not in agent_cache:
            print(f"  Creating agent for {project}/{ifc_model}...")
            agent_cache[cache_key] = create_ifc_agent(args.model, str(fs_root))
        agent = agent_cache[cache_key]

        # Run agent with retry on failure
        last_error = None
        elapsed = 0.0
        predicted = ""
        trace: list = []
        num_tool_calls = 0
        for attempt in range(1, args.agent_retries + 2):
            t0 = time.time()
            try:
                while True:
                    try:
                        agent_result = answer_question(agent, question, verbose=args.verbose)
                        break
                    except Exception as exc:
                        if is_glm_agent and _wait_for_glm_quota_reset(str(exc)):
                            continue
                        raise
                predicted = agent_result.answer
                trace = agent_result.trace
                elapsed = time.time() - t0
                num_tool_calls = sum(
                    len(e.get("tool_calls", [])) for e in trace if e.get("role") == "assistant"
                )
                print(f"  Agent answered in {elapsed:.1f}s ({num_tool_calls} tool calls)")
                last_error = None
                break
            except Exception as e:
                last_error = e
                elapsed = time.time() - t0
                if attempt <= args.agent_retries:
                    print(f"  Agent FAILED (attempt {attempt}/{args.agent_retries + 1}) "
                          f"after {elapsed:.1f}s: {e}")
                    print(f"  Retrying with fresh context...")
                else:
                    print(f"  Agent FAILED after {args.agent_retries + 1} attempts "
                          f"({elapsed:.1f}s): {e}")
                    traceback.print_exc()
        if last_error is not None:
            predicted = ""
            trace = []
            num_tool_calls = 0

        # Incrementally save trace grouped by project/model
        trace_key = f"{project}/{ifc_model}"
        all_traces.setdefault(trace_key, []).append({
            "question": question,
            "answer": predicted if last_error is None else f"ERROR: {last_error}",
            "trace": trace,
        })
        with open(trace_file, "w") as f:
            json.dump(all_traces, f, indent=2, ensure_ascii=False, default=str)

        _append_result(results_csv, {
            "question_id": qid, "project": project, "ifc_model": ifc_model,
            "category": category, "question": question, "ground_truth": ground_truth,
            "predicted": predicted, "elapsed_s": round(elapsed, 2),
            "num_tool_calls": num_tool_calls,
            "error": f"{type(last_error).__name__}: {last_error}" if last_error is not None else "",
        })

        if args.delay > 0 and i < total - 1:
            time.sleep(args.delay)

    # Compress traces for version control
    if trace_file.exists():
        gz_file = trace_file.with_suffix(".json.gz")
        with open(trace_file, "rb") as f_in, gzip.open(gz_file, "wb") as f_out:
            f_out.writelines(f_in)
        print(f"Compressed traces saved to {gz_file}")

    print(f"done -- results saved to {run_dir}")


if __name__ == "__main__":
    main()
