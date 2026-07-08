"""CLI: run the IFC DeepAgent against the ifc-bench-v2 dataset.

Usage:
    uv run python run.py                          # uses default minimax:MiniMax-M2.7
    uv run python run.py --model gemini:gemini-2.5-pro --limit 5 --category 1
    uv run python run.py --model anthropic:claude-sonnet-4-5-20250929 --verbose

Outputs:
    results/ifc/<model_tag>_<timestamp>/
        results.csv   -- one row per question (answer, tokens, tool calls, elapsed, span_id)
        config.json   -- run metadata (model, timestamp, filters, run_id)
    Full execution traces are captured by Phoenix -> traces/phoenix.sqlite
    (inspect at localhost:6006 -- see README for phoenix serve setup).
    Traces are grouped by run via session.id (using_session).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

from agent import create_ifc_agent
from shared import answer_question, init_tracing, using_metadata, using_session


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_QUESTIONS_CSV = _REPO_ROOT / "data" / "questions" / "ifc-bench-v2.csv"
_SPLITS_JSON = _REPO_ROOT / "data" / "questions" / "splits.json"
_RESULTS_DIR = _REPO_ROOT / "results" / "ifc"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the IFC DeepAgent against the ifc-bench-v2 dataset.",
    )
    parser.add_argument("--model", default="minimax:MiniMax-M2.7", help="Prefixed model id (default: minimax:MiniMax-M2.7).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N questions.")
    parser.add_argument("--category", type=int, default=None, help="Filter by category (1-4).")
    parser.add_argument("--project", type=str, default=None, help="Filter by project id.")
    parser.add_argument("--split", choices=["dev", "test"], default=None,
                        help="Restrict to projects in the given split (see data/questions/splits.json).")
    parser.add_argument("--verbose", action="store_true", help="Stream tool calls and results in real time.")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between questions.")
    parser.add_argument("--max-retries", type=int, default=3, help="LLM HTTP retry budget (retries on 429/5xx/timeout).")
    parser.add_argument("--retry-attempts", type=int, default=3, help="Retry budget for a failed question.")
    parser.add_argument("--recursion-limit", type=int, default=120, help="LangGraph recursion limit.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume a previous run. Pass the run directory name (e.g. minimax_MiniMax-M2.7_20260419_212612).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_tag(model: str) -> str:
    """Convert 'anthropic:claude-sonnet-4-5' -> 'anthropic_claude-sonnet-4-5' (filesystem-safe)."""
    return model.replace(":", "_").replace("/", "-").replace(" ", "-")


def _resolve_ifc_path(bench_dir: Path, project: str, ifc_model: str) -> Path:
    return bench_dir / str(project) / f"{ifc_model}.ifc"


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
        sys.exit(f"error: questions file not found at {_QUESTIONS_CSV}")
    df = pd.read_csv(_QUESTIONS_CSV)
    df = df.reset_index().rename(columns={"index": "question_id"})
    if args.split is not None:
        projects = _load_split_projects(args.split)
        df = df[df["project"].astype(str).isin(projects)]
    if args.category is not None:
        df = df[df["category"] == args.category]
    if args.project is not None:
        df = df[df["project"].astype(str) == str(args.project)]
    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit is not None:
        df = df.iloc[: args.limit]
    return df.reset_index(drop=True)


_TOOL_CALL_XML_RE = re.compile(r"<minimax:tool_call>.*?</minimax:tool_call>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _sanitize_answer(text: str) -> str:
    """Strip MiniMax artifacts from the final answer.

    Removes: (1) hallucinated tool-call XML emitted when tools are stripped by
    RecursionGuardMiddleware, and (2) <think>...</think> reasoning blocks so
    only the final user-facing answer is saved and evaluated.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    text = _TOOL_CALL_XML_RE.sub("", text)
    return text.strip()


def _append_result(results_csv: Path, row: dict) -> None:
    """Append a row to results.csv, creating header on first write."""
    df_row = pd.DataFrame([row])
    header = not results_csv.exists()
    df_row.to_csv(results_csv, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # Phoenix OTEL tracing -- all LangChain invocations in this process are captured.
    init_tracing(project_name="ifc")

    questions = _load_questions(args)
    if questions.empty:
        sys.exit("error: no questions match the given filters.")

    # Output dir
    answered_qids: set[int] = set()
    if args.resume:
        run_dir = _RESULTS_DIR / args.resume
        if not run_dir.is_dir():
            sys.exit(f"error: resume directory not found: {run_dir}")
        run_id = args.resume
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            answered_qids = set(pd.read_csv(results_csv)["question_id"].astype(int))
        config_json = run_dir / "config.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{_model_tag(args.model)}_{timestamp}"
        run_dir = _RESULTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        results_csv = run_dir / "results.csv"
        config_json = run_dir / "config.json"

        config_json.write_text(json.dumps({
            "model": args.model,
            "timestamp": timestamp,
            "run_id": run_id,
            "questions_csv": str(_QUESTIONS_CSV),
            "ifc_bench_dir": str(bench_dir),
            "num_questions": int(len(questions)),
            "filters": {"category": args.category, "project": args.project,
                        "split": args.split, "offset": args.offset, "limit": args.limit},
            "recursion_limit": args.recursion_limit,
            "max_retries": args.max_retries,
        }, indent=2))

    print(f"> model: {args.model}")
    print(f"> questions: {len(questions)} (out of {len(pd.read_csv(_QUESTIONS_CSV))})")
    if answered_qids:
        print(f"> resuming: {len(answered_qids)} already answered, skipping those")
    print(f"> output: {run_dir}")
    print(f"> Phoenix traces: traces/phoenix.sqlite (UI: http://localhost:6006)")
    print()

    # Agent cache: one agent per (project, ifc_model). Interpreter is reset between
    # questions on the same model so namespace is clean.
    agents: dict[tuple[str, str], object] = {}
    current_key: tuple[str, str] | None = None

    try:
        for i, row in questions.iterrows():
            with using_session(run_id):
                qid = int(row["question_id"])
                project = str(row["project"])
                ifc_model = str(row["ifc_model"])
                question = str(row["question"])
                category = int(row["category"]) if pd.notna(row.get("category")) else None
                ground_truth = str(row["ground_truth"])

                if qid in answered_qids:
                    continue

                ifc_path = _resolve_ifc_path(bench_dir, project, ifc_model)
                if not ifc_path.is_file():
                    print(f"[{i + 1}/{len(questions)}] SKIP qid={qid} -- missing IFC: {ifc_path}")
                    _append_result(results_csv, {
                        "question_id": qid, "project": project, "ifc_model": ifc_model,
                        "category": category, "question": question, "ground_truth": ground_truth,
                        "predicted": "", "elapsed_s": 0.0,
                        "input_tokens": 0, "output_tokens": 0, "num_tool_calls": 0,
                        "span_id": "",
                        "error": f"missing IFC file: {ifc_path}",
                    })
                    continue

                print(f"[{i + 1}/{len(questions)}] qid={qid} project={project} cat={category}")
                print(f"  Q: {question[:180]}{'...' if len(question) > 180 else ''}")

                key = (project, ifc_model)
                if key not in agents:
                    # New (project, model) pair -> build a new agent bound to this IFC path.
                    agents[key] = create_ifc_agent(args.model, str(ifc_path), max_retries=args.max_retries)
                    current_key = key
                else:
                    # Reuse -- reset the kernel for a clean namespace.
                    agent = agents[key]
                    interp = getattr(agent, "_ifc_interpreter", None)
                    if interp is not None and current_key == key:
                        interp.reset()
                    current_key = key

                agent = agents[key]

                # Retry loop
                attempt = 0
                result = None
                last_error: str = ""
                while attempt < args.retry_attempts:
                    attempt += 1
                    try:
                        metadata = {
                            "question_id": str(qid),
                            "project": project,
                            "ifc_model": ifc_model,
                            "category": str(category) if category is not None else "",
                            "model": args.model,
                        }
                        with using_metadata(metadata):
                            result = answer_question(
                                agent,
                                question,
                                recursion_limit=args.recursion_limit,
                                verbose=args.verbose,
                            )
                        break
                    except GraphRecursionError as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        print(f"  ! recursion limit hit -- skipping retries: {last_error}")
                        if args.verbose:
                            traceback.print_exc()
                        break
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        print(f"  ! attempt {attempt}/{args.retry_attempts} failed: {last_error}")
                        if args.verbose:
                            traceback.print_exc()
                        # Fresh kernel before retry
                        interp = getattr(agent, "_ifc_interpreter", None)
                        if interp is not None:
                            try:
                                interp.reset()
                            except Exception:
                                pass

                if result is None:
                    _append_result(results_csv, {
                        "question_id": qid, "project": project, "ifc_model": ifc_model,
                        "category": category, "question": question, "ground_truth": ground_truth,
                        "predicted": "", "elapsed_s": 0.0,
                        "input_tokens": 0, "output_tokens": 0, "num_tool_calls": 0,
                        "span_id": "",
                        "error": last_error,
                    })
                    print(f"  ! giving up after {args.retry_attempts} attempts.\n")
                    continue

                predicted_preview = result.answer[:240].replace("\n", " ")
                print(
                    f"  A: {predicted_preview}{'...' if len(result.answer) > 240 else ''}\n"
                    f"  -> {result.num_tool_calls} tool calls, "
                    f"{result.input_tokens}+{result.output_tokens} tok, "
                    f"{result.elapsed_s}s\n"
                )

                _append_result(results_csv, {
                    "question_id": qid, "project": project, "ifc_model": ifc_model,
                    "category": category, "question": question, "ground_truth": ground_truth,
                    "predicted": _sanitize_answer(result.answer), "elapsed_s": result.elapsed_s,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "num_tool_calls": result.num_tool_calls,
                    "span_id": result.span_id,
                    "error": "",
                })

                if args.delay > 0:
                    time.sleep(args.delay)

    finally:
        # Shut down every cached kernel so the process can exit cleanly.
        for agent in agents.values():
            interp = getattr(agent, "_ifc_interpreter", None)
            if interp is not None:
                try:
                    interp.shutdown()
                except Exception:
                    pass

    print(f"done -- results saved to {run_dir}")


if __name__ == "__main__":
    main()
