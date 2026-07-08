"""Phoenix-free runner: run the IFC or SQL agent and persist answers + traces.

This is a standalone alternative to pipelines/{ifc,sql}/run.py for when you want
the per-question execution trace on disk and don't want Phoenix at all. It does
NOT call init_tracing(), so no OTEL spans are exported and no traces/phoenix.db
is written. The trace data comes from AgentResult.trace (built in-process from
the LangGraph message history), not from Phoenix.

It does not modify or replace run.py -- run.py keeps writing to Phoenix as
before. Use whichever you need.

Outputs (results/<pipeline>/<run>/):
    results.csv    -- same schema as run.py (feeds evaluate.py unchanged)
    traces.json    -- {"<project>/<ifc_model>": [{question, answer, trace}, ...]}
                      (suite-format trace dump)
    traces.jsonl   -- one trace record per line, written incrementally (crash-safe)
    config.json    -- run metadata

Run it with the target pipeline's uv environment so the agent deps resolve:

    uv run --project pipelines/ifc python scripts/run_export.py --pipeline ifc --split test
    uv run --project pipelines/sql python scripts/run_export.py --pipeline sql --split test

Then judge + export to the suite format (both already Phoenix-free):

    uv run --project shared/eval python shared/eval/evaluate.py --run results/ifc/<run>
    uv run --project shared/eval python shared/eval/export_compat.py --run results/ifc/<run>
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_QUESTIONS_CSV = _REPO_ROOT / "data" / "questions" / "ifc-bench-v2.csv"
_SPLITS_JSON = _REPO_ROOT / "data" / "questions" / "splits.json"

# Per-pipeline wiring. Each entry: agent dir, factory name, interpreter attr,
# how to resolve the model file, and where results go.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TOOL_XML_RE = re.compile(r"<minimax:tool_call>.*?</minimax:tool_call>", re.DOTALL)


def _sanitize_answer(text: str) -> str:
    return _TOOL_XML_RE.sub("", _THINK_RE.sub("", text)).strip()


def _model_tag(model: str) -> str:
    return model.replace(":", "_").replace("/", "-").replace(" ", "-")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pipeline", required=True, choices=["ifc", "sql"])
    p.add_argument("--model", default="minimax:MiniMax-M2.7")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--category", type=int, default=None)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--split", choices=["dev", "test"], default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-attempts", type=int, default=3)
    p.add_argument("--recursion-limit", type=int, default=120)
    p.add_argument("--resume", type=str, default=None,
                   help="Resume a previous run dir name under results/<pipeline>/.")
    return p.parse_args()


def _load_split_projects(split: str) -> list[str]:
    with _SPLITS_JSON.open() as f:
        splits = json.load(f)
    if split not in splits:
        sys.exit(f"error: split '{split}' not in {_SPLITS_JSON}")
    return list(splits[split])


def _load_questions(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(_QUESTIONS_CSV)
    df = df.reset_index().rename(columns={"index": "question_id"})
    if args.split is not None:
        df = df[df["project"].astype(str).isin(_load_split_projects(args.split))]
    if args.category is not None:
        df = df[df["category"] == args.category]
    if args.project is not None:
        df = df[df["project"].astype(str) == str(args.project)]
    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit is not None:
        df = df.iloc[: args.limit]
    return df.reset_index(drop=True)


def _resolve_pipeline(pipeline: str):
    """Add the pipeline dir to sys.path and return (factory, interp_attr, resolver, results_dir)."""
    agent_dir = _REPO_ROOT / "pipelines" / pipeline
    sys.path.insert(0, str(agent_dir))
    results_dir = _REPO_ROOT / "results" / pipeline

    if pipeline == "ifc":
        from agent import create_ifc_agent  # noqa: E402

        bench = os.environ.get("IFC_BENCH_DIR")
        if not bench:
            sys.exit("error: IFC_BENCH_DIR env var not set (path to ifc-bench projects/ dir).")
        bench_dir = Path(bench).expanduser()
        if not bench_dir.is_absolute():
            bench_dir = _REPO_ROOT / bench_dir  # relative paths anchor at the repo root
        bench_dir = bench_dir.resolve()

        def resolver(project: str, ifc_model: str) -> Path:
            return bench_dir / str(project) / f"{ifc_model}.ifc"

        return create_ifc_agent, "_ifc_interpreter", resolver, results_dir

    from agent import create_sql_agent  # noqa: E402
    models_dir = _REPO_ROOT / "data" / "conversions" / "sql"

    def resolver(project: str, ifc_model: str) -> Path:
        return models_dir / str(project) / f"{ifc_model}.sqlite"

    return create_sql_agent, "_sql_interpreter", resolver, results_dir


def _append_csv(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _assemble_traces(jsonl: Path, out: Path) -> None:
    """Group the incremental jsonl records into the suite-format traces.json."""
    grouped: dict[str, list] = {}
    if not jsonl.exists():
        return
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            grouped.setdefault(rec["key"], []).append(
                {"question": rec["question"], "answer": rec["answer"], "trace": rec["trace"]}
            )
    out.write_text(json.dumps(grouped, indent=2, ensure_ascii=False))


def main() -> None:
    load_dotenv(_REPO_ROOT / ".env")
    args = _parse_args()

    # Import shared (installed in the pipeline venv) -- NOTE: no init_tracing call.
    from shared import answer_question  # noqa: E402

    factory, interp_attr, resolver, results_dir = _resolve_pipeline(args.pipeline)

    questions = _load_questions(args)
    if questions.empty:
        sys.exit("error: no questions match the given filters.")

    answered_qids: set[int] = set()
    if args.resume:
        run_dir = results_dir / args.resume
        if not run_dir.is_dir():
            sys.exit(f"error: resume directory not found: {run_dir}")
        run_id = args.resume
        if (run_dir / "results.csv").exists():
            answered_qids = set(pd.read_csv(run_dir / "results.csv")["question_id"].astype(int))
    else:
        run_id = f"{_model_tag(args.model)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    results_csv = run_dir / "results.csv"
    traces_jsonl = run_dir / "traces.jsonl"
    traces_json = run_dir / "traces.json"
    (run_dir / "config.json").write_text(json.dumps({
        "pipeline": args.pipeline, "model": args.model, "run_id": run_id,
        "tracing": "disabled (Phoenix bypassed)",
        "filters": {"category": args.category, "project": args.project,
                    "split": args.split, "offset": args.offset, "limit": args.limit},
        "recursion_limit": args.recursion_limit, "max_retries": args.max_retries,
    }, indent=2))

    print(f"> pipeline: {args.pipeline} | model: {args.model} | tracing: OFF (no Phoenix)")
    print(f"> questions: {len(questions)} | output: {run_dir}")
    if answered_qids:
        print(f"> resuming: {len(answered_qids)} already answered")
    print()

    agents: dict[tuple[str, str], object] = {}

    try:
        for i, row in questions.iterrows():
            qid = int(row["question_id"])
            project = str(row["project"])
            ifc_model = str(row["ifc_model"])
            question = str(row["question"])
            category = int(row["category"]) if pd.notna(row.get("category")) else None
            ground_truth = str(row["ground_truth"])

            if qid in answered_qids:
                continue

            model_path = resolver(project, ifc_model)
            if not model_path.is_file():
                print(f"[{i + 1}/{len(questions)}] SKIP qid={qid} -- missing: {model_path}")
                _append_csv(results_csv, {
                    "question_id": qid, "project": project, "ifc_model": ifc_model,
                    "category": category, "question": question, "ground_truth": ground_truth,
                    "predicted": "", "elapsed_s": 0.0, "input_tokens": 0, "output_tokens": 0,
                    "num_tool_calls": 0, "span_id": "", "error": f"missing model file: {model_path}",
                })
                continue

            print(f"[{i + 1}/{len(questions)}] qid={qid} project={project} cat={category}")
            print(f"  Q: {question[:160]}{'...' if len(question) > 160 else ''}")

            key = (project, ifc_model)
            if key not in agents:
                agents[key] = factory(args.model, str(model_path), max_retries=args.max_retries)
            else:
                interp = getattr(agents[key], interp_attr, None)
                if interp is not None:
                    try:
                        interp.reset()
                    except Exception:
                        agents[key] = factory(args.model, str(model_path), max_retries=args.max_retries)
            agent = agents[key]

            attempt = 0
            result = None
            last_error = ""
            while attempt < args.retry_attempts:
                attempt += 1
                try:
                    result = answer_question(
                        agent, question,
                        recursion_limit=args.recursion_limit, verbose=args.verbose,
                    )
                    break
                except GraphRecursionError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    print(f"  ! recursion limit hit -- skipping retries: {last_error}")
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    print(f"  ! attempt {attempt}/{args.retry_attempts} failed: {last_error}")
                    if args.verbose:
                        traceback.print_exc()
                    interp = getattr(agent, interp_attr, None)
                    if interp is not None:
                        try:
                            interp.reset()
                        except Exception:
                            pass

            if result is None:
                _append_csv(results_csv, {
                    "question_id": qid, "project": project, "ifc_model": ifc_model,
                    "category": category, "question": question, "ground_truth": ground_truth,
                    "predicted": "", "elapsed_s": 0.0, "input_tokens": 0, "output_tokens": 0,
                    "num_tool_calls": 0, "span_id": "", "error": last_error,
                })
                print(f"  ! giving up after {args.retry_attempts} attempts.\n")
                continue

            print(f"  A: {result.answer[:200].replace(chr(10), ' ')}"
                  f"{'...' if len(result.answer) > 200 else ''}")
            print(f"  -> {result.num_tool_calls} tool calls, "
                  f"{result.input_tokens}+{result.output_tokens} tok, {result.elapsed_s}s\n")

            _append_csv(results_csv, {
                "question_id": qid, "project": project, "ifc_model": ifc_model,
                "category": category, "question": question, "ground_truth": ground_truth,
                "predicted": _sanitize_answer(result.answer), "elapsed_s": result.elapsed_s,
                "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                "num_tool_calls": result.num_tool_calls, "span_id": result.span_id, "error": "",
            })
            # Trace keeps the raw answer (incl. <think>), matching the suite trace format.
            _append_jsonl(traces_jsonl, {
                "key": f"{project}/{ifc_model}", "question_id": qid,
                "question": question, "answer": result.answer, "trace": result.trace,
            })
            _assemble_traces(traces_jsonl, traces_json)

            if args.delay > 0:
                time.sleep(args.delay)

    finally:
        for agent in agents.values():
            interp = getattr(agent, interp_attr, None)
            if interp is not None:
                try:
                    interp.shutdown()
                except Exception:
                    pass
        _assemble_traces(traces_jsonl, traces_json)

    print(f"done -- results: {results_csv}")
    print(f"        traces:  {traces_json}")
    # Stable marker for orchestration scripts to detect the run directory.
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
