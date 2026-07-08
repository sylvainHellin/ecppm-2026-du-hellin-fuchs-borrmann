"""BAML multi-criteria LLM judge for BIM QA evaluation.

Input:  results/<pipeline>/<run>/results.csv
        (columns required: question, ground_truth, predicted, category;
         extra columns are preserved and merged into the output.)

Output: results/<pipeline>/<run>/judged.csv
        (all input columns + classification, abstention, faithfulness,
         completeness, transparency, relevance, justification, judge,
         judge_duration, judge_input_tokens, judge_output_tokens)

A summary JSON is written next to judged.csv.

Prerequisites:
    The generated BAML client (shared/eval/baml_client/) is committed;
    `uv sync` in shared/eval is all that is needed (see README 'Evaluation and judging').

Usage:
    uv run python shared/eval/evaluate.py --run results/ifc/<run>
    uv run python shared/eval/evaluate.py --run <run> --judge gemini
    uv run python shared/eval/evaluate.py --run <run> --limit 20 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from baml_py.baml_py import Collector
from dotenv import load_dotenv

# Import BAML client. This module runs from two possible roots:
#  (a) as a script: `shared/eval/evaluate.py`  -> baml_client is a sibling
#  (b) as a module: `from eval.evaluate import ...` -> baml_client is a subpackage
try:
    from baml_client import b
    from baml_client.types import AnswerEvaluationResult, CriterionResult, QuestionCategory
except ImportError:
    # Add this script's directory to sys.path and retry.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from baml_client import b  # type: ignore
    from baml_client.types import AnswerEvaluationResult, CriterionResult, QuestionCategory  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[int, QuestionCategory] = {
    1: QuestionCategory.Category1,
    2: QuestionCategory.Category2,
    3: QuestionCategory.Category3,
    4: QuestionCategory.Category4,
}

CATEGORY_NAMES: dict[int, str] = {
    1: "Direct Retrieval",
    2: "Computational Aggregation",
    3: "Geometric/Spatial",
    4: "Incomplete Information",
}

JUDGE_CHOICES = ("minimax", "gemini", "gpt")

Classification = Literal["correct", "wrong", "abstained", "error"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JudgeOutput:
    classification: Classification
    abstention: bool | None = None
    faithfulness: str | None = None
    completeness: str | None = None
    transparency: str | None = None
    relevance: str | None = None
    justification: str | None = None
    error_message: str | None = None
    judge_duration: float = 0.0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0


@dataclass
class AggregateMetrics:
    total: int = 0
    correct: int = 0
    wrong: int = 0
    abstained: int = 0
    errors: int = 0
    accuracy: float = 0.0
    abstention_rate: float = 0.0
    faithfulness_rate: float = 0.0
    completeness_rate: float = 0.0
    transparency_rate: float = 0.0
    relevance_rate: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration: float = 0.0
    by_category: dict[int, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core judge
# ---------------------------------------------------------------------------

def _judge_fn(judge: str):
    if judge == "gemini":
        return b.EvaluateResponseGemini
    if judge == "gpt":
        return b.EvaluateResponseGPT
    return b.EvaluateResponse  # default = MinimaxM2_Fireworks


def _derive_classification(result: AnswerEvaluationResult) -> Classification:
    """Map the 5-criterion BAML result to a 3-class label (+ 'error' on failure)."""
    if result.abstention:
        return "abstained"
    if (
        result.faithfulness == CriterionResult.Yes
        and result.completeness == CriterionResult.Yes
        and result.transparency == CriterionResult.Yes
        and result.relevance == CriterionResult.Yes
    ):
        return "correct"
    return "wrong"


def judge_single(
    *,
    question: str,
    ground_truth: str,
    predicted: str,
    category: int,
    judge: str = "minimax",
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> JudgeOutput:
    """Run the BAML judge on a single QA triple. Retries transient failures."""
    baml_category = CATEGORY_MAP.get(category)
    if baml_category is None:
        return JudgeOutput(
            classification="error",
            error_message=f"invalid category: {category} (must be 1-4)",
        )

    collector = Collector(name="BimJudge")
    client = b.with_options(collector=collector)
    fn = _judge_fn(judge)
    fn = getattr(client, fn.__name__)

    last_error: Exception | None = None
    duration = 0.0
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            result: AnswerEvaluationResult = fn(
                question=question,
                category=baml_category,
                ground_truth=ground_truth,
                system_response=predicted,
            )
            duration = time.time() - t0
            input_tok = 0
            output_tok = 0
            if collector.last:
                usage = collector.last.usage
                input_tok = usage.input_tokens or 0
                output_tok = usage.output_tokens or 0
            return JudgeOutput(
                classification=_derive_classification(result),
                abstention=result.abstention,
                faithfulness=result.faithfulness.value,
                completeness=result.completeness.value,
                transparency=result.transparency.value,
                relevance=result.relevance.value,
                justification=result.justification,
                judge_duration=round(duration, 2),
                judge_input_tokens=input_tok,
                judge_output_tokens=output_tok,
            )
        except Exception as exc:
            last_error = exc
            duration = time.time() - t0
            print(
                f"    ! judge attempt {attempt}/{max_retries} failed: "
                f"{type(exc).__name__}: {str(exc)[:160]}"
            )
            if attempt < max_retries:
                time.sleep(retry_delay)

    return JudgeOutput(
        classification="error",
        error_message=f"all {max_retries} attempts failed; last: {last_error}",
        judge_duration=round(duration, 2),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(df: pd.DataFrame) -> AggregateMetrics:
    m = AggregateMetrics()
    m.total = len(df)

    successful = df[df["classification"] != "error"]
    m.correct = int((successful["classification"] == "correct").sum())
    m.wrong = int((successful["classification"] == "wrong").sum())
    m.abstained = int((successful["classification"] == "abstained").sum())
    m.errors = m.total - len(successful)

    evaluated = m.correct + m.wrong
    m.accuracy = m.correct / evaluated if evaluated > 0 else 0.0

    total_evaluated = m.correct + m.wrong + m.abstained
    m.abstention_rate = m.abstained / total_evaluated if total_evaluated > 0 else 0.0

    def _rate(col: str) -> float:
        yes = int((successful[col] == "Yes").sum())
        no = int((successful[col] == "No").sum())
        return yes / (yes + no) if (yes + no) > 0 else 0.0

    m.faithfulness_rate = _rate("faithfulness")
    m.completeness_rate = _rate("completeness")
    m.transparency_rate = _rate("transparency")
    m.relevance_rate = _rate("relevance")

    m.total_input_tokens = int(df["judge_input_tokens"].sum())
    m.total_output_tokens = int(df["judge_output_tokens"].sum())
    m.total_duration = float(df["judge_duration"].sum())

    for cat in sorted(df["category"].dropna().unique()):
        cat_df = df[df["category"] == cat]
        cat_ok = cat_df[cat_df["classification"] != "error"]
        c = int((cat_ok["classification"] == "correct").sum())
        w = int((cat_ok["classification"] == "wrong").sum())
        a = int((cat_ok["classification"] == "abstained").sum())
        m.by_category[int(cat)] = {
            "count": int(len(cat_df)),
            "correct": c,
            "wrong": w,
            "abstained": a,
            "accuracy": c / (c + w) if (c + w) > 0 else 0.0,
        }

    return m


def print_report(m: AggregateMetrics) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal questions: {m.total}")
    print(f"  correct:   {m.correct}")
    print(f"  wrong:     {m.wrong}")
    print(f"  abstained: {m.abstained}")
    print(f"  errors:    {m.errors}")
    print(f"\nAccuracy (correct / (correct + wrong)): {m.accuracy:.2%}")
    print(f"Abstention rate: {m.abstention_rate:.2%}")
    print("\nCriterion-level rates (Yes / (Yes + No)):")
    print(f"  faithfulness:  {m.faithfulness_rate:.2%}")
    print(f"  completeness:  {m.completeness_rate:.2%}")
    print(f"  transparency:  {m.transparency_rate:.2%}")
    print(f"  relevance:     {m.relevance_rate:.2%}")
    if m.by_category:
        print("\nPer-category breakdown:")
        print(f"  {'category':<36} {'n':>5} {'acc':>8} {'ok':>5} {'no':>5} {'abs':>5}")
        print("  " + "-" * 65)
        for cat, stats in sorted(m.by_category.items()):
            name = CATEGORY_NAMES.get(cat, "?")
            label = f"Cat {cat} ({name})"
            print(
                f"  {label:<36} {stats['count']:>5} "
                f"{stats['accuracy']:>7.2%} {stats['correct']:>5} "
                f"{stats['wrong']:>5} {stats['abstained']:>5}"
            )
    tok = m.total_input_tokens + m.total_output_tokens
    print(f"\nJudge token usage: {m.total_input_tokens:,} in + {m.total_output_tokens:,} out = {tok:,}")
    print(f"Judge wall time:  {m.total_duration:.1f}s")
    print("=" * 70)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

JUDGE_COLUMNS = [
    "classification", "abstention",
    "faithfulness", "completeness", "transparency", "relevance",
    "justification", "error_message",
    "judge", "judge_duration", "judge_input_tokens", "judge_output_tokens",
]


def _append_row(csv_path: Path, row: dict) -> None:
    """Append one row, writing the header on the first write."""
    header = not csv_path.exists()
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", header=header, index=False)


def run_eval(
    results_csv: Path,
    judged_csv: Path,
    *,
    judge: str = "minimax",
    limit: int | None = None,
    resume: bool = False,
    max_retries: int = 3,
    retry_delay: float = 10.0,
) -> pd.DataFrame:
    if not results_csv.is_file():
        sys.exit(f"error: results file not found: {results_csv}")

    df = pd.read_csv(results_csv)
    required = {"question", "ground_truth", "predicted", "category"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"error: results.csv missing required columns: {sorted(missing)}")

    # Resume: skip rows whose question_id already appears in judged.csv.
    if resume and judged_csv.exists():
        already = pd.read_csv(judged_csv)
        if "question_id" in df.columns and "question_id" in already.columns:
            done = set(already["question_id"].astype(str))
            df = df[~df["question_id"].astype(str).isin(done)].reset_index(drop=True)
            print(f"> resume: skipping {len(already)} already-judged rows.")
        else:
            # Fall back to row-index resume.
            df = df.iloc[len(already):].reset_index(drop=True)
            print(f"> resume: skipping first {len(already)} rows (by index).")
    elif judged_csv.exists():
        print(f"> overwriting {judged_csv}")
        judged_csv.unlink()

    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)

    total = len(df)
    print(f"> judge: {judge}")
    print(f"> input: {results_csv}")
    print(f"> rows to judge: {total}")
    print(f"> output: {judged_csv}\n")

    for i, row in df.iterrows():
        predicted = row.get("predicted", "") or ""
        # Handle NaN / missing predicted answers as error rows -- do not waste a judge call.
        if not isinstance(predicted, str) or not predicted.strip():
            out = JudgeOutput(classification="error", error_message="empty predicted answer")
        else:
            try:
                category = int(row["category"])
            except (TypeError, ValueError):
                out = JudgeOutput(classification="error", error_message=f"invalid category: {row.get('category')}")
            else:
                print(f"[{i + 1}/{total}] q={str(row.get('question_id', i))[:10]:<10} cat={category}")
                out = judge_single(
                    question=str(row["question"]),
                    ground_truth=str(row["ground_truth"]),
                    predicted=predicted,
                    category=category,
                    judge=judge,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
                print(f"    -> {out.classification.upper()} ({out.judge_duration:.1f}s)")

        # Merge agent row with judge output and persist incrementally.
        merged = {**row.to_dict(), **asdict(out), "judge": judge}
        _append_row(judged_csv, merged)

    return pd.read_csv(judged_csv)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Path to a run directory (contains results.csv). judged.csv is written alongside.",
    )
    parser.add_argument("--judge", choices=JUDGE_CHOICES, default="minimax")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in judged.csv (matched by question_id).")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = _parse_args()

    run_dir: Path = args.run.expanduser().resolve()
    results_csv = run_dir / "results.csv"
    judged_csv = run_dir / "judged.csv"
    summary_json = run_dir / "judged_summary.json"

    df = run_eval(
        results_csv,
        judged_csv,
        judge=args.judge,
        limit=args.limit,
        resume=args.resume,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    metrics = calculate_metrics(df)
    print_report(metrics)

    summary_json.write_text(json.dumps({
        "run_dir": str(run_dir),
        "judge": args.judge,
        "metrics": asdict(metrics),
    }, indent=2))
    print(f"\n> summary: {summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
