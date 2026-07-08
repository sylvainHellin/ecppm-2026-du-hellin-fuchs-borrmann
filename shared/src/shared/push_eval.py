"""Push evaluation results from judged.csv back to Phoenix as span annotations.

Reads a run directory's ``judged.csv`` (must have ``span_id`` and
``classification`` columns) and creates annotations visible as filterable
columns in the Phoenix UI.

Annotations created per span:
  - ``eval`` -- label = classification (correct / wrong / abstained / error)
  - ``eval_faithfulness`` -- label + score (Yes=1.0, No=0.0)
  - ``eval_completeness`` -- label + score
  - ``eval_transparency`` -- label + score
  - ``eval_relevance`` -- label + score

Usage (from any pipeline venv that has ``shared`` installed):
    uv run python -m shared.push_eval --run results/ifc/<run_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from phoenix.client import Client
from phoenix.client.resources.spans import SpanAnnotationData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CRITERIA = ("faithfulness", "completeness", "transparency", "relevance")


def _criterion_score(value: str | float | None) -> float | None:
    """Map a criterion value (Yes/No/NaN) to a numeric score."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if s == "yes":
        return 1.0
    if s == "no":
        return 0.0
    return None


def build_annotations(df: pd.DataFrame) -> list[SpanAnnotationData]:
    """Build a flat list of SpanAnnotationData dicts from a judged DataFrame."""
    annotations: list[SpanAnnotationData] = []

    for _, row in df.iterrows():
        raw_span_id = row.get("span_id", "")
        if pd.isna(raw_span_id):
            continue
        span_id = str(raw_span_id).strip()
        if not span_id:
            continue

        classification = str(row.get("classification", "")).strip()
        if not classification:
            continue

        # Primary annotation: overall classification
        annotations.append(SpanAnnotationData(
            span_id=span_id,
            name="eval",
            annotator_kind="CODE",
            result={"label": classification},
        ))

        # Per-criterion annotations
        for criterion in _CRITERIA:
            value = row.get(criterion)
            label = str(value).strip() if pd.notna(value) else None
            score = _criterion_score(value)
            if label is None and score is None:
                continue
            result: dict = {}
            if label is not None:
                result["label"] = label
            if score is not None:
                result["score"] = score
            annotations.append(SpanAnnotationData(
                span_id=span_id,
                name=f"eval_{criterion}",
                annotator_kind="CODE",
                result=result,
            ))

    return annotations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push eval classifications to Phoenix as span annotations.",
    )
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Path to a run directory containing judged.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run.expanduser().resolve()
    judged_csv = run_dir / "judged.csv"

    if not judged_csv.is_file():
        print(f"error: judged.csv not found at {judged_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(judged_csv)
    required = {"span_id", "classification"}
    missing = required - set(df.columns)
    if missing:
        print(
            f"error: judged.csv is missing required columns: {sorted(missing)}",
            file=sys.stderr,
        )
        return 1

    annotations = build_annotations(df)
    if not annotations:
        print("no annotations to push (all span_id values are empty).")
        return 0

    # Count how many unique spans we are annotating
    span_ids = {a["span_id"] for a in annotations}
    print(f"> pushing {len(annotations)} annotations for {len(span_ids)} spans ...")

    client = Client()
    client.spans.log_span_annotations(span_annotations=annotations, sync=True)

    print(f"> done -- {len(annotations)} annotations logged to Phoenix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
