"""Compare two judged.csv eval runs side-by-side.

Usage:
    uv run python compare_runs.py --baseline <dir> --new <dir> [--out <path>]

Produces:
- Per-question delta table (improved / regressed / unchanged)
- Aggregate accuracy comparison (overall, per-category, per-project)
- Criterion-level comparison
- Writes results to --out (default: stdout)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_judged(path: Path) -> dict[str, dict]:
    """Load judged.csv into {question_id: row}."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["question_id"]] = row
    return rows


def accuracy(rows: dict[str, dict]) -> dict:
    total = len(rows)
    correct = sum(1 for r in rows.values() if r["classification"] == "correct")
    wrong = sum(1 for r in rows.values() if r["classification"] == "wrong")
    abstained = sum(1 for r in rows.values() if r["classification"] == "abstained")
    error = total - correct - wrong - abstained
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "error": error,
        "accuracy": correct / total if total else 0,
    }


def group_accuracy(rows: dict[str, dict], key: str) -> dict[str, dict]:
    groups: dict[str, dict[str, dict]] = {}
    for r in rows.values():
        g = r[key]
        groups.setdefault(g, {})[r["question_id"]] = r
    return {g: accuracy(sub) for g, sub in sorted(groups.items())}


def criterion_rate(rows: dict[str, dict], criterion: str) -> float:
    yes = sum(1 for r in rows.values() if r.get(criterion, "").strip().lower() == "yes")
    rated = sum(1 for r in rows.values() if r.get(criterion, "").strip().lower() in ("yes", "no"))
    return yes / rated if rated else 0


def main():
    parser = argparse.ArgumentParser(description="Compare two eval runs")
    parser.add_argument("--baseline", required=True, help="Baseline run directory")
    parser.add_argument("--new", required=True, help="New run directory")
    parser.add_argument("--out", default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    base_dir = Path(args.baseline)
    new_dir = Path(args.new)
    base = load_judged(base_dir / "judged.csv")
    new = load_judged(new_dir / "judged.csv")

    # Align on shared question IDs
    shared_ids = sorted(set(base) & set(new), key=int)
    only_base = sorted(set(base) - set(new), key=int)
    only_new = sorted(set(new) - set(base), key=int)

    # Per-question deltas
    improved = []  # was wrong/abstained, now correct
    regressed = []  # was correct, now wrong/abstained
    unchanged_correct = []
    unchanged_wrong = []
    abstention_changes = []

    for qid in shared_ids:
        b_cls = base[qid]["classification"]
        n_cls = new[qid]["classification"]
        if b_cls != "correct" and n_cls == "correct":
            improved.append(qid)
        elif b_cls == "correct" and n_cls != "correct":
            regressed.append(qid)
        elif b_cls == "correct" and n_cls == "correct":
            unchanged_correct.append(qid)
        else:
            unchanged_wrong.append(qid)
        # Track abstention changes
        if b_cls == "abstained" and n_cls != "abstained":
            abstention_changes.append((qid, "resolved"))
        elif b_cls != "abstained" and n_cls == "abstained":
            abstention_changes.append((qid, "new_abstention"))

    lines = []

    def w(s=""):
        lines.append(s)

    w(f"# Eval Run Comparison")
    w()
    w(f"Baseline: `{base_dir.name}`")
    w(f"New:      `{new_dir.name}`")
    w(f"Shared questions: {len(shared_ids)}")
    if only_base:
        w(f"Only in baseline: {len(only_base)} ({', '.join(only_base[:5])}{'...' if len(only_base)>5 else ''})")
    if only_new:
        w(f"Only in new: {len(only_new)} ({', '.join(only_new[:5])}{'...' if len(only_new)>5 else ''})")
    w()

    # Overall accuracy
    ba = accuracy(base)
    na = accuracy(new)
    delta = na["accuracy"] - ba["accuracy"]
    w("## Overall accuracy")
    w()
    w(f"| Metric | Baseline | New | Delta |")
    w(f"|---|---|---|---|")
    w(f"| Accuracy | {ba['accuracy']:.1%} | {na['accuracy']:.1%} | {delta:+.1%} |")
    w(f"| Correct | {ba['correct']} | {na['correct']} | {na['correct']-ba['correct']:+d} |")
    w(f"| Wrong | {ba['wrong']} | {na['wrong']} | {na['wrong']-ba['wrong']:+d} |")
    w(f"| Abstained | {ba['abstained']} | {na['abstained']} | {na['abstained']-ba['abstained']:+d} |")
    w()

    # Per-category
    w("## Per-category accuracy")
    w()
    bg = group_accuracy(base, "category")
    ng = group_accuracy(new, "category")
    w("| Category | Baseline | New | Delta |")
    w("|---|---|---|---|")
    for cat in sorted(set(bg) | set(ng)):
        b_acc = bg.get(cat, {}).get("accuracy", 0)
        n_acc = ng.get(cat, {}).get("accuracy", 0)
        b_n = bg.get(cat, {}).get("total", 0)
        w(f"| {cat} (n={b_n}) | {b_acc:.1%} | {n_acc:.1%} | {n_acc - b_acc:+.1%} |")
    w()

    # Per-project
    w("## Per-project accuracy")
    w()
    bg = group_accuracy(base, "project")
    ng = group_accuracy(new, "project")
    w("| Project | Baseline | New | Delta |")
    w("|---|---|---|---|")
    for proj in sorted(set(bg) | set(ng)):
        b_acc = bg.get(proj, {}).get("accuracy", 0)
        n_acc = ng.get(proj, {}).get("accuracy", 0)
        b_n = bg.get(proj, {}).get("total", 0)
        w(f"| {proj} (n={b_n}) | {b_acc:.1%} | {n_acc:.1%} | {n_acc - b_acc:+.1%} |")
    w()

    # Criteria
    w("## Criterion-level rates")
    w()
    criteria = ["faithfulness", "completeness", "transparency", "relevance"]
    w("| Criterion | Baseline | New | Delta |")
    w("|---|---|---|---|")
    for c in criteria:
        br = criterion_rate(base, c)
        nr = criterion_rate(new, c)
        w(f"| {c} | {br:.1%} | {nr:.1%} | {nr - br:+.1%} |")
    w()

    # Movement summary
    w("## Question-level movements")
    w()
    w(f"- Improved (wrong/abstained -> correct): **{len(improved)}**")
    w(f"- Regressed (correct -> wrong/abstained): **{len(regressed)}**")
    w(f"- Unchanged correct: {len(unchanged_correct)}")
    w(f"- Unchanged wrong/abstained: {len(unchanged_wrong)}")
    w(f"- Net gain: **{len(improved) - len(regressed):+d}**")
    w()

    # Improved details
    if improved:
        w("### Improved questions")
        w()
        w("| QID | Project | Category | Old | New | Old transparency | New transparency |")
        w("|---|---|---|---|---|---|---|")
        for qid in improved:
            b = base[qid]
            n = new[qid]
            w(f"| {qid} | {b['project']} | {b['category']} | {b['classification']} | {n['classification']} | {b.get('transparency','')} | {n.get('transparency','')} |")
        w()

    # Regressed details
    if regressed:
        w("### Regressed questions")
        w()
        w("| QID | Project | Category | Old | New | Criteria failed |")
        w("|---|---|---|---|---|---|")
        for qid in regressed:
            b = base[qid]
            n = new[qid]
            failed = [c for c in criteria if n.get(c, "").strip().lower() == "no"]
            w(f"| {qid} | {b['project']} | {b['category']} | {b['classification']} | {n['classification']} | {', '.join(failed)} |")
        w()

    # Abstention changes
    if abstention_changes:
        w("### Abstention changes")
        w()
        w("| QID | Project | Direction | New classification |")
        w("|---|---|---|---|")
        for qid, direction in abstention_changes:
            n = new[qid]
            w(f"| {qid} | {n['project']} | {direction} | {n['classification']} |")
        w()

    output = "\n".join(lines)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Comparison written to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
