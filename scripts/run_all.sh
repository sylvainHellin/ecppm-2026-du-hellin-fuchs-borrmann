#!/usr/bin/env bash
#
# Phoenix-free one-shot: run the agent, judge it, and export the suite-format
# CSV + summary + traces -- all in one command. Chains the three steps, each in
# its own uv environment (pipeline venv for the agent, eval venv for BAML).
#
# This does NOT touch run.py / run_eval_push_trace.fish; it is a separate path
# that bypasses Phoenix entirely.
#
# Usage:
#   scripts/run_all.sh --pipeline ifc --split test
#   scripts/run_all.sh --pipeline ifc --out results/ifc_export --split test
#   scripts/run_all.sh --pipeline sql --limit 5 --name v2_minimax_sql
#
# Recognised flags (consumed by this script):
#   --pipeline ifc|sql   (required) which pipeline to run
#   --out <dir>          (optional) export dir for step 3 (default: <run>/export_compat)
#   --name <name>        (optional) base filename for step 3 exports
# All other flags (--split, --limit, --model, --category, --project, ...) are
# forwarded to scripts/run_export.py.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

pipeline=""
out=""
name=""
forward=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline) pipeline="$2"; forward+=("--pipeline" "$2"); shift 2 ;;
        --out)      out="$2"; shift 2 ;;
        --name)     name="$2"; shift 2 ;;
        *)          forward+=("$1"); shift ;;
    esac
done

if [[ -z "$pipeline" ]]; then
    echo "error: --pipeline ifc|sql is required" >&2
    exit 1
fi

# ── Step 1: run the agent (pipeline venv, no Phoenix) ───────────────────
echo "[1/3] Running $pipeline agent (Phoenix bypassed) ..."
run_log="$(mktemp)"
uv run --project "$repo_root/pipelines/$pipeline" python -u "$repo_root/scripts/run_export.py" \
    "${forward[@]}" 2>&1 | tee "$run_log"

run_dir="$(sed -n 's/^RUN_DIR=//p' "$run_log" | tail -1)"
rm -f "$run_log"

if [[ -z "$run_dir" ]]; then
    echo "error: could not detect run directory (no RUN_DIR= line from run_export.py)." >&2
    exit 1
fi
echo ""
echo "Run directory: $run_dir"

# ── Step 2: judge (eval venv) ───────────────────────────────────────────
echo ""
echo "[2/3] Evaluating with BAML judge ..."
uv run --project "$repo_root/shared/eval" python "$repo_root/shared/eval/evaluate.py" --run "$run_dir"

# ── Step 3: export suite-format CSV + summary + traces (eval venv) ───────
echo ""
echo "[3/3] Exporting suite-format results ..."
export_args=(--run "$run_dir")
[[ -n "$out"  ]] && export_args+=(--out "$out")
[[ -n "$name" ]] && export_args+=(--name "$name")
uv run --project "$repo_root/shared/eval" python "$repo_root/shared/eval/export_compat.py" "${export_args[@]}"

echo ""
echo "All done. Phoenix-free results in: $run_dir"
[[ -n "$out" ]] && echo "Suite-format export in: $out"
