#!/usr/bin/env fish
#
# Run the IFC pipeline, evaluate the results, and push eval annotations to Phoenix.
#
# All arguments are forwarded to run.py.  The script detects the output
# directory from the last line of run.py's stdout ("done -- results saved to <dir>").
#
# Requires a local Phoenix server for trace capture and the annotation push in
# step 3 (see the README section on tracing). For a Phoenix-free alternative
# use scripts/run_all.sh.
#
# Usage:
#     ./run_eval_push_trace.fish --split dev
#     ./run_eval_push_trace.fish --model gemini:gemini-2.5-pro --limit 5

set script_dir (status dirname)
set repo_root (realpath "$script_dir/../..")

# ── Step 1: run the agent ──────────────────────────────────────────────
echo "[1/3] Running IFC agent ..."
set run_output (uv run python "$script_dir/run.py" $argv)
set run_status $status

# Print all output so the user sees progress
for line in $run_output
    echo $line
end

if test $run_status -ne 0
    echo "run.py failed (exit $run_status) -- aborting."
    exit $run_status
end

# Extract run directory from the "done -- results saved to <path>" line.
set run_dir ""
for line in $run_output
    if string match -q "done -- results saved to *" $line
        set run_dir (string replace "done -- results saved to " "" $line)
    end
end

if test -z "$run_dir"
    echo "error: could not detect run directory from run.py output."
    exit 1
end

echo ""
echo "Run directory: $run_dir"

# ── Step 2: evaluate ──────────────────────────────────────────────────
echo ""
echo "[2/3] Evaluating results ..."
cd "$repo_root/shared/eval"
uv run python evaluate.py --run "$run_dir"
set eval_status $status

if test $eval_status -ne 0
    echo "evaluate.py failed (exit $eval_status) -- skipping annotation push."
    exit $eval_status
end

# ── Step 3: push annotations to Phoenix ──────────────────────────────
echo ""
echo "[3/3] Pushing eval annotations to Phoenix ..."
cd "$script_dir"
uv run python -m shared.push_eval --run "$run_dir"
set push_status $status

if test $push_status -ne 0
    echo "push_eval failed (exit $push_status)."
    exit $push_status
end

echo ""
echo "All done. Results, judgements, and Phoenix annotations in: $run_dir"
