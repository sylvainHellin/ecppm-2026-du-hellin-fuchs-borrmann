"""
Comparable agent-effort metrics derived from traces.json (same schema for all 4).
Needed because in the original result sets the CSV token columns were NOT comparable:
  - filesystem / graphdb : input/output_tokens = BAML JUDGE tokens (earlier inline-judge harness)
  - ifcopenshell / sql   : input/output_tokens = AGENT cumulative tokens
elapsed_s IS comparable (agent wall time) in all four.

Trace-based proxies (comparable across all four):
  - n_tool_calls : number of tool invocations the agent made
  - n_turns      : number of assistant messages
  - context_chars: total characters across all trace messages (proxy for how
                   much data the agent pulled into its context window)
"""
import glob, json, sys, statistics as st
import pandas as pd

# Input: compat exports from shared/eval/export_compat.py, run from the repo
# root. Traces require runs recorded with an on-disk traces.json (the
# filesystem/cypher run.py, or scripts/run_export.py for ifc/sql).
BASE = "results"


def _export_path(pipeline: str, pattern: str) -> str:
    hits = glob.glob(pattern)
    if not hits:
        sys.exit(
            f"error: no export found for the '{pipeline}' pipeline (expected {pattern}). "
            "Produce it with shared/eval/export_compat.py; see the README 'Evaluation and judging' section."
        )
    return hits[0]


TRACES = {
    "filesystem":   _export_path("filesystem", f"{BASE}/filesystem/*/export_compat/*_traces.json"),
    "graphdb":      _export_path("cypher", f"{BASE}/cypher/*/export_compat/*_traces.json"),
    "ifcopenshell": _export_path("ifc", f"{BASE}/ifc/*/export_compat/*_traces.json"),
    "sql":          _export_path("sql", f"{BASE}/sql/*/export_compat/*_traces.json"),
}
CSV = {
    "filesystem":   _export_path("filesystem", f"{BASE}/filesystem/*/export_compat/*.csv"),
    "graphdb":      _export_path("cypher", f"{BASE}/cypher/*/export_compat/*.csv"),
    "ifcopenshell": _export_path("ifc", f"{BASE}/ifc/*/export_compat/*.csv"),
    "sql":          _export_path("sql", f"{BASE}/sql/*/export_compat/*.csv"),
}
ORDER = ["filesystem", "graphdb", "ifcopenshell", "sql"]

# question -> category map (categories identical across runs)
catmap = {}
df = pd.read_csv(CSV["filesystem"])
for _, r in df.iterrows():
    catmap[str(r["question"])] = int(r["category"])

def metrics_for(path):
    with open(path) as f:
        d = json.load(f)
    rows = []
    for grp in d.values():
        for item in grp:
            tr = item.get("trace", [])
            n_calls = sum(len(e.get("tool_calls") or []) for e in tr
                          if isinstance(e, dict) and e.get("role") == "assistant")
            n_turns = sum(1 for e in tr if isinstance(e, dict) and e.get("role") == "assistant")
            chars = sum(len(str(e.get("content") or "")) for e in tr if isinstance(e, dict))
            rows.append({
                "question": item.get("question", ""),
                "cat": catmap.get(str(item.get("question", "")), 0),
                "n_calls": n_calls, "n_turns": n_turns, "chars": chars,
            })
    return pd.DataFrame(rows)

out = []
P = out.append
allm = {}
P("="*72)
P("AGENT EFFORT FROM TRACES (comparable proxy; tokens columns are NOT comparable)")
P("="*72)
P(f"{'approach':14s} {'n':>4s} {'calls_mean':>10s} {'calls_med':>9s} {'turns_med':>9s} {'chars_mean':>12s} {'chars_med':>11s}")
for k in ORDER:
    m = metrics_for(TRACES[k]); allm[k] = m
    P(f"{k:14s} {len(m):4d} {m['n_calls'].mean():10.1f} {m['n_calls'].median():9.0f} "
      f"{m['n_turns'].median():9.0f} {m['chars'].mean():12,.0f} {m['chars'].median():11,.0f}")

P("\nContext chars by category (mean):")
for cat in [1,2,3,4]:
    line = f"  cat{cat}: " + "  ".join(f"{k}={allm[k][allm[k]['cat']==cat]['chars'].mean():,.0f}" for k in ORDER)
    P(line)

P("\nTool calls by category (mean):")
for cat in [1,2,3,4]:
    line = f"  cat{cat}: " + "  ".join(f"{k}={allm[k][allm[k]['cat']==cat]['n_calls'].mean():.1f}" for k in ORDER)
    P(line)

# ratio of context chars vs filesystem baseline
P("\nContext-char ratio vs filesystem (mean):")
base = allm['filesystem']['chars'].mean()
for k in ORDER:
    P(f"  {k:14s} {allm[k]['chars'].mean()/base:.1f}x")

text = "\n".join(out)
print(text)
with open("analysis/trace_effort_output.txt","w") as f:
    f.write(text)
