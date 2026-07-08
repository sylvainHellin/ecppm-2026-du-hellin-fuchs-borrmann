"""
Paired statistical comparison of 4 BIM data representations / query tools.

Approaches (same 507 questions, same agent MiniMax-M2.7, same BAML judge):
  - filesystem   : IFC exported to a JSON file-system tree (grep/read tools)
  - graphdb      : IFC loaded into Neo4j, queried with Cypher
  - ifcopenshell : raw IFC queried with IfcOpenShell in a Python kernel
  - sql          : IFC converted to SQLite, queried with SQL

No scipy available -> all tests implemented from scratch (numpy + math only).
"""
import glob
import math
import itertools
import json
import sys
import numpy as np
import pandas as pd

# Input: per-pipeline compat exports produced by shared/eval/export_compat.py
# (run from the repo root: results/<pipeline>/<run>/export_compat/<name>.csv).
# Each glob expects exactly one exported run per pipeline.
BASE = "results"


def _export_path(pipeline: str, pattern: str) -> str:
    hits = glob.glob(pattern)
    if not hits:
        sys.exit(
            f"error: no export found for the '{pipeline}' pipeline (expected {pattern}). "
            "Produce it with shared/eval/export_compat.py; see the README 'Evaluation and judging' section."
        )
    return hits[0]


PATHS = {
    "filesystem":   _export_path("filesystem", f"{BASE}/filesystem/*/export_compat/*.csv"),
    "graphdb":      _export_path("cypher", f"{BASE}/cypher/*/export_compat/*.csv"),
    "ifcopenshell": _export_path("ifc", f"{BASE}/ifc/*/export_compat/*.csv"),
    "sql":          _export_path("sql", f"{BASE}/sql/*/export_compat/*.csv"),
}
ORDER = ["filesystem", "graphdb", "ifcopenshell", "sql"]
CAT_NAMES = {
    1: "Direct Info Retrieval",
    2: "Computational Aggregation",
    3: "Geometric/Spatial Computation",
    4: "Incomplete Information",
}

# ----------------------------------------------------------------------------
# statistics helpers (no scipy)
# ----------------------------------------------------------------------------
def _gammln(xx):
    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
    x = xx; y = xx
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1.0
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)

def _gser(a, x):
    gln = _gammln(a)
    if x <= 0:
        return 0.0, gln
    ap = a; s = 1.0 / a; delta = s
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * 1e-14:
            break
    return s * math.exp(-x + a * math.log(x) - gln), gln

def _gcf(a, x):
    gln = _gammln(a)
    FPMIN = 1e-300
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h, gln

def gammq(a, x):
    """Regularized upper incomplete gamma Q(a,x)=1-P(a,x)."""
    if x < 0 or a <= 0:
        raise ValueError
    if x < a + 1.0:
        p, _ = _gser(a, x)
        return 1.0 - p
    q, _ = _gcf(a, x)
    return q

def chi2_sf(stat, df):
    """Survival function of chi-square (p-value)."""
    if stat <= 0:
        return 1.0
    return gammq(df / 2.0, stat / 2.0)

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)

def mcnemar(b, c):
    """b = #(A correct, B wrong); c = #(A wrong, B correct).
    Returns (chi2 with continuity correction, p_chi2, p_exact_binomial)."""
    n = b + c
    if n == 0:
        return 0.0, 1.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    p_chi2 = chi2_sf(chi2, 1)
    # exact two-sided binomial, p=0.5
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p_exact = min(1.0, 2 * tail)
    return chi2, p_chi2, p_exact

def cochran_q(mat):
    """mat: N x k binary array of related samples. Returns (Q, df, p)."""
    N, k = mat.shape
    col = mat.sum(axis=0).astype(float)      # successes per treatment
    row = mat.sum(axis=1).astype(float)      # successes per subject
    G = col.sum()
    num = (k - 1) * (k * np.sum(col ** 2) - G ** 2)
    den = k * G - np.sum(row ** 2)
    if den == 0:
        return 0.0, k - 1, 1.0
    Q = num / den
    df = k - 1
    return Q, df, chi2_sf(Q, df)

# ----------------------------------------------------------------------------
# load + pair
# ----------------------------------------------------------------------------
def load():
    dfs = {}
    for k, p in PATHS.items():
        df = pd.read_csv(p)
        df["key"] = (df["project"].astype(str) + "||" + df["ifc_model"].astype(str)
                     + "||" + df["question"].astype(str))
        # stable occurrence index so the one duplicated key still pairs 1:1
        df = df.sort_values("key", kind="stable").reset_index(drop=True)
        df["occ"] = df.groupby("key").cumcount()
        df["pair_key"] = df["key"] + "##" + df["occ"].astype(str)
        dfs[k] = df
    return dfs

def build_paired(dfs):
    cols = ["pair_key", "category", "project", "correct", "abstention",
            "faithfulness", "completeness", "transparency", "relevance",
            "input_tokens", "output_tokens", "elapsed_s", "classification"]
    merged = None
    for k in ORDER:
        sub = dfs[k][cols].copy()
        ren = {c: f"{c}__{k}" for c in cols if c not in ("pair_key", "category", "project")}
        sub = sub.rename(columns=ren)
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub.drop(columns=["category", "project"]),
                                  on="pair_key", how="inner")
    return merged

# ----------------------------------------------------------------------------
def fmt_ci(k, n):
    lo, hi = wilson_ci(k, n)
    return f"{k/n*100:.1f}% [{lo*100:.1f},{hi*100:.1f}]"

def main():
    dfs = load()
    m = build_paired(dfs)
    N = len(m)
    out = []
    P = out.append
    P(f"PAIRED ROWS: {N}\n")

    # ---- overall ----
    P("="*70); P("OVERALL ACCURACY (correct / 507)"); P("="*70)
    overall = {}
    for k in ORDER:
        c = int(m[f"correct__{k}"].sum())
        overall[k] = c
        P(f"  {k:14s} {c:3d}/{N}  acc={fmt_ci(c, N)}")
    # abstention
    P("\nABSTENTION counts:")
    for k in ORDER:
        a = int(m[f"abstention__{k}"].sum())
        P(f"  {k:14s} {a}  ({a/N*100:.2f}%)")

    # accuracy on evaluated-only (excluding abstained)
    P("\nACCURACY excluding abstained (correct / non-abstained):")
    for k in ORDER:
        nonabs = (~m[f"abstention__{k}"]).sum()
        c = int(m[f"correct__{k}"].sum())
        P(f"  {k:14s} {c}/{nonabs} = {fmt_ci(c, int(nonabs))}")

    # ---- per category ----
    P("\n" + "="*70); P("ACCURACY BY CATEGORY"); P("="*70)
    for cat in [1, 2, 3, 4]:
        sub = m[m["category"] == cat]
        n = len(sub)
        P(f"\nCat {cat} ({CAT_NAMES[cat]}) n={n}")
        for k in ORDER:
            c = int(sub[f"correct__{k}"].sum())
            P(f"  {k:14s} {fmt_ci(c, n)}")

    # ---- per criterion (over evaluated/non-Na rows) ----
    P("\n" + "="*70); P("CRITERION PASS RATES (Yes / (Yes+No), Na excluded)"); P("="*70)
    crit_data = {}
    for crit in ["faithfulness", "completeness", "transparency", "relevance"]:
        P(f"\n{crit}")
        crit_data[crit] = {}
        for k in ORDER:
            col = m[f"{crit}__{k}"]
            yes = (col == "Yes").sum()
            no = (col == "No").sum()
            tot = yes + no
            crit_data[crit][k] = (int(yes), int(tot))
            P(f"  {k:14s} {fmt_ci(int(yes), int(tot))}  (Yes={yes} No={no} Na={(col=='Na').sum()})")

    # ---- failure-mode breakdown among WRONG answers ----
    P("\n" + "="*70); P("FAILURE MODE among non-correct rows: which criterion fails"); P("="*70)
    for k in ORDER:
        wrong = m[~m[f"correct__{k}"] & ~m[f"abstention__{k}"]]
        nf = len(wrong)
        if nf == 0: continue
        fails = {c: int((wrong[f"{c}__{k}"] == "No").sum()) for c in
                 ["faithfulness", "completeness", "transparency", "relevance"]}
        P(f"  {k:14s} wrong={nf}  faith_No={fails['faithfulness']} compl_No={fails['completeness']} transp_No={fails['transparency']} relev_No={fails['relevance']}")

    # ---- paired McNemar between all pairs ----
    P("\n" + "="*70); P("PAIRWISE McNEMAR (accuracy, paired by question)"); P("="*70)
    P("  b = row-approach correct & col wrong ; c = row wrong & col correct")
    for a, b in itertools.combinations(ORDER, 2):
        ca = m[f"correct__{a}"].values
        cb = m[f"correct__{b}"].values
        b_ = int(np.sum(ca & ~cb))   # a right, b wrong
        c_ = int(np.sum(~ca & cb))   # a wrong, b right
        both = int(np.sum(ca & cb))
        neither = int(np.sum(~ca & ~cb))
        chi2, pchi, pex = mcnemar(b_, c_)
        sig = "***" if pex < 0.001 else "**" if pex < 0.01 else "*" if pex < 0.05 else "ns"
        P(f"  {a:12s} vs {b:12s}: only_{a}={b_:3d} only_{b}={c_:3d} both={both} neither={neither} | chi2={chi2:.2f} p_exact={pex:.4f} {sig}")

    # ---- Cochran's Q across all 4 ----
    P("\n" + "="*70); P("COCHRAN'S Q (are the 4 approaches equal in accuracy?)"); P("="*70)
    mat = np.column_stack([m[f"correct__{k}"].astype(int).values for k in ORDER])
    Q, df, p = cochran_q(mat)
    P(f"  Q={Q:.3f}  df={df}  p={p:.4f}")
    # per-category Cochran Q
    P("  by category:")
    for cat in [1, 2, 3, 4]:
        sub = m[m["category"] == cat]
        matc = np.column_stack([sub[f"correct__{k}"].astype(int).values for k in ORDER])
        Qc, dfc, pc = cochran_q(matc)
        P(f"    cat{cat}: Q={Qc:.3f} df={dfc} p={pc:.4f} (n={len(sub)})")

    # ---- agreement structure ----
    P("\n" + "="*70); P("QUESTION-LEVEL AGREEMENT (how many of 4 got it right)"); P("="*70)
    nright = mat.sum(axis=1)
    for v in range(5):
        cnt = int(np.sum(nright == v))
        P(f"  {v}/4 correct: {cnt:3d}  ({cnt/N*100:.1f}%)")
    P(f"  -> consensus-correct (4/4): {int(np.sum(nright==4))}")
    P(f"  -> consensus-hard (0/4):    {int(np.sum(nright==0))}")

    # unique strengths: only this approach correct
    P("\nUNIQUE STRENGTH (only this approach correct, other 3 wrong):")
    for i, k in enumerate(ORDER):
        only = np.sum((mat[:, i] == 1) & (mat.sum(axis=1) == 1))
        P(f"  {k:14s} {int(only)}")
    P("\nUNIQUE WEAKNESS (only this approach wrong, other 3 correct):")
    for i, k in enumerate(ORDER):
        only = np.sum((mat[:, i] == 0) & (mat.sum(axis=1) == 3))
        P(f"  {k:14s} {int(only)}")

    # consensus-hard breakdown by category
    P("\nCONSENSUS-HARD (0/4) breakdown by category:")
    hard = m[nright == 0]
    for cat in [1, 2, 3, 4]:
        P(f"  cat{cat}: {int((hard['category']==cat).sum())}")
    P("\nCONSENSUS-HARD by project:")
    for proj, cnt in hard["project"].value_counts().items():
        P(f"  {proj:28s} {cnt}")

    # ---- efficiency: tokens + latency ----
    P("\n" + "="*70); P("EFFICIENCY (per-question medians/means)"); P("="*70)
    for k in ORDER:
        it = m[f"input_tokens__{k}"]
        ot = m[f"output_tokens__{k}"]
        el = m[f"elapsed_s__{k}"]
        P(f"  {k:14s} in_tok: mean={it.mean():,.0f} median={it.median():,.0f} sum={it.sum():,.0f} | "
          f"out_tok: mean={ot.mean():,.0f} median={ot.median():,.0f} | elapsed: mean={el.mean():.1f}s median={el.median():.1f}s")

    # token by category for ifcopenshell vs filesystem (context blow-up)
    P("\nInput tokens by category (mean):")
    for cat in [1,2,3,4]:
        sub = m[m["category"]==cat]
        row = "  cat%d: " % cat + " ".join(f"{k}={sub[f'input_tokens__{k}'].mean():,.0f}" for k in ORDER)
        P(row)

    # ---- per-project accuracy matrix ----
    P("\n" + "="*70); P("ACCURACY BY PROJECT"); P("="*70)
    projs = sorted(m["project"].unique())
    P(f"  {'project':28s} " + " ".join(f"{k[:6]:>7s}" for k in ORDER) + "   n")
    for proj in projs:
        sub = m[m["project"]==proj]
        n=len(sub)
        cells=[]
        for k in ORDER:
            cells.append(f"{sub[f'correct__{k}'].mean()*100:6.1f}%")
        P(f"  {proj:28s} " + " ".join(f"{c:>7s}" for c in cells) + f"  {n}")

    text = "\n".join(out)
    print(text)
    with open("analysis/stats_output.txt", "w") as f:
        f.write(text)

    # also dump machine-readable
    res = {
        "N": N,
        "overall_correct": overall,
        "abstention": {k: int(m[f"abstention__{k}"].sum()) for k in ORDER},
    }
    with open("analysis/stats_summary.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
