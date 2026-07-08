import glob, math, itertools, sys
import numpy as np, pandas as pd

# Input: compat exports from shared/eval/export_compat.py (one run per pipeline).
BASE="results"

def _export_path(pipeline, pattern):
    hits = glob.glob(pattern)
    if not hits:
        sys.exit(
            f"error: no export found for the '{pipeline}' pipeline (expected {pattern}). "
            "Produce it with shared/eval/export_compat.py; see the README 'Evaluation and judging' section."
        )
    return hits[0]

PATHS={"filesystem":_export_path("filesystem", f"{BASE}/filesystem/*/export_compat/*.csv"),
 "graphdb":_export_path("cypher", f"{BASE}/cypher/*/export_compat/*.csv"),
 "ifcopenshell":_export_path("ifc", f"{BASE}/ifc/*/export_compat/*.csv"),
 "sql":_export_path("sql", f"{BASE}/sql/*/export_compat/*.csv")}
ORDER=["filesystem","graphdb","ifcopenshell","sql"]

def chi2_sf1(stat):
    return math.erfc(math.sqrt(stat/2.0)) if stat>0 else 1.0
def mcnemar(b,c):
    n=b+c
    if n==0: return 0.0,1.0,1.0
    chi2=(abs(b-c)-1)**2/n
    k=min(b,c)
    pex=min(1.0,2*sum(math.comb(n,i) for i in range(k+1))*0.5**n)
    return chi2,chi2_sf1(chi2),pex

dfs={}
for k,p in PATHS.items():
    df=pd.read_csv(p)
    df["key"]=df["project"].astype(str)+"||"+df["ifc_model"].astype(str)+"||"+df["question"].astype(str)
    df=df.sort_values("key",kind="stable").reset_index(drop=True)
    df["occ"]=df.groupby("key").cumcount(); df["pk"]=df["key"]+"##"+df["occ"].astype(str)
    dfs[k]=df
base=dfs["filesystem"][["pk","category","project","question","ground_truth"]].copy()
m=base.copy()
for k in ORDER:
    m=m.merge(dfs[k][["pk","correct","abstention","classification"]].rename(
        columns={"correct":f"c_{k}","abstention":f"a_{k}","classification":f"cl_{k}"}),on="pk")

out=[];P=out.append

P("### Abstention by category")
for k in ORDER:
    line=f"  {k:14s} "
    for cat in [1,2,3,4]:
        sub=m[m["category"]==cat]
        line+=f"cat{cat}={int(sub[f'a_{k}'].sum())} "
    line+=f"| total={int(m[f'a_{k}'].sum())}"
    P(line)

P("\n### Category 3 paired McNemar (sql/ifcopenshell strong here)")
sub=m[m["category"]==3]
for a,b in itertools.combinations(ORDER,2):
    ca=sub[f"c_{a}"].values; cb=sub[f"c_{b}"].values
    b_=int(np.sum(ca&~cb)); c_=int(np.sum(~ca&cb))
    chi2,pc,pe=mcnemar(b_,c_)
    P(f"  {a:12s} vs {b:12s}: only_{a}={b_} only_{b}={c_} p_exact={pe:.3f}")

P("\n### Category 2 paired McNemar (largest category n=288)")
sub=m[m["category"]==2]
for a,b in itertools.combinations(ORDER,2):
    ca=sub[f"c_{a}"].values; cb=sub[f"c_{b}"].values
    b_=int(np.sum(ca&~cb)); c_=int(np.sum(~ca&cb))
    chi2,pc,pe=mcnemar(b_,c_)
    P(f"  {a:12s} vs {b:12s}: only_{a}={b_} only_{b}={c_} p_exact={pe:.3f}")

mat=np.column_stack([m[f"c_{k}"].astype(int) for k in ORDER])
nright=mat.sum(axis=1)
P("\n### CONSENSUS-HARD questions (0/4 correct), n=%d"%int((nright==0).sum()))
hard=m[nright==0]
for _,r in hard.iterrows():
    q=str(r["question"])[:95].replace("\n"," ")
    P(f"  [cat{r['category']}|{r['project']}] {q}")

P("\n### Per-project: which approach wins each project (paired)")
for proj in sorted(m["project"].unique()):
    sub=m[m["project"]==proj]
    accs={k:sub[f"c_{k}"].mean() for k in ORDER}
    best=max(accs,key=accs.get); worst=min(accs,key=accs.get)
    P(f"  {proj:26s} n={len(sub):3d} best={best}({accs[best]*100:.0f}%) worst={worst}({accs[worst]*100:.0f}%)")

# correlation of per-question difficulty across approaches (phi / agreement)
P("\n### Pairwise agreement rate (fraction of questions with same correct/wrong outcome)")
for a,b in itertools.combinations(ORDER,2):
    agree=(m[f"c_{a}"]==m[f"c_{b}"]).mean()
    P(f"  {a:12s} vs {b:12s}: {agree*100:.1f}%")

text="\n".join(out); print(text)
open("analysis/targeted_output.txt","w").write(text)
