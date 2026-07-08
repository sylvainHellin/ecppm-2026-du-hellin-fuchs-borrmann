# BIM Data Representation × Query Tool Comparison Report

> A paired statistical analysis of the experimental results for four IFC data representations / query tools on IFC-Bench v2.
> Data source: suite-format result CSVs of the paper's four runs (`ifc_filesystem`, `ifc_graphdb`, `ifc_ifcopenshell`, `ifc_sql`); in this repository the same format is produced by `shared/eval/export_compat.py` under `results/<pipeline>/<run>/export_compat/`.
> Analysis scripts: `analysis/compare_representations.py`, `analysis/trace_effort.py`, `analysis/targeted.py`

---

## 1. Experimental setup (a single controlled-variable experiment)

The four experiments **vary only one variable -- "data representation + query tool"** -- everything else is identical:

| Dimension | Setting |
|------|------|
| Agent model | `minimax:MiniMax-M2.7` (identical across all four) |
| Evaluation judge | BAML LLM-judge (`baml:minimax`, MiniMax-M2.7, temperature 0) |
| Question set | IFC-Bench v2, **507 questions** (test split, 10 projects) |
| Agent framework | The same DeepAgent scaffold (`shared` package) |

The four "data representation + query tool" combinations being compared:

| Code | Data representation | Query tool used by the agent |
|------|----------|----------------------|
| **filesystem** | IFC exported as a JSON filesystem tree | `grep` / `read` file-reading tools |
| **graphdb** | IFC imported into a Neo4j graph database | Cypher queries |
| **ifcopenshell** | Native IFC file | Python kernel + IfcOpenShell API |
| **sql** | IFC converted to SQLite | SQL queries |

**Paired nature**: all four ran the same batch of 507 questions (verified question-by-question: 506 unique question keys are exactly identical, with 1 duplicate question paired by occurrence index). This allows **paired statistical tests** (McNemar, Cochran's Q), which have far higher statistical power than independent-sample comparisons.

**Scoring definition** (from the BAML judge, `shared/eval/evaluate.py`):
- A question is judged **correct** if and only if: it was not abstained (abstention=False) **and all four dimensions are Yes** (faithfulness ∧ completeness ∧ transparency ∧ relevance).
- Any dimension being No → **wrong**; an explicit refusal → **abstained**.
- Meaning of the four dimensions: **faithfulness** (are all claims backed by valid data = correctness / no fabrication), **completeness** (does it include all the facts needed to answer), **transparency** (does it clearly disclose sources/methods), **relevance** (does it directly answer the question).

---

## 2. A data-quality issue that must be stated up front (the token columns are not cross-group comparable)

Before drawing any "cost/efficiency" conclusions, it must be noted: **the `input_tokens` / `output_tokens` columns in the four CSVs do not measure the same thing**:

- `filesystem` / `graphdb` (original runs made with an earlier inline-judge harness): these two columns record the **BAML judge's tokens**, ≈2,350/question -- **not the agent's tokens**.
- `ifcopenshell` / `sql` (from this repo's pipeline, exported via `export_compat.py`): these two columns record the **agent's cumulative tokens** (summing the prompt tokens over every LLM call), ≈140K–200K/question.

> Confirmed by reading the earlier harness's inline-judge evaluate script (which wrote `baml_result.input_tokens`) and `shared/eval/export_compat.py` (which writes the agent's `input_tokens`).

**Conclusion: directly comparing the token columns across the four groups yields a spurious "60×" gap, which is an artifact of differing measurement definitions and is unusable.** Therefore the efficiency analysis in this report instead uses two **cross-group comparable** metrics:
1. `elapsed_s` (agent wall-clock time, consistent across all four groups);
2. The **number of tool calls** and **trace context character count** reconstructed from `traces.json` (the trace schema is consistent across all four groups).

---

## 3. Overall accuracy: the four representations show **no statistically significant difference**

| Representation | Correct | Accuracy (95% Wilson CI) | Abstained | Rank |
|------|--------|------------------------|------|------|
| **ifcopenshell** | 399/507 | **78.7%** [74.9, 82.0] | 1 | 1 |
| filesystem | 387/507 | 76.3% [72.4, 79.8] | 5 | 2 |
| graphdb | 383/507 | 75.5% [71.6, 79.1] | 8 | 3 |
| sql | 378/507 | 74.6% [70.6, 78.2] | 4 | 4 |

- **The range is only 4.1 percentage points** (78.7% vs 74.6%), and all 95% confidence intervals overlap heavily.
- **Cochran's Q test** (whether the four paired groups have equal accuracy): **Q = 3.55, df = 3, p = 0.314 → not significant**.
- **All 6 pairwise McNemar tests are non-significant** (the closest to significant is ifcopenshell vs sql: ifcopenshell-only correct on 66 questions, sql-only correct on 45, χ²=3.60, **p = 0.057**, still above 0.05).

> **Key observation 1: at the scale of 507 questions, "which data representation / query tool you use" has no statistically significant effect on overall accuracy.** Model capability and the inherent difficulty of the questions determine success more than the representation. IfcOpenShell is nominally highest (78.7%), but this lead cannot be confirmed statistically.

---

## 4. Comparison by question category

Category definitions (IFC-Bench taxonomy, with the number of questions per category in this test split):

| Cat | Name | n | filesystem | graphdb | ifcopenshell | sql |
|-----|------|------|-----------|---------|--------------|-----|
| 1 | Direct Info Retrieval | 65 | 83.1% | 83.1% | **86.2%** | 83.1% |
| 2 | Computational Aggregation | 288 | 74.0% | 72.6% | **74.7%** | 70.5% |
| 3 | Geometric/Spatial | 62 | 75.8% | 80.6% | 87.1% | **88.7%** |
| 4 | Incomplete Information | 92 | 79.3% | 76.1% | **80.4%** | 71.7% |

Cochran's Q tests within each category are **all non-significant** (cat1 p=0.93, cat2 p=0.56, cat3 p=0.17, cat4 p=0.37), i.e. the within-category differences also fail to reach statistical significance (limited by sample size, especially cat3 with only 62 questions). However, the following **trends** are worth noting:

- **Category 2 (Computational Aggregation) is the biggest weakness across all representations** (70.5%–74.7%), and it makes up more than half the question set (288/507). **Overall accuracy is essentially determined by cat2.**
- **Category 3 (Geometric/Spatial) shows the largest divergence between representations**: sql (88.7%) and ifcopenshell (87.1%) are clearly higher than filesystem (75.8%). Paired McNemar: filesystem vs sql **p=0.077**, filesystem vs ifcopenshell p=0.118 (close but not significant, n=62 lacks power).
  - **Interpretation**: geometric/spatial calculations require pulling out large numbers of elements for numerical computation, so **programmable/aggregatable representations (Python+IfcOpenShell, SQL) have a natural advantage**; whereas flattening the data into a file tree and stitching it together with grep/read is at a disadvantage for geometric computation.
- **In Category 4 (Incomplete Information), sql is clearly the worst (71.7%)**, while ifcopenshell is the best (80.4%). SQL conversion loses some semantics (especially MEP psets, see §8), which makes it more likely to produce an unfaithful guess rather than appropriately acknowledging a limitation when facing questions where "the information actually isn't in the model."

---

## 5. Comparison by scoring dimension (criteria): faithfulness is the common bottleneck across all representations

Pass rate per dimension (Yes / (Yes+No), with abstained Na excluded):

| Dimension | filesystem | graphdb | ifcopenshell | sql | Range across groups |
|------|-----------|---------|--------------|-----|----------|
| **faithfulness** | 82.1% | 82.1% | **84.2%** | 79.1% | 79–84% ← lowest |
| completeness | 90.8% | 90.3% | **93.2%** | 90.8% | 90–93% |
| transparency | 93.6% | 93.8% | **96.0%** | 94.2% | 94–96% |
| relevance | 96.6% | 96.4% | 96.6% | 95.8% | 96–97% ← highest |

> **Key observation 2: the dimension ordering is identical across all four representations: faithfulness < completeness < transparency < relevance.** In other words, the agent almost always produces a "relevant, transparent, seemingly complete" answer, and **what really determines correctness is faithfulness (whether the numbers/facts are actually right)**.

**Failure attribution** (count of No per dimension among the wrong questions):

| Representation | Wrong | faithfulness=No | completeness=No | transparency=No | relevance=No |
|------|----------|-----------------|-----------------|-----------------|--------------|
| filesystem | 115 | **90 (78%)** | 46 | 32 | 17 |
| graphdb | 116 | **89 (77%)** | 48 | 31 | 18 |
| ifcopenshell | 107 | **80 (75%)** | 34 | 20 | 17 |
| sql | 125 | **105 (84%)** | 46 | 29 | 21 |

- **75%–84% of errors come with a faithfulness failure** -- i.e. the answer went wrong on "did it count right, did it pull the right value," rather than being off-topic or opaque.
- **sql has the worst faithfulness (79.1%, 84% of its errors due to unfaithfulness)**, which directly echoes its overall last-place finish and its poor cat4 performance: the data loss from SQL conversion makes it easier for the model to produce plausible-looking but actually incorrect numbers.
- **ifcopenshell is best or tied for best on all four dimensions**, leading especially on completeness (93.2%) and transparency (96.0%) -- properties read directly through the native API are more complete and have clearer provenance.

---

## 6. Efficiency comparison (using cross-group comparable metrics)

| Representation | Tool calls (mean/median) | Agent wall-clock time (mean/median) | Trace context chars (mean) |
|------|----------------------------|------------------------------|------------------------|
| **ifcopenshell** | **8.8 / 8** (fewest) | **73.2s / 58.1s** (fastest) | 26,329 |
| sql | 12.3 / 10 | 105.6s / 77.8s | **23,951** (fewest) |
| filesystem | 20.4 / 18 | 81.4s / 63.9s | 45,875 (most) |
| graphdb | 23.3 / 21 (most) | **129.7s / 102.6s** (slowest) | 42,378 |

> **Key observation 3: the "expressiveness" of the query tool determines how many steps the agent has to iterate.** Tools where "one query can aggregate" -- like IfcOpenShell / SQL -- need only **8.8 / 12.3** steps on average to solve a question; whereas filesystem (which needs grep+read across many small files) and graphdb (which needs repeated schema probing and iterative Cypher) need **20+** steps.

- **graphdb is the least efficient**: most calls (23.3), slowest (130s), and still no higher accuracy to show for it -- the Cypher route spends a lot of overhead on "exploring the schema + repeated trial and error."
- **ifcopenshell is the best all-around**: fewest calls, fastest, highest accuracy.
- **Tool calls rise with category**: all representations need more steps on cat3/cat4 (e.g. graphdb reaches 29.8 calls on cat4), confirming that geometric and incomplete-information questions really are more convoluted.
- Regarding tokens: on a comparable basis, **sql's cumulative agent input tokens (≈200K/question) are about 1.4× higher than ifcopenshell's (≈139K/question)** (both are agent-cumulative, hence comparable); the agent tokens for filesystem/graphdb were not recorded (the CSV holds judge tokens, see §2) and cannot be included in the comparison.

---

## 7. Question-level: consistency, unique advantages, and "all-fail" questions

Counting, for each question, how many of the four representations answered it correctly:

| Representations correct | n | Share |
|---------------|------|------|
| 4/4 (all correct) | 240 | 47.3% |
| 3/4 | 137 | 27.0% |
| 2/4 | 63 | 12.4% |
| 1/4 | 50 | 9.9% |
| **0/4 (all wrong)** | **17** | **3.4%** |

- **47.3% of questions are answered correctly by all four representations, and only 3.4% are missed by all four** -- again confirming that success is mostly determined by the question and the model, not the representation. **The middle ~half (49.3%) of questions have disagreement among representations**, and this is exactly the space where representation choice can make a difference.
- **Unique advantage** (only this representation got it right, the other three all wrong): graphdb **20** questions (most) > filesystem 12 > ifcopenshell 10 > sql 8.
- **Unique weakness** (only this representation got it wrong, the other three all right): graphdb **43** (most) > sql 30 > filesystem 38 > ifcopenshell **26** (fewest).
  - graphdb has both the most "unique advantages" and the most "unique weaknesses" → **highest variance, least stable**; ifcopenshell has the fewest unique weaknesses → **most robust**.
- **Pairwise agreement rate**: ifcopenshell vs sql is highest (**78.1%**), filesystem vs graphdb is lower (70.4%).
  - **Interpretation**: ifcopenshell and sql both belong to the "compute over structured data" paradigm, so they also make the most similar mistakes; filesystem and graphdb, although both "navigational query" paradigms, have tool differences that actually make their error distributions less consistent.

### The 17 "all-fail" questions (all 4 representations wrong) -- the most valuable diagnostic clue

| Distribution | Result |
|------|------|
| By category | **cat2: 13**, cat1: 2, cat4: 2, cat3: 0 |
| By project | **wbdg_office: 10**, digital_hub: 3, duplex: 3, ettenheim_gis: 1 |

The vast majority of these 17 questions are **MEP / system-class aggregation questions**, for example:
- "The types and counts of duct fittings"
- "Air flow of HVAC terminals, including total and average"
- "Fans classified by size and system"
- "Apparent electrical load breakdown of lighting fixtures by type / voltage level and load distribution"
- "Count of heating-system components by system type"

> **Key observation 4: these 17 "all-fail" questions are not a representation problem but a data/model problem.** Since four completely different representations all got them wrong, the root cause is very likely: (a) the `wbdg_office` MEP-dense office model itself has defects or ambiguities in how its MEP information is modeled/exported; (b) the ground truth for these MEP system-aggregation questions is hard to reproduce strictly from the model. **This is the batch of questions most in need of manual review going forward.**

---

## 8. Comparison by project

| Project | n | filesystem | graphdb | ifcopenshell | sql | Hardest/easiest |
|------|---|-----------|---------|--------------|-----|-----------|
| wbdg_office | 81 | 59.3% | **71.6%** | 69.1% | 63.0% | **hardest in the set** |
| ettenheim_gis | 21 | 66.7% | 61.9% | 57.1% | **71.4%** | second hardest |
| 4351 | 44 | 77.3% | 77.3% | 75.0% | 75.0% | |
| ac20 | 48 | **85.4%** | 83.3% | 79.2% | 68.8% | sql crashes here |
| digital_hub | 102 | **82.4%** | 74.5% | 74.5% | 70.6% | |
| duplex | 69 | 75.4% | 76.8% | **87.0%** | 79.7% | |
| fantasy_hotel_1 | 41 | 82.9% | 70.7% | **85.4%** | 82.9% | |
| fantasy_office_building_1 | 35 | 85.7% | 74.3% | **88.6%** | 77.1% | |
| fantasy_office_building_2 | 31 | 80.6% | 87.1% | **93.5%** | 83.9% | |
| fantasy_office_building_3 | 35 | 71.4% | 77.1% | 82.9% | **91.4%** | |

- **`wbdg_office` is the universally hardest project** (59–72%), and it contributes 10/17 of the "all-fail" questions -- MEP-dense models are a challenge for all representations (and the diagnostic focus of §7). Notably, graphdb is actually best here (71.6%) and filesystem worst (59.3%).
- **The best representation differs by project**: ifcopenshell is best on 4 projects, filesystem on 3, sql on 2, graphdb on 1. **No single representation dominates across all projects**, further supporting the "no statistically significant overall difference" conclusion.
- **sql is clearly low on ac20 (68.8%) and digital_hub (70.6%)**, echoing its faithfulness weakness and MEP/property conversion loss (see CLAUDE.md: pset conversion fails for some MEP models, the SQL database lacks a `psets` table, so property-based questions cannot be answered).

---

## 9. Abstention behavior

| Representation | Total abstained | cat1 | cat2 | cat3 | cat4 |
|------|--------|------|------|------|------|
| graphdb | **8** (most) | 0 | 1 | 3 | 4 |
| filesystem | 5 | 1 | 1 | 1 | 2 |
| sql | 4 | 0 | 0 | 3 | 1 |
| ifcopenshell | **1** (almost never abstains) | 0 | 1 | 0 | 0 |

- **graphdb is the most prone to "giving up"**, concentrated in cat3 (geometric) and cat4 (incomplete information) -- when it hits geometric computations that are hard to express in Cypher, or information that is simply missing from the model, it is more inclined to refuse outright.
- **ifcopenshell almost never abstains (only once)**: the native API can almost always retrieve data and produce an answer. This is a double-edged sword -- not abstaining while having the highest faithfulness means it "dares to answer and answers accurately."
- **sql abstains 3 times in cat3**: when the data needed for geometric computation is incomplete in the converted tables, the SQL route chooses to refuse.

---

## 10. Summary of key conclusions

1. **Representation choice has no statistically significant effect on overall accuracy** (Cochran Q p=0.31, all pairwise McNemar non-significant). The range across the four representations on 507 questions is only 4.1pp, with heavily overlapping CIs. **"Switching the database / switching the query tool" is not a lever for improving accuracy.**
2. **IfcOpenShell (native API + Python) is the best all-around choice**: nominally highest accuracy (78.7%), best or tied-best on all four scoring dimensions, fewest tool calls (8.8 steps), fastest (73s), almost never abstains, and fewest unique weaknesses (most robust).
3. **faithfulness is the common bottleneck across all representations**: the dimension ordering is invariably faithfulness < completeness < transparency < relevance; 75–84% of errors are due to faithfulness failures. **To improve accuracy, target "numerical/factual correctness" (better data retrieval, unit conversion, aggregation logic, self-checking), not the representation.**
4. **Query-tool expressiveness mainly affects efficiency, not accuracy**: programmable/aggregatable representations (IfcOpenShell, SQL) solve questions in 1/2~1/3 the steps; navigational representations (filesystem grep/read, graphdb Cypher) take many steps, and graphdb is also the slowest.
5. **Capability differences are concentrated in geometric/spatial questions (cat3)**: sql (88.7%) and ifcopenshell (87.1%) clearly outperform filesystem (75.8%) -- geometric computation favors programmable aggregatable representations (a clear trend, but n=62 does not reach statistical significance, p≈0.08).
6. **The cost of the SQL route**: last in overall accuracy, worst faithfulness, weakest cat4, largest token cost -- rooted in the semantic/property loss of the IFC→SQLite conversion (especially MEP psets).
7. **The top target for manual review**: the 17 "all-fail" questions (13 in cat2, 10 in wbdg_office, almost all MEP system aggregation) -- failed by all four representations simultaneously.

## 11. Limitations and caveats

- **Sample size**: 507 questions, 10 projects, a single agent model (MiniMax-M2.7). The conclusions hold for "this model + this data"; a stronger model could amplify the differences between representations. Increasing the number of test questions may reveal new findings.
- **Imbalanced category samples**: cat3 has only 62 questions and cat1 only 65, so within-category tests have limited power; the several p≈0.08 trends need larger samples to confirm.
- A single run, without repeated runs to estimate the agent's stochastic variance (the run-to-run jitter from temperature is not quantified).

---

### Scripts used to reproduce the experimental data

- `analysis/compare_representations.py`: overall/category/dimension/project accuracy, Wilson CI, McNemar, Cochran's Q, consistency structure
- `analysis/trace_effort.py`: comparable efficiency metrics reconstructed from traces.json (tool-call count, context characters)
- `analysis/targeted.py`: abstention distribution, within-category paired tests, all-fail question list, per-project winners
- Intermediate outputs: `analysis/stats_output.txt`, `analysis/trace_effort_output.txt`, `analysis/targeted_output.txt`
