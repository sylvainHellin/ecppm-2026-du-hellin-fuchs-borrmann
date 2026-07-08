# Querying BIM Models with LLM Agents: A Comparison of Data Representations

Companion code for the ECPPM 2026 paper by Changyu Du, Sylvain Hellin, Stefan Fuchs, and André Borrmann (Technical University of Munich).

## Overview

This repository compares how well an LLM agent extracts information from BIM models when the same IFC data is offered in four different representations, each paired with its natural query tool: the native IFC file queried through a Python code interpreter with IfcOpenShell, a SQLite conversion queried with SQL, a JSON filesystem tree explored with file-reading tools plus shell execution, and a Neo4j labeled property graph queried with Cypher.
Everything else is held constant: all four pipelines run the same LangChain deepagents harness (the `shared` package), answer the same ifc-bench v2 benchmark questions (~1000 QA pairs across 19 projects and 4 categories; evaluation uses the frozen test split over 10 projects), and are scored by the same BAML LLM-as-judge with MiniMax M2.7 at temperature zero.
The paper's headline numbers (1027 QA pairs, 507 test questions) refer to the frozen evaluation recorded in `analysis/data_representation_comparison_report.md`.

The judged runs are compared with paired statistical tests (McNemar, Cochran's Q) plus efficiency and trace-effort metrics; the resulting report and figures live in `analysis/`.

## Repository layout

| Path | Contents |
|------|----------|
| `shared/` | Common agent harness (`init_llm`, `answer_question`, Phoenix tracing) installed into each pipeline venv |
| `shared/eval/` | BAML LLM-as-judge (`evaluate.py`), suite-format exporter (`export_compat.py`), run comparison (`compare_runs.py`) |
| `pipelines/ifc/` | Native IFC pipeline: Python code interpreter with IfcOpenShell |
| `pipelines/sql/` | SQLite pipeline, including the IFC-to-SQLite converter (`convert.py`) |
| `pipelines/filesystem/` | Filesystem pipeline, including the IFC-to-filesystem converter (`ifc2fs.py`, `convert_all.py`) |
| `pipelines/cypher/` | Neo4j/Cypher pipeline, including the IFC-to-graph importer (`ifc2neo4j/`) and a Neo4j docker compose file |
| `analysis/` | Cross-pipeline statistics, the comparison report, and paper figure generation |
| `scripts/` | Dataset download, Phoenix-free run wrapper, result merging and re-judging helpers |
| `data/` | Benchmark questions and split definition; models and conversions land here (gitignored) |
| `results/` | One directory per run: `results.csv`, `judged.csv`, `judged_summary.json`, `config.json` |

Each pipeline has its own uv-managed virtual environment because their dependencies diverge.

## Setup

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker (only for the cypher pipeline).

```bash
# Copy the env template and fill in the API key(s) for the providers you use
cp .env.example .env

# Per-pipeline environment (repeat for sql, filesystem, cypher)
cd pipelines/ifc
uv sync
```

The default agent model is `minimax:MiniMax-M2.7`, which needs `MINIMAX_API_KEY`.
Other supported providers (Fireworks, Z.AI/GLM, xAI/Grok, OpenAI, Anthropic, Gemini) are listed in `.env.example`.

## Dataset

The benchmark is [ifc-bench](https://huggingface.co/datasets/sylvainHellin/ifc-bench), a public Hugging Face dataset (no token needed).
Run the download commands from the repo root:

```bash
# Questions CSV only (data/questions/ifc-bench-v2.csv)
uv run scripts/download_data.py

# Questions plus the IFC models (several GB), linked at data/ifc-bench/projects
uv run scripts/download_data.py --with-models
```

Pipelines that read IFC files resolve them as `$IFC_BENCH_DIR/<project>/<ifc_model>.ifc`.
The `.env.example` default `IFC_BENCH_DIR=data/ifc-bench/projects` matches the symlink created by `--with-models`; relative values are resolved against the repo root, so the default works no matter which directory you run from.

`data/questions/splits.json` defines a project-level dev/test/excluded split (4/10/5 projects).
Use `--split dev` while iterating and `--split test` for frozen evaluation runs.

## Running a pipeline

All four `run.py` entry points share the core flags `--model`, `--split dev|test`, `--limit`, `--offset`, `--category 1..4`, `--project`, `--verbose`, and `--resume <run_dir_name>`, and write to `results/<pipeline>/<model_tag>_<timestamp>/` (`results.csv` plus `config.json`).

### Native IFC (IfcOpenShell code interpreter)

```bash
cd pipelines/ifc
uv run python run.py --split test
```

### SQL (SQLite)

Convert the IFC models once, then run:

```bash
cd pipelines/sql
uv run python convert.py            # writes data/conversions/sql/<project>/<model>.sqlite
uv run python run.py --split test
```

`convert.py` accepts `--project <id>` to convert a single project and `--force` to re-convert.

### Filesystem (JSON tree)

Convert the IFC models once, then run:

```bash
cd pipelines/filesystem
uv run python convert_all.py        # writes data/conversions/ifc_filesys/<project>/<model>_fs/
uv run python run.py --split test
```

### Cypher (Neo4j graph)

Start Neo4j, then run; `run.py` truncates and re-imports the graph once per (project, model) group before answering that group's questions:

```bash
cd pipelines/cypher
docker compose -f docker-compose.neo4j.yml up -d
uv run python run.py --split test
```

Connection settings come from `NEO4J_BENCH_URI/USER/PASSWORD` in `.env` (defaults match the compose file) or the `--neo4j-*` flags.
Because the run is stateful, exactly one run may talk to a given Neo4j instance at a time.
The cypher pipeline also supports `--projects <id> ...` sharding (as does the filesystem pipeline) and `--retry-errors` on resume.

### Tracing

The ifc and sql pipelines export OTEL traces to a local [Phoenix](https://docs.arize.com/phoenix) server when one is running (`uv run phoenix serve` from a pipeline venv, with `PHOENIX_WORKING_DIR` from `.env`); `results.csv` rows carry a `span_id` linking to the trace.
The filesystem and cypher pipelines instead write `traces.json` (+ `.gz`) directly into the run directory.
`scripts/run_export.py` is a Phoenix-free runner for ifc and sql that also writes `traces.json`.

## Evaluation and judging

The shared BAML judge reads any pipeline's `results.csv` and writes `judged.csv` plus `judged_summary.json` alongside it:

```bash
cd shared/eval
uv sync
uv run python evaluate.py --run ../../results/<pipeline>/<run>
# Options: --judge minimax|gemini|gpt  --limit N  --resume
```

A question counts as correct only if the judge marks all four criteria (faithfulness, completeness, transparency, relevance) as Yes and the agent did not abstain.

Export the judged run into the flat suite CSV format the analysis scripts consume:

```bash
uv run python export_compat.py --run ../../results/<pipeline>/<run>
# writes <run>/export_compat/<name>.csv + <name>_summary.json (+ traces if present)
```

`scripts/run_all.sh --pipeline ifc|sql --split test` chains run, judge, and export in one Phoenix-free command.
`scripts/rejudge_errors.py` re-judges only the error-classified rows of a judged run, and `scripts/merge_results.py` merges the CSVs of sharded runs.

## Analysis

With one exported run per pipeline in place (`results/{ifc,sql,filesystem,cypher}/*/export_compat/`), run from the repo root:

```bash
uv run --project shared/eval python analysis/compare_representations.py
uv run --project shared/eval python analysis/trace_effort.py
uv run --project shared/eval python analysis/targeted.py
```

`analysis/data_representation_comparison_report.md` is the frozen comparison report behind the paper's numbers, and `analysis/make_paper_figures.py` regenerates the paper figures from it:

```bash
env -u VIRTUAL_ENV uv run --with matplotlib --with numpy python analysis/make_paper_figures.py
# writes PDFs to analysis/figures/
```

## Shipped results

`results/ifc/` (7 runs) and `results/sql/` (4 runs) ship the raw and judged outputs (`results.csv`, `judged.csv`, `judged_summary.json`, `config.json`) of the development runs for those two pipelines, all on the dev split.
No filesystem or cypher runs are included; those results were produced with an earlier harness and must be reproduced by re-running the pipelines and judging as described above.
The frozen 507-question test-split numbers behind the paper are recorded in `analysis/data_representation_comparison_report.md`.

## Citation

<!-- TODO: final citation once the proceedings are published -->
Citation: TBA (to appear in Proceedings of ECPPM 2026, Cardiff, UK, 9-11 September 2026).

## License

MIT, see [LICENSE](LICENSE).
