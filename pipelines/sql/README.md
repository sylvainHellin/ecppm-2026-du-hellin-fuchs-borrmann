# SQL pipeline (SQLite)

The agent answers each ifc-bench question by querying a **SQLite conversion** of the IFC model with SQL (executed through a Python kernel). Test-split result: 378/507, 74.6%.

## Conversion

`convert.py` turns each IFC file into a SQLite database using [IfcOpenShell's `ifcpatch` `Ifc2Sql`](https://docs.ifcopenshell.org/) recipe, writing to `data/conversions/sql/<project>/<model>.sqlite`.

```bash
cd pipelines/sql
uv sync
uv run python convert.py            # convert all projects
uv run python convert.py --project <id>   # single project
uv run python convert.py --force          # re-convert existing
```

Geometry and inverse attributes are skipped to keep the databases lean. Property sets (psets) are requested, but if a model's pset conversion fails, `convert.py` retries **without psets** so the entity and relationship tables still load. This pset loss is the main reason SQL trails on property/MEP questions in the analysis.

## How it works

`agent.py` builds a deepagents agent with a single `python_exec` tool backed by a persistent Jupyter kernel (`interpreter.py`) that comes with a pre-connected `sqlite3` database (`conn` and `cursor` are ready to use). The system prompt (`system_prompt.jinja2` via `prompts.py`) embeds the full database schema so the agent starts informed.

- `RestrictToolsMiddleware` hides deepagents' built-in virtual-filesystem tools, leaving only `python_exec`, so the agent can't "discover" an empty in-memory FS and hallucinate.
- `RecursionGuardMiddleware` forces a final answer once the turn budget is spent.
- One agent per `(project, ifc_model)`; the kernel is `reset()` (which re-injects the DB connection) between questions.

## Run

```bash
uv run python run.py --split test
```

Shared flags: `--model`, `--split dev|test`, `--limit`, `--offset`, `--category 1..4`, `--project`, `--verbose`, `--resume <run_dir_name>`. Needs `IFC_BENCH_DIR` and the agent model's API key in the repo-root `.env`.

## Output

Writes to `results/sql/<model_tag>_<timestamp>/` (`results.csv` + `config.json`). Traces are captured by [Phoenix](https://docs.arize.com/phoenix) when running, or via `scripts/run_export.py` (Phoenix-free, also writes `traces.json`).

## Files

| File | Purpose |
|------|---------|
| `convert.py` | IFC → SQLite converter (`ifcpatch` `Ifc2Sql`) |
| `run.py` | CLI entry point |
| `agent.py` | DeepAgent factory + tool-restriction / recursion-guard middleware |
| `interpreter.py` | Jupyter kernel with a pre-connected SQLite database |
| `prompts.py` / `system_prompt.jinja2` | System prompt embedding the DB schema |

## Evaluation

```bash
uv run --project ../../shared/eval python ../../shared/eval/evaluate.py --run ../../results/sql/<run>
uv run --project ../../shared/eval python ../../shared/eval/export_compat.py --run ../../results/sql/<run>
```
