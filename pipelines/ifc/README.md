# Native IFC pipeline (IfcOpenShell code interpreter)

The agent answers each ifc-bench question by writing and running Python against the **native IFC file** — no conversion step. This is the strongest representation in the paper's evaluation (399/507, 78.7% on the test split).

## How it works

`agent.py` builds a [deepagents](https://github.com/langchain-ai/deepagents) agent with a single `python_exec` tool backed by a persistent Jupyter kernel (`interpreter.py`). The system prompt (`system_prompt.jinja2`, rendered by `prompts.py`) tells the agent to open the model with `ifcopenshell.open(<ifc_path>)` and explore it iteratively; variables, imports, and loaded data persist across tool calls within a single question.

- `RecursionGuardMiddleware` counts AI turns and, once the budget is exhausted, strips the tools and asks the agent to write its final answer, preventing runaway loops.
- One agent (and kernel) is cached per `(project, ifc_model)` pair; the kernel is `reset()` between questions for a clean namespace, and shut down at the end of the run.
- Answers are sanitized to strip MiniMax `<think>` blocks and hallucinated tool-call XML before saving.

## Run

```bash
cd pipelines/ifc
uv sync
uv run python run.py --split test
```

Key flags (shared across pipelines): `--model`, `--split dev|test`, `--limit`, `--offset`, `--category 1..4`, `--project`, `--verbose`, `--resume <run_dir_name>`. Requires `IFC_BENCH_DIR` (IFC models) and the agent model's API key (default `minimax:MiniMax-M2.7` → `MINIMAX_API_KEY`) in the repo-root `.env`.

## Output

Writes to `results/ifc/<model_tag>_<timestamp>/`:
- `results.csv` — one row per question (answer, tokens, tool calls, elapsed, `span_id`)
- `config.json` — run metadata

Full execution traces are captured by [Phoenix](https://docs.arize.com/phoenix) when a server is running (`uv run phoenix serve`); each `results.csv` row carries a `span_id` linking to its trace. `scripts/run_export.py` is a Phoenix-free alternative that also writes `traces.json`.

## Files

| File | Purpose |
|------|---------|
| `run.py` | CLI entry point: load questions, run the agent, write `results.csv` |
| `agent.py` | DeepAgent factory + recursion-guard middleware |
| `interpreter.py` | Persistent Jupyter kernel backing `python_exec` |
| `prompts.py` / `system_prompt.jinja2` | System prompt with the IFC path and answer-format rules |

## Evaluation

Judge and export with the shared BAML judge:

```bash
uv run --project ../../shared/eval python ../../shared/eval/evaluate.py --run ../../results/ifc/<run>
uv run --project ../../shared/eval python ../../shared/eval/export_compat.py --run ../../results/ifc/<run>
```
