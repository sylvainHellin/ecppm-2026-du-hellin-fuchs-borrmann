# Cypher pipeline (Neo4j graph)

The agent answers each ifc-bench question by querying the IFC model stored as a **Neo4j labeled property graph** with read-only Cypher. Test-split result: 383/507, 75.5%.

## Import (ifc2neo4j)

The IFC file is imported into Neo4j by [`ifc2neo4j/`](ifc2neo4j/README.md), a graph importer **adapted from [ConMan2](https://github.com/seb-esser/ConMan2)** (Sebastian Esser et al., TUM). Every IFC entity becomes a node (`PrimaryNode` / `ConnectionNode` / `SecondaryNode` / `InlineNode`), and every IFC reference becomes a `:rel` edge carrying the originating attribute name. See the [ifc2neo4j README](ifc2neo4j/README.md) for the full graph schema and attribution.

## How it works

The run is **stateful**: questions are grouped by `(project, ifc_model)`, and for each group `run.py` truncates the shared Neo4j database and re-imports that model before answering the group's questions. Because of this, **exactly one run may talk to a given Neo4j instance at a time** — use separate instances (or sequential runs) for parallelism.

`agent.py` builds a deepagents agent with a single `execute_cypher` tool. Queries run inside a Neo4j **read transaction**, so any write (`CREATE`, `SET`, `DELETE`, ...) is rejected at the database level. Deepagents' built-in filesystem tools are stripped; results are truncated to the first 500 rows.

## Run

```bash
cd pipelines/cypher
uv sync
docker compose -f docker-compose.neo4j.yml up -d
uv run python run.py --split test
```

Connection settings come from `NEO4J_BENCH_URI` / `NEO4J_BENCH_USER` / `NEO4J_BENCH_PASSWORD` in the repo-root `.env` (defaults match the compose file: `bolt://localhost:7687`, `neo4j` / `bench_admin`) or the `--neo4j-uri/--neo4j-user/--neo4j-password` flags. Also needs `IFC_BENCH_DIR` and the agent model's API key.

Shared flags plus `--projects <id> ...` (sharding) and `--retry-errors` (with `--resume`, re-runs previously errored questions).

## Output

Writes to `results/cypher/<model_tag>_<timestamp>/`: `results.csv`, `config.json`, and `traces.json` (+ `.gz`) written directly (no Phoenix). Traces are grouped by `<project>/<ifc_model>`.

## Files

| File | Purpose |
|------|---------|
| `ifc2neo4j/` | IFC → Neo4j graph importer (adapted from ConMan2) |
| `docker-compose.neo4j.yml` | Local Neo4j instance for the pipeline |
| `run.py` | CLI entry point: truncate + import per model group, then answer |
| `agent.py` | DeepAgent factory with the read-only `execute_cypher` tool |
| `system_prompt.jinja2` | System prompt with the graph schema and answer-format rules |

## Evaluation

```bash
uv run --project ../../shared/eval python ../../shared/eval/evaluate.py --run ../../results/cypher/<run>
uv run --project ../../shared/eval python ../../shared/eval/export_compat.py --run ../../results/cypher/<run>
```
