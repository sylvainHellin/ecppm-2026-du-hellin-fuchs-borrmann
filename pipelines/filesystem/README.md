# Filesystem pipeline (JSON tree)

The agent answers each ifc-bench question by exploring a **JSON filesystem tree** exported from the IFC model, using read-only file tools (`ls`, `read_file`, `glob`, `grep`) plus shell execution. Test-split result: 387/507, 76.3%.

## Conversion

`ifc2fs.py` maps the IFC spatial hierarchy to directories and semantic data to JSON files, so an agent can navigate a BIM model with ordinary CLI tools. Layout:

```
<output>/
├── __meta__/            header.json, units.json, project.json
├── __types__/<IfcType>/ one JSON per type definition
├── __materials__/       material and layer-set definitions
├── __systems__/         MEP / distribution systems
└── Site__<name>/Building__<name>/<Storey>/
    ├── __storey__.json     elevation, element summary
    ├── __geometry__.json   all geometry for the storey, keyed by element id
    ├── spaces/             one JSON per room
    └── <IfcClass>/         one JSON per element (props, type, material, openings, geometry_key, ...)
```

Each element JSON carries a `geometry_key` pointing into the storey-level `__geometry__.json`. Relationships (openings/hosts, space boundaries, wall connections, systems, ports, coverings, services) are resolved and embedded during conversion.

```bash
cd pipelines/filesystem
uv sync
uv run python convert_all.py        # all (project, ifc_model) pairs → data/conversions/ifc_filesys/<project>/<model>_fs/
uv run python convert_all.py --force

# single file (ad hoc)
uv run python ifc2fs.py path/to/model.ifc -o out_fs
```

`audit_ifc2fs.py` sanity-checks a conversion against the source IFC.

## How it works

`agent.py` builds a deepagents agent over the converted directory. Built-in write tools (`write_file`, `edit_file`) are removed so the agent is read-only; it computes with `execute('python3 -c "..."')`. The system prompt instructs it to navigate with `ls`/`glob`/`grep`/`read_file` and aggregate via shell/Python. Because the data is spread across many small files, this representation needs the most tool calls of the four (≈20/question).

## Run

```bash
uv run python run.py --split test
```

Shared flags plus `--projects <id> ...` for sharding. Needs `IFC_BENCH_DIR` and the agent model's API key in the repo-root `.env`.

## Output

Writes to `results/filesystem/<model_tag>_<timestamp>/`: `results.csv`, `config.json`, and `traces.json` (+ `.gz`) written directly (this pipeline does not use Phoenix).

## Files

| File | Purpose |
|------|---------|
| `ifc2fs.py` | IFC → JSON filesystem converter |
| `convert_all.py` | Batch-convert every model referenced in the question set |
| `audit_ifc2fs.py` | Verify a conversion against the source IFC |
| `run.py` | CLI entry point |
| `agent.py` | DeepAgent factory (read-only filesystem tools + shell) |
| `system_prompt.jinja2` | System prompt with navigation and answer-format rules |

## Evaluation

```bash
uv run --project ../../shared/eval python ../../shared/eval/evaluate.py --run ../../results/filesystem/<run>
uv run --project ../../shared/eval python ../../shared/eval/export_compat.py --run ../../results/filesystem/<run>
```
