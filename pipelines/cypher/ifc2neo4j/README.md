# ifc2neo4j

Standalone IFC-to-Neo4j import module (extracted from the authors' ConMan2 codebase).  
Parses an IFC (STEP P21) file and stores the complete model structure as a labeled property graph in Neo4j — ready for Cypher queries.

## Graph Schema

### Node Labels

| Label | Source in IFC | Key Properties |
|---|---|---|
| `PrimaryNode` | `IfcObjectDefinition` + `IfcPropertyDefinition` | `GlobalId`, `EntityType`, `p21_id`, `timestamp` |
| `ConnectionNode` | `IfcRelationship` | `GlobalId`, `EntityType`, `p21_id`, `timestamp` |
| `SecondaryNode` | Everything else (geometry, placements, etc.) | `EntityType`, `p21_id`, `timestamp` |
| `InlineNode` | Anonymous inline entities (id=0) | `EntityType`, `wrappedValue`, `timestamp` |

All nodes also carry the shared labels `GenericNode` and `Node` for polymorphic queries.  
Primitive IFC attributes (strings, numbers, tuples-as-strings) are stored directly as node properties.

### Edge Type

All edges use the label `:rel` with two properties:

- `rel_type` — the IFC attribute name that caused this reference (e.g. `Representation`, `RelatingObject`)
- `list_index` — preserves ordering when the IFC attribute is a list

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from ifc2neo4j import Neo4jConnection, IfcToNeo4j

# 1. Connect (reads .env or falls back to localhost)
Neo4jConnection(password="my-secret")

# 2. Import an IFC file
IfcToNeo4j().run("path/to/model.ifc", timestamp="v1")
```

Then query freely in Neo4j Browser or from Python:

```cypher
-- Find all walls
MATCH (w:PrimaryNode {EntityType: 'IfcWall', timestamp: 'v1'})
RETURN w.GlobalId, w.Name;

-- Find what a wall is connected to
MATCH (w:PrimaryNode {EntityType: 'IfcWall'})-[:rel]->(related)
RETURN w.Name, r.rel_type, related.EntityType;

-- Spatial hierarchy
MATCH path = (p:PrimaryNode {EntityType: 'IfcProject'})-[:rel*]->(child:PrimaryNode)
WHERE ALL(r IN relationships(path) WHERE r.rel_type IN ['RelatingObject', 'RelatedObjects'])
RETURN [n IN nodes(path) | n.EntityType + ': ' + coalesce(n.Name, '')] AS hierarchy;
```

## Configuration

Copy `.env.example` to `.env` and fill in your credentials.  
Without a `.env` file the module connects to `bolt://neo4j:password@localhost:7687`.

## Files

```
ifc2neo4j/
├── __init__.py           # public API re-exports
├── ifc_to_graph.py       # IFC → Neo4j conversion logic
├── neo4j_connection.py   # connection setup (.env / localhost)
├── neo4j_helper.py       # batched Cypher execution
├── neo4j_model.py        # neomodel node & relationship schema
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT, same as the repository root `LICENSE`.
