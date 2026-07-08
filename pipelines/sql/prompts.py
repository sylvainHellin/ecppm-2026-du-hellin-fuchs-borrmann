"""Render the SQL system prompt from its Jinja template.

Schema introspection runs at render time: table names, columns, row counts
are read from the SQLite database and injected into the template as
``{{ schema_info }}``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from jinja2 import Template

_TEMPLATE_PATH = Path(__file__).parent / "system_prompt.jinja2"

# Tables that are geometry primitives, representation internals, or bookkeeping.
# The agent should discover these via relationships if needed, not scan them directly.
_SCHEMA_EXCLUDE_PREFIXES = (
    "IfcCartesian", "IfcDirection", "IfcAxis", "IfcLocalPlacement",
    "IfcExtrudedAreaSolid", "IfcRevolvedAreaSolid", "IfcSweptDisk",
    "IfcFace", "IfcPolyLoop", "IfcPolyline", "IfcPolygonalBounded",
    "IfcShape", "IfcProductDefinitionShape", "IfcRepresentationMap",
    "IfcMappedItem", "IfcStyledItem", "IfcStyledRepresentation",
    "IfcColour", "IfcSurfaceStyle", "IfcSurfaceStyleRendering",
    "IfcPresentation", "IfcCurveStyle", "IfcFillArea",
    "IfcClosedShell", "IfcOpenShell", "IfcShellBased", "IfcConnectedFaceSet",
    "IfcBooleanClipping", "IfcHalfSpace", "IfcPlane",
    "IfcArbitrary", "IfcCircleProfileDef", "IfcCircleHollowProfileDef",
    "IfcRectangleProfileDef", "IfcRectangleHollowProfileDef",
    "IfcCompositeCurve", "IfcCompositeCurveSegment",
    "IfcTrimmedCurve", "IfcCircle", "IfcLine",
    "IfcFaceBasedSurfaceModel", "IfcFaceBound", "IfcFaceOuterBound",
    "IfcGeometricRepresentation", "IfcGeometricSet",
    "IfcSurfaceOfLinearExtrusion", "IfcTriangulatedFaceSet",
    "IfcDimensionalExponents", "IfcDraughtingPreDefined",
    "IfcConnectionSurfaceGeometry",
    "id_map",
)


def _introspect_schema(db_path: str) -> str:
    """Return a compact text description of every table in the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]

    lines: list[str] = []
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM [{table}]")
        count = cur.fetchone()[0]
        if count == 0:
            continue
        if table.startswith(_SCHEMA_EXCLUDE_PREFIXES):
            continue

        cur.execute(f"PRAGMA table_info([{table}])")
        cols = cur.fetchall()
        col_descs = [f"{c[1]} ({c[2]})" for c in cols]

        lines.append(f"  {table} ({count} rows): {', '.join(col_descs)}")

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='psets'")
    if cur.fetchone() is None:
        lines.append("\n  NOTE: This database has no `psets` table. Property lookups must use "
                     "raw IfcPropertySet/IfcPropertySingleValue tables instead.")

    conn.close()
    return "\n".join(lines)


def render_sql_prompt(db_path: str) -> str:
    """Render the SQL system prompt with schema info for the given database."""
    schema_info = _introspect_schema(db_path)
    template = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.render(db_path=db_path, schema_info=schema_info)
