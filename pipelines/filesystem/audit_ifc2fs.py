#!/usr/bin/env python3
"""
Audit ifc2fs.py across all IFC models.

Three passes:
  Pass 1: Use IFC schema introspection to list ALL possible IfcRelationship
          subtypes, then check which ones actually appear in our models and
          whether the converter handles them.
  Pass 2: Check non-IfcRelationship entities that carry topological info
          (e.g. IfcPresentationLayerAssignment, IfcClassification, etc.).
  Pass 3: Per-model correctness checks (containment, relationship maps,
          hierarchy completeness).

Usage:
    uv run python audit_ifc2fs.py              # full audit
    uv run python audit_ifc2fs.py --rels-only  # only Pass 1 + Pass 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

import ifcopenshell
import ifcopenshell.util.element as element_util

from ifc2fs import (
    _build_host_map,
    _build_layer_map,
    _build_port_map,
    _build_space_boundary_map,
    _build_system_map,
    _build_wall_connection_map,
)

# Source IFC models: $IFC_BENCH_DIR if set (see scripts/download_data.py),
# otherwise data/models/<project>/<ifc_model>.ifc at the repo root.
_bench_dir = os.environ.get("IFC_BENCH_DIR")
if _bench_dir:
    PROJECTS = Path(_bench_dir).expanduser()
    if not PROJECTS.is_absolute():
        PROJECTS = _REPO_ROOT / PROJECTS  # relative paths anchor at the repo root
    PROJECTS = PROJECTS.resolve()
else:
    PROJECTS = _REPO_ROOT / "data" / "models"
SKIP_CLASSES = {
    "IfcOpeningElement", "IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey",
}

# Relationship types the converter explicitly handles
HANDLED_RELS = {
    "IfcRelContainedInSpatialStructure",
    "IfcRelAggregates",
    "IfcRelNests",
    "IfcRelVoidsElement",
    "IfcRelFillsElement",
    "IfcRelSpaceBoundary",
    "IfcRelSpaceBoundary1stLevel",
    "IfcRelSpaceBoundary2ndLevel",
    "IfcRelConnectsPathElements",
    "IfcRelAssignsToGroup",
    "IfcRelConnectsPorts",
    "IfcRelServicesBuildings",
    "IfcRelAssociatesMaterial",
    "IfcRelDefinesByProperties",
    "IfcRelDefinesByType",
    "IfcRelDeclares",
    "IfcRelAssociatesClassification",
}

# Non-IfcRelationship entity types that also carry topological/association info
NON_REL_TOPO_TYPES = [
    "IfcPresentationLayerAssignment",
    "IfcClassification",
    "IfcClassificationReference",
    "IfcMaterialLayerSetUsage",
    "IfcMaterialProfileSetUsage",
    "IfcMaterialConstituentSet",
    "IfcGroup",
    "IfcZone",
    "IfcSystem",
    "IfcDistributionSystem",
    "IfcDistributionCircuit",
    "IfcStructuralAnalysisModel",
]


def _safe_by_type(f, cls):
    try:
        return f.by_type(cls)
    except RuntimeError:
        return []


def _get_contained(spatial):
    return [e for rel in getattr(spatial, "ContainsElements", [])
            for e in rel.RelatedElements]


def _expand_nested(elements):
    expanded = list(elements)
    seen = {e.id() for e in expanded}
    queue = list(elements)
    while queue:
        parent = queue.pop()
        for rel in getattr(parent, "IsDecomposedBy", []):
            for child in rel.RelatedObjects:
                if child.id() not in seen and child.is_a("IfcProduct"):
                    expanded.append(child)
                    seen.add(child.id())
                    queue.append(child)
        for rel in getattr(parent, "IsNestedBy", []):
            for child in rel.RelatedObjects:
                if child.id() not in seen and child.is_a("IfcProduct"):
                    expanded.append(child)
                    seen.add(child.id())
                    queue.append(child)
    return expanded


def _get_spaces(spatial):
    spaces = [c for rel in getattr(spatial, "IsDecomposedBy", [])
              for c in rel.RelatedObjects if c.is_a("IfcSpace")]
    sp_ids = {s.id() for s in spaces}
    for rel in getattr(spatial, "ContainsElements", []):
        for e in rel.RelatedElements:
            if e.is_a("IfcSpace") and e.id() not in sp_ids:
                spaces.append(e)
                sp_ids.add(e.id())
    return spaces


# ── Pass 1: schema introspection + relationship enumeration ──────────────────

def _get_all_rel_subtypes(schema_name: str = "IFC4") -> set[str]:
    """Use IFC schema introspection to get ALL subtypes of IfcRelationship."""
    schema = ifcopenshell.schema_by_name(schema_name)
    rel_decl = schema.declaration_by_name("IfcRelationship")

    all_subtypes: set[str] = set()

    def _walk(decl):
        all_subtypes.add(decl.name())
        for sub in decl.subtypes():
            _walk(sub)

    _walk(rel_decl)
    return all_subtypes


def pass1_rel_types(ifc_files: list[Path]):
    """Enumerate all IfcRelationship subtypes across models, cross-reference
    with schema and converter coverage."""

    # Get all possible subtypes from IFC2X3 and IFC4 schemas
    schema_subtypes: set[str] = set()
    for s in ("IFC2X3", "IFC4"):
        try:
            schema_subtypes |= _get_all_rel_subtypes(s)
        except Exception:
            pass

    print(f"  Schema knows {len(schema_subtypes)} IfcRelationship (sub)types total\n")

    # Enumerate what's actually present in our models
    global_rel_types: dict[str, dict] = defaultdict(lambda: {"count": 0, "models": []})

    for ifc_path in ifc_files:
        rel = str(ifc_path.relative_to(PROJECTS))
        try:
            f = ifcopenshell.open(str(ifc_path))
        except Exception as e:
            print(f"  ERROR loading {rel}: {e}")
            continue
        model_rels: dict[str, int] = defaultdict(int)
        for entity in f.by_type("IfcRelationship"):
            model_rels[entity.is_a()] += 1
        for cls, count in model_rels.items():
            global_rel_types[cls]["count"] += count
            global_rel_types[cls]["models"].append(rel)
        print(f"  {rel}: {sum(model_rels.values())} rels, {len(model_rels)} types")

    print(f"\n{'='*80}")
    print("  RELATIONSHIP TYPE COVERAGE")
    print(f"{'='*80}")
    for cls in sorted(global_rel_types.keys()):
        info = global_rel_types[cls]
        n_models = len(info["models"])
        handled = cls in HANDLED_RELS
        marker = "      OK" if handled else ">> NOT OK"
        print(f"  {marker}  {cls:50s} "
              f"count={info['count']:6d}  in {n_models:2d} models")

    found = set(global_rel_types.keys())
    not_handled = found - HANDLED_RELS
    print(f"\n  Found in models: {len(found)}, "
          f"Handled by converter: {len(found - not_handled)}, "
          f"NOT handled: {len(not_handled)}")
    if not_handled:
        print(f"  Unhandled types: {', '.join(sorted(not_handled))}")
    return global_rel_types


# ── Pass 2: non-IfcRelationship topological entities ─────────────────────────

def pass2_non_rel(ifc_files: list[Path]):
    """Check non-IfcRelationship entity types that carry topological info."""
    print(f"\n{'='*80}")
    print("  NON-IfcRelationship TOPOLOGICAL ENTITIES")
    print(f"{'='*80}")

    global_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "models": []})

    for ifc_path in ifc_files:
        rel = str(ifc_path.relative_to(PROJECTS))
        try:
            f = ifcopenshell.open(str(ifc_path))
        except Exception as e:
            continue
        for cls in NON_REL_TOPO_TYPES:
            instances = _safe_by_type(f, cls)
            if instances:
                global_counts[cls]["count"] += len(instances)
                global_counts[cls]["models"].append(rel)

    for cls in sorted(global_counts.keys()):
        info = global_counts[cls]
        n_models = len(info["models"])
        print(f"  {cls:50s} count={info['count']:6d}  in {n_models:2d} models")

    if not global_counts:
        print("  (none found)")


# ── Pass 3: per-model correctness checks ─────────────────────────────────────

def pass3_audit(ifc_files: list[Path]):
    """Check containment, relationship maps, hierarchy for each model."""
    total_issues = 0

    for ifc_path in ifc_files:
        rel = str(ifc_path.relative_to(PROJECTS))
        print(f"\n{'='*70}")
        print(f"  {rel}")
        print(f"{'='*70}")
        t0 = time.time()
        try:
            issues, summary = _audit_one(str(ifc_path))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            total_issues += 1
            continue
        elapsed = time.time() - t0

        print(f"  Products: {summary['products']}, "
              f"Sites: {summary['sites']}, Buildings: {summary['buildings']}, "
              f"Storeys: {summary['storeys']}, Spaces: {summary['spaces']}")
        print(f"  Hosts: {summary['host_pairs']}, "
              f"Boundaries: {summary['boundaries']}, "
              f"WallConn: {summary['wall_connections']}, "
              f"Systems: {summary['systems']} ({summary['system_members']} members), "
              f"Ports: {summary['port_connections']}")
        print(f"  Orphans: {summary['orphans_total']} "
              f"(containment-bug={summary['orphans_with_containment_bug']}, "
              f"ref-only={summary['orphans_referenced_only']}, "
              f"decomp={summary['orphans_decomposed']}, "
              f"true={summary['orphans_true']})")

        if issues:
            total_issues += len(issues)
            for iss in issues:
                print(f"  !! {iss}")
        else:
            print(f"  OK  ({elapsed:.1f}s)")

    print(f"\n{'='*70}")
    if total_issues == 0:
        print("ALL MODELS PASSED")
    else:
        print(f"TOTAL ISSUES: {total_issues}")
    print(f"{'='*70}")
    return total_issues


def _audit_one(ifc_path: str):
    f = ifcopenshell.open(ifc_path)
    issues: list[str] = []
    all_products = [p for p in f.by_type("IfcProduct") if p.is_a() not in SKIP_CLASSES]

    # ── 1. Spatial containment ────────────────────────────────────────────
    written_ids: set[int] = set()

    def _collect_space_elements(space):
        for e in _expand_nested(_get_contained(space)):
            if not e.is_a("IfcSpace"):
                written_ids.add(e.id())

    def walk_spatial(entity):
        for rel in getattr(entity, "IsDecomposedBy", []):
            for child in rel.RelatedObjects:
                if child.is_a("IfcBuildingStorey"):
                    raw = _get_contained(child)
                    elements = _expand_nested(raw)
                    spaces = _get_spaces(child)
                    seen = {e.id() for e in elements}
                    for sp in spaces:
                        for e in _expand_nested(_get_contained(sp)):
                            if e.id() not in seen:
                                elements.append(e)
                                seen.add(e.id())
                    for e in elements:
                        written_ids.add(e.id())
                    for s in spaces:
                        written_ids.add(s.id())
                    walk_spatial(child)
                elif child.is_a("IfcBuilding"):
                    for e in _expand_nested(_get_contained(child)):
                        written_ids.add(e.id())
                    walk_spatial(child)
                elif child.is_a("IfcSpace"):
                    written_ids.add(child.id())
                    _collect_space_elements(child)
                    walk_spatial(child)
                elif child.is_a("IfcSite"):
                    for e in _expand_nested(_get_contained(child)):
                        written_ids.add(e.id())
                    walk_spatial(child)
                else:
                    walk_spatial(child)

    for proj in f.by_type("IfcProject"):
        walk_spatial(proj)

    orphaned = [p for p in all_products if p.id() not in written_ids]

    orphans_with_rel = []
    orphans_with_ref = []
    orphans_with_decomp = []
    true_orphans = []
    for p in orphaned:
        c = getattr(p, "ContainedInStructure", [])
        r = getattr(p, "ReferencedInStructures", [])
        d = getattr(p, "Decomposes", [])
        if c:
            container = c[0].RelatingStructure
            orphans_with_rel.append(
                f"#{p.id()} {p.is_a()} in {container.is_a()} \"{container.Name}\"")
        elif r:
            ref = r[0].RelatingStructure
            orphans_with_ref.append(
                f"#{p.id()} {p.is_a()} ref'd in {ref.is_a()} \"{ref.Name}\"")
        elif d:
            parent = d[0].RelatingObject
            orphans_with_decomp.append(
                f"#{p.id()} {p.is_a()} decomp from {parent.is_a()} \"{parent.Name}\"")
        else:
            true_orphans.append(f"#{p.id()} {p.is_a()}")

    if orphans_with_rel:
        issues.append(f"CONTAINMENT BUG: {len(orphans_with_rel)} elements have "
                      f"ContainedInStructure but orphaned")
        for o in orphans_with_rel[:5]:
            issues.append(f"    {o}")
        if len(orphans_with_rel) > 5:
            issues.append(f"    ... and {len(orphans_with_rel) - 5} more")
    if orphans_with_ref:
        issues.append(f"REFERENCED NOT CAPTURED: {len(orphans_with_ref)} elements")
        for o in orphans_with_ref[:3]:
            issues.append(f"    {o}")
    if orphans_with_decomp:
        issues.append(f"DECOMP NOT CAPTURED: {len(orphans_with_decomp)} elements")
        for o in orphans_with_decomp[:3]:
            issues.append(f"    {o}")

    # ── 2. Host map ───────────────────────────────────────────────────────
    ifc_host_pairs = set()
    for r in f.by_type("IfcRelVoidsElement"):
        host = r.RelatingBuildingElement
        opening = r.RelatedOpeningElement
        for fr in getattr(opening, "HasFillings", []):
            ifc_host_pairs.add((fr.RelatedBuildingElement.id(), host.id()))
    converter_host = _build_host_map(f)
    converter_host_pairs = {(fid, v["id"]) for fid, v in converter_host.items()}
    missing_hosts = ifc_host_pairs - converter_host_pairs
    if missing_hosts:
        issues.append(f"HOST MAP MISSING: {len(missing_hosts)} pairs")

    # ── 3. Space boundaries ───────────────────────────────────────────────
    ifc_bc = sum(1 for r in _safe_by_type(f, "IfcRelSpaceBoundary")
                 if r.RelatedBuildingElement and r.RelatingSpace)
    conv_bc = sum(len(v) for v in _build_space_boundary_map(f).values())
    if conv_bc != ifc_bc:
        issues.append(f"BOUNDARY MISMATCH: ifc={ifc_bc}, converter={conv_bc}")

    # ── 4. Wall connections ───────────────────────────────────────────────
    ifc_wc = sum(1 for r in _safe_by_type(f, "IfcRelConnectsPathElements")
                 if r.RelatingElement and r.RelatedElement)
    conv_wc = sum(len(v) for v in _build_wall_connection_map(f).values())
    if conv_wc != ifc_wc * 2:
        issues.append(f"WALL CONN MISMATCH: ifc={ifc_wc} (expect {ifc_wc*2} entries), "
                      f"converter={conv_wc}")

    # ── 5. System assignments ─────────────────────────────────────────────
    ifc_sys_members: set[int] = set()
    n_systems = 0
    for r in _safe_by_type(f, "IfcRelAssignsToGroup"):
        g = r.RelatingGroup
        if not g:
            continue
        if g.is_a("IfcSystem") or g.is_a("IfcDistributionSystem"):
            n_systems += 1
            for o in r.RelatedObjects:
                if o.is_a("IfcProduct"):
                    ifc_sys_members.add(o.id())
    conv_sys = set(_build_system_map(f).keys())
    missing_sys = ifc_sys_members - conv_sys
    if missing_sys:
        issues.append(f"SYSTEM MAP MISSING: {len(missing_sys)} elements")

    # ── 6. Port connections ───────────────────────────────────────────────
    ifc_pc = sum(1 for r in _safe_by_type(f, "IfcRelConnectsPorts")
                 if r.RelatingPort and r.RelatedPort)
    conv_ports = _build_port_map(f)
    conv_connected = sum(1 for entries in conv_ports.values()
                         for e in entries if e.get("connected_to") is not None)
    if conv_connected != ifc_pc * 2:
        issues.append(f"PORT CONN MISMATCH: ifc={ifc_pc} (expect {ifc_pc*2}), "
                      f"converter={conv_connected}")

    # ── 7. Spatial hierarchy ──────────────────────────────────────────────
    tree_ids: set[int] = set()

    def count_tree(entity):
        for r in getattr(entity, "IsDecomposedBy", []):
            for child in r.RelatedObjects:
                tree_ids.add(child.id())
                count_tree(child)

    for proj in f.by_type("IfcProject"):
        count_tree(proj)

    ms = [s for s in f.by_type("IfcSite") if s.id() not in tree_ids]
    mb = [b for b in f.by_type("IfcBuilding") if b.id() not in tree_ids]
    mst = [s for s in f.by_type("IfcBuildingStorey") if s.id() not in tree_ids]
    msp = [s for s in f.by_type("IfcSpace")
           if s.id() not in tree_ids and s.id() not in written_ids]
    if ms:
        issues.append(f"HIERARCHY: {len(ms)} IfcSite(s) not in tree")
    if mb:
        issues.append(f"HIERARCHY: {len(mb)} IfcBuilding(s) not in tree")
    if mst:
        issues.append(f"HIERARCHY: {len(mst)} IfcBuildingStorey(s) not in tree")
    if msp:
        issues.append(f"HIERARCHY: {len(msp)} IfcSpace(s) unreachable")

    summary = {
        "products": len(all_products),
        "orphans_total": len(orphaned),
        "orphans_true": len(true_orphans),
        "orphans_with_containment_bug": len(orphans_with_rel),
        "orphans_referenced_only": len(orphans_with_ref),
        "orphans_decomposed": len(orphans_with_decomp),
        "sites": len(f.by_type("IfcSite")),
        "buildings": len(f.by_type("IfcBuilding")),
        "storeys": len(f.by_type("IfcBuildingStorey")),
        "spaces": len(f.by_type("IfcSpace")),
        "host_pairs": len(ifc_host_pairs),
        "boundaries": ifc_bc,
        "wall_connections": ifc_wc,
        "systems": n_systems,
        "system_members": len(ifc_sys_members),
        "port_connections": ifc_pc,
    }
    return issues, summary


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rels-only", action="store_true",
                    help="Only run Pass 1 + Pass 2 (skip per-model checks)")
    ap.add_argument("--pass3-only", action="store_true",
                    help="Only run Pass 3 (per-model correctness checks)")
    args = ap.parse_args()

    ifc_files = sorted(PROJECTS.glob("*/*.ifc"))
    print(f"Found {len(ifc_files)} IFC files\n")

    if not args.pass3_only:
        print("=" * 80)
        print("PASS 1: IfcRelationship subtypes (schema introspection + model scan)")
        print("=" * 80)
        pass1_rel_types(ifc_files)

        print()
        pass2_non_rel(ifc_files)

        if args.rels_only:
            return 0

    print(f"\n\n{'='*80}")
    print("PASS 3: Per-model correctness checks")
    print("=" * 80)
    n = pass3_audit(ifc_files)
    return 0 if n == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
