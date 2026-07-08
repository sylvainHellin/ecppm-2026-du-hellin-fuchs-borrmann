#!/usr/bin/env python3
"""
ifc2fs — Convert IFC files to a filesystem-based representation.

Maps the IFC spatial hierarchy to directories and semantic data to JSON files,
enabling AI agents to query BIM models using standard CLI tools (ls, cat, grep, find).

Filesystem layout:
    <output>/
    ├── __meta__/
    │   ├── header.json          # Schema, software, author, timestamp
    │   ├── units.json           # Unit system (metre, degree, EUR, …)
    │   └── project.json         # IfcProject with owner history
    ├── __types__/
    │   ├── IfcWallType/         # One JSON per type definition
    │   └── …
    ├── __materials__/           # Material and layer-set definitions
    └── Site__<name>/
        └── Building__<name>/
            ├── __building__.json
            └── <StoreyName>/
                ├── __storey__.json       # Elevation, element summary
                ├── __geometry__.json     # All geometry for this storey, keyed by element id
                ├── spaces/
                │   └── 001__Room.json    # geometry_key → __geometry__.json
                ├── IfcWall/
                │   └── 0001__Wall.json   # Props, type, material, openings, styles, geometry_key
                ├── IfcDoor/
                └── …

Each element JSON contains a "geometry_key" field whose value is a key
into the storey-level __geometry__.json file (matching the element's "id").
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import ifcopenshell
import ifcopenshell.util.classification as classification_util
import ifcopenshell.util.element as element_util
import ifcopenshell.util.placement as placement_util


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

_IFC_UNICODE_RE = re.compile(r"\\X2\\([0-9A-Fa-f]+)\\X0\\")
_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def _decode_ifc(s):
    """Decode IFC \\X2\\xxxx\\X0\\ Unicode escapes."""
    if not isinstance(s, str):
        return s

    def _repl(m):
        h = m.group(1)
        return "".join(chr(int(h[i : i + 4], 16)) for i in range(0, len(h), 4))

    return _IFC_UNICODE_RE.sub(_repl, s)


def _safe(name: str | None, max_len: int = 120) -> str:
    """Turn an IFC name into a filesystem-safe string."""
    if not name:
        return "_unnamed_"
    name = _decode_ifc(name)
    name = _UNSAFE_CHARS_RE.sub("_", name)
    name = name.replace(" ", "_")
    name = _MULTI_UNDERSCORE_RE.sub("_", name).strip("_.")
    if len(name) > max_len:
        name = name[:max_len].rstrip("_")
    return name or "_unnamed_"


def _d(val):
    """Decode any IFC string value; pass through non-strings."""
    return _decode_ifc(val) if isinstance(val, str) else val


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------


def _write(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Property / quantity helpers
# ---------------------------------------------------------------------------


def _strip_id(psets: dict) -> dict:
    """Remove the internal 'id' key that element_util.get_psets() injects."""
    return {
        name: {k: v for k, v in props.items() if k != "id"}
        for name, props in psets.items()
    }


def _get_type_info(element):
    """Return a compact type-info dict via element_util, or None."""
    t = element_util.get_type(element)
    if t is None:
        return None
    info = {"name": _d(t.Name), "ifc_class": t.is_a(), "global_id": t.GlobalId}
    pt = getattr(t, "PredefinedType", None)
    if pt:
        info["predefined_type"] = str(pt)
    return info


def _get_classification_info(element):
    """Return classification references via classification_util, or None."""
    refs = classification_util.get_references(element)
    if not refs:
        return None
    out = []
    for ref in refs:
        ident = (
            getattr(ref, "Identification", None)
            or getattr(ref, "ItemReference", None)
        )
        desc = getattr(ref, "Description", None)
        out.append({
            "name": _d(ref.Name) if ref.Name else None,
            "identification": _d(ident) if ident else None,
            "description": _d(desc) if desc else None,
        })
    return out or None


def _extract_material(mat):
    """Recursive material extraction supporting all material-select types."""
    if mat is None:
        return None
    if mat.is_a("IfcMaterial"):
        return {"type": "IfcMaterial", "name": _d(mat.Name), "category": getattr(mat, "Category", None)}
    if mat.is_a("IfcMaterialLayerSetUsage"):
        ls = mat.ForLayerSet
        return {
            "type": "IfcMaterialLayerSetUsage",
            "name": _d(ls.LayerSetName) if ls.LayerSetName else None,
            "direction": str(mat.LayerSetDirection) if mat.LayerSetDirection else None,
            "offset": mat.OffsetFromReferenceLine,
            "layers": [
                {
                    "material": _d(la.Material.Name) if la.Material else None,
                    "thickness": la.LayerThickness,
                }
                for la in ls.MaterialLayers
            ],
        }
    if mat.is_a("IfcMaterialLayerSet"):
        return {
            "type": "IfcMaterialLayerSet",
            "name": _d(mat.LayerSetName) if mat.LayerSetName else None,
            "layers": [
                {"material": _d(la.Material.Name) if la.Material else None, "thickness": la.LayerThickness}
                for la in mat.MaterialLayers
            ],
        }
    if mat.is_a("IfcMaterialList"):
        return {"type": "IfcMaterialList", "materials": [_d(m.Name) for m in mat.Materials]}
    if mat.is_a("IfcMaterialConstituentSet"):
        return {
            "type": "IfcMaterialConstituentSet",
            "name": _d(mat.Name) if mat.Name else None,
            "constituents": [
                {
                    "name": _d(c.Name) if c.Name else None,
                    "material": _d(c.Material.Name) if c.Material else None,
                    "fraction": c.Fraction,
                }
                for c in (mat.MaterialConstituents or [])
            ],
        }
    if mat.is_a("IfcMaterialProfileSetUsage"):
        ps = mat.ForProfileSet
        return {
            "type": "IfcMaterialProfileSetUsage",
            "name": _d(ps.Name) if ps.Name else None,
            "profiles": [
                {
                    "name": _d(p.Name) if p.Name else None,
                    "material": _d(p.Material.Name) if p.Material else None,
                    "profile": p.Profile.is_a() if p.Profile else None,
                }
                for p in (ps.MaterialProfiles or [])
            ],
        }
    if mat.is_a("IfcMaterialProfileSet"):
        return {
            "type": "IfcMaterialProfileSet",
            "name": _d(mat.Name) if mat.Name else None,
            "profiles": [
                {
                    "name": _d(p.Name) if p.Name else None,
                    "material": _d(p.Material.Name) if p.Material else None,
                    "profile": p.Profile.is_a() if p.Profile else None,
                }
                for p in (mat.MaterialProfiles or [])
            ],
        }
    return {"type": mat.is_a(), "raw": str(mat)}


def _material_info(element):
    mat = element_util.get_material(element, should_inherit=False)
    return _extract_material(mat) if mat else None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _placement(element):
    try:
        if not element.ObjectPlacement:
            return None
        m = placement_util.get_local_placement(element.ObjectPlacement)
        if m is not None:
            return {
                "x": round(float(m[0][3]), 6),
                "y": round(float(m[1][3]), 6),
                "z": round(float(m[2][3]), 6),
            }
    except Exception:
        pass
    return None


def _bbox(element):
    rep = getattr(element, "Representation", None)
    if rep is None:
        return None
    for sr in rep.Representations:
        if sr.RepresentationType == "BoundingBox":
            for item in sr.Items:
                if item.is_a("IfcBoundingBox"):
                    return {
                        "x_dim": round(item.XDim, 6),
                        "y_dim": round(item.YDim, 6),
                        "z_dim": round(item.ZDim, 6),
                    }
    return None


def _coords(point):
    """Extract coordinates as a rounded list."""
    return [round(c, 6) for c in point.Coordinates]


def _axis2placement(p):
    if p is None:
        return None
    d = {"location": _coords(p.Location)}
    if hasattr(p, "Axis") and p.Axis:
        d["axis"] = list(p.Axis.DirectionRatios)
    if hasattr(p, "RefDirection") and p.RefDirection:
        d["ref_direction"] = list(p.RefDirection.DirectionRatios)
    return d


def _extract_profile(profile):
    """Extract a readable description of an IFC profile."""
    d = {"type": profile.is_a(), "profile_type": str(profile.ProfileType)}
    if hasattr(profile, "ProfileName") and profile.ProfileName:
        d["name"] = _d(profile.ProfileName)
    if profile.is_a("IfcRectangleProfileDef"):
        d["x_dim"] = profile.XDim
        d["y_dim"] = profile.YDim
    elif profile.is_a("IfcCircleProfileDef"):
        d["radius"] = profile.Radius
    elif profile.is_a("IfcArbitraryClosedProfileDef"):
        curve = profile.OuterCurve
        if curve.is_a("IfcPolyline"):
            d["outer_curve"] = {"type": "IfcPolyline", "points": [_coords(p) for p in curve.Points]}
        elif curve.is_a("IfcCompositeCurve"):
            d["outer_curve"] = {"type": "IfcCompositeCurve", "segment_count": len(curve.Segments)}
        else:
            d["outer_curve"] = {"type": curve.is_a()}
    elif profile.is_a("IfcIShapeProfileDef"):
        d.update({"overall_width": profile.OverallWidth, "overall_depth": profile.OverallDepth,
                   "web_thickness": profile.WebThickness, "flange_thickness": profile.FlangeThickness})
    if hasattr(profile, "Position") and profile.Position:
        d["position"] = _axis2placement(profile.Position)
    return d


def _extract_rep_item(item, depth=0):
    """Recursively extract a representation item into a JSON-friendly dict."""
    if depth > 10:
        return {"type": item.is_a(), "_truncated": True}

    if item.is_a("IfcMappedItem"):
        src = item.MappingSource
        return {
            "type": "IfcMappedItem",
            "source_type": src.MappedRepresentation.RepresentationType if src and src.MappedRepresentation else None,
            "mapping_origin": _axis2placement(src.MappingOrigin) if src and src.MappingOrigin else None,
        }

    if item.is_a("IfcExtrudedAreaSolid"):
        return {
            "type": "IfcExtrudedAreaSolid",
            "profile": _extract_profile(item.SweptArea),
            "position": _axis2placement(item.Position) if item.Position else None,
            "direction": list(item.ExtrudedDirection.DirectionRatios),
            "depth": item.Depth,
        }

    if item.is_a("IfcFacetedBrep"):
        faces = []
        for face in item.Outer.CfsFaces:
            for bound in face.Bounds:
                loop = bound.Bound
                if loop.is_a("IfcPolyLoop"):
                    faces.append({
                        "vertices": [_coords(p) for p in loop.Polygon],
                        "orientation": bound.Orientation,
                    })
        return {"type": "IfcFacetedBrep", "face_count": len(item.Outer.CfsFaces), "faces": faces}

    if item.is_a("IfcShellBasedSurfaceModel"):
        shells = []
        for shell in item.SbsmBoundary:
            faces = []
            for face in shell.CfsFaces:
                for bound in face.Bounds:
                    loop = bound.Bound
                    if loop.is_a("IfcPolyLoop"):
                        faces.append({"vertices": [_coords(p) for p in loop.Polygon], "orientation": bound.Orientation})
            shells.append({"face_count": len(shell.CfsFaces), "faces": faces})
        return {"type": "IfcShellBasedSurfaceModel", "shells": shells}

    if item.is_a("IfcBooleanClippingResult"):
        return {
            "type": "IfcBooleanClippingResult",
            "operator": str(item.Operator),
            "first_operand": _extract_rep_item(item.FirstOperand, depth + 1),
            "second_operand": _extract_half_space(item.SecondOperand) if item.SecondOperand.is_a("IfcHalfSpaceSolid") else _extract_rep_item(item.SecondOperand, depth + 1),
        }

    if item.is_a("IfcBoundingBox"):
        return {"type": "IfcBoundingBox", "corner": _coords(item.Corner), "x_dim": item.XDim, "y_dim": item.YDim, "z_dim": item.ZDim}

    if item.is_a("IfcPolyline"):
        return {"type": "IfcPolyline", "points": [_coords(p) for p in item.Points]}

    if item.is_a("IfcGeometricCurveSet"):
        elems = []
        for e in item.Elements:
            if e.is_a("IfcPolyline"):
                elems.append({"type": "IfcPolyline", "points": [_coords(p) for p in e.Points]})
            else:
                elems.append({"type": e.is_a()})
        return {"type": "IfcGeometricCurveSet", "elements": elems}

    if item.is_a("IfcTextLiteralWithExtent"):
        return {"type": "IfcTextLiteralWithExtent", "literal": _d(item.Literal) if item.Literal else None}

    return {"type": item.is_a()}


def _extract_half_space(hs):
    d = {"type": hs.is_a(), "agreement_flag": hs.AgreementFlag}
    if hs.BaseSurface and hs.BaseSurface.is_a("IfcPlane"):
        d["base_surface"] = {"type": "IfcPlane", "position": _axis2placement(hs.BaseSurface.Position)}
    if hs.is_a("IfcPolygonalBoundedHalfSpace"):
        if hs.PolygonalBoundary and hs.PolygonalBoundary.is_a("IfcPolyline"):
            d["boundary"] = {"type": "IfcPolyline", "points": [_coords(p) for p in hs.PolygonalBoundary.Points]}
        if hs.Position:
            d["position"] = _axis2placement(hs.Position)
    return d


def _extract_geometry(element):
    """Extract full geometric representation of an element as a sidecar dict."""
    rep = getattr(element, "Representation", None)
    if rep is None:
        return None
    representations = []
    for sr in rep.Representations:
        items = [_extract_rep_item(item) for item in sr.Items]
        representations.append({
            "identifier": sr.RepresentationIdentifier,
            "type": sr.RepresentationType,
            "items": items,
        })
    return {"representations": representations} if representations else None


# ---------------------------------------------------------------------------
# Style / appearance extraction
# ---------------------------------------------------------------------------


def _extract_styles(element):
    """Extract surface/curve style info for embedding in element JSON."""
    rep = getattr(element, "Representation", None)
    if rep is None:
        return None
    seen = set()
    styles = []
    for sr in rep.Representations:
        for item in sr.Items:
            _collect_styles_from_item(item, styles, seen)
    return styles if styles else None


def _collect_styles_from_item(item, styles, seen, depth=0):
    if depth > 10:
        return
    for si in getattr(item, "StyledByItem", []):
        for sa in si.Styles:
            for st in getattr(sa, "Styles", []):
                if st.is_a("IfcSurfaceStyle"):
                    sd = {"name": _d(st.Name) if st.Name else None}
                    for rendering in st.Styles:
                        if rendering.is_a("IfcSurfaceStyleRendering") or rendering.is_a("IfcSurfaceStyleShading"):
                            c = rendering.SurfaceColour
                            if c:
                                sd["surface_color"] = {"r": round(c.Red, 4), "g": round(c.Green, 4), "b": round(c.Blue, 4)}
                            t = getattr(rendering, "Transparency", None)
                            if t is not None:
                                sd["transparency"] = round(t, 4)
                    key = json.dumps(sd, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        styles.append(sd)
                elif st.is_a("IfcCurveStyle"):
                    sd = {"name": _d(st.Name) if st.Name else None, "type": "curve"}
                    if st.CurveColour and st.CurveColour.is_a("IfcColourRgb"):
                        c = st.CurveColour
                        sd["curve_color"] = {"r": round(c.Red, 4), "g": round(c.Green, 4), "b": round(c.Blue, 4)}
                    key = json.dumps(sd, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        styles.append(sd)
    # Recurse into sub-items for mapped representations
    if item.is_a("IfcMappedItem"):
        src = item.MappingSource
        if src and src.MappedRepresentation:
            for sub_item in src.MappedRepresentation.Items:
                _collect_styles_from_item(sub_item, styles, seen, depth + 1)
    if item.is_a("IfcBooleanClippingResult"):
        _collect_styles_from_item(item.FirstOperand, styles, seen, depth + 1)


# ---------------------------------------------------------------------------
# Presentation layer map
# ---------------------------------------------------------------------------


def _build_layer_map(f):
    """Build element_id → layer_name mapping from IfcPresentationLayerAssignment."""
    rep_to_layer: dict[int, str] = {}
    for pla in f.by_type("IfcPresentationLayerAssignment"):
        name = _d(pla.Name) if pla.Name else None
        for item in pla.AssignedItems:
            rep_to_layer[item.id()] = name

    elem_to_layer: dict[int, str] = {}
    for product in f.by_type("IfcProduct"):
        prep = getattr(product, "Representation", None)
        if prep is None:
            continue
        for sr in prep.Representations:
            if sr.id() in rep_to_layer:
                elem_to_layer[product.id()] = rep_to_layer[sr.id()]
                break
            for item in sr.Items:
                if item.id() in rep_to_layer:
                    elem_to_layer[product.id()] = rep_to_layer[item.id()]
                    break
    return elem_to_layer


# ---------------------------------------------------------------------------
# Opening / host relationship helpers
# ---------------------------------------------------------------------------


def _opening_info(element):
    """Extract openings (voids) in this element and their fillings (doors/windows)."""
    openings = []
    for rel in getattr(element, "HasOpenings", []):
        opening = rel.RelatedOpeningElement
        od = {
            "id": opening.id(),
            "name": _d(opening.Name) if opening.Name else None,
        }
        bb = _bbox(opening)
        if bb:
            od["bounding_box"] = bb
        pl = _placement(opening)
        if pl:
            od["placement"] = pl
        for fill_rel in getattr(opening, "HasFillings", []):
            filling = fill_rel.RelatedBuildingElement
            od["filling"] = {
                "id": filling.id(),
                "ifc_class": filling.is_a(),
                "name": _d(filling.Name) if filling.Name else None,
                "global_id": filling.GlobalId,
            }
        openings.append(od)
    return openings if openings else None


def _build_host_map(f):
    """Build filling_element_id → host_element info for doors/windows."""
    host_map: dict[int, dict] = {}
    for rel in f.by_type("IfcRelVoidsElement"):
        host = rel.RelatingBuildingElement
        opening = rel.RelatedOpeningElement
        for fill_rel in getattr(opening, "HasFillings", []):
            filling = fill_rel.RelatedBuildingElement
            host_map[filling.id()] = {
                "id": host.id(),
                "ifc_class": host.is_a(),
                "name": _d(host.Name) if host.Name else None,
                "global_id": host.GlobalId,
            }
    return host_map


# ---------------------------------------------------------------------------
# Relationship maps (space boundaries, wall connections, systems, ports)
# ---------------------------------------------------------------------------


def _build_space_boundary_map(f):
    """Build element_id → [space info] from IfcRelSpaceBoundary (reverse of space.boundaries)."""
    boundary_map: dict[int, list] = defaultdict(list)
    for rel in _safe_by_type(f, "IfcRelSpaceBoundary"):
        space = rel.RelatingSpace
        be = rel.RelatedBuildingElement
        if not be or not space:
            continue
        boundary_map[be.id()].append({
            "id": space.id(),
            "global_id": space.GlobalId,
            "ifc_class": space.is_a(),
            "name": _d(space.Name) if space.Name else None,
            "long_name": _d(space.LongName) if getattr(space, "LongName", None) else None,
            "physical_or_virtual": str(rel.PhysicalOrVirtualBoundary) if rel.PhysicalOrVirtualBoundary else None,
            "internal_or_external": str(rel.InternalOrExternalBoundary) if rel.InternalOrExternalBoundary else None,
        })
    return dict(boundary_map)


def _build_wall_connection_map(f):
    """Build wall_id → [connected wall info] from IfcRelConnectsPathElements."""
    conn_map: dict[int, list] = defaultdict(list)
    for rel in _safe_by_type(f, "IfcRelConnectsPathElements"):
        a = rel.RelatingElement
        b = rel.RelatedElement
        if not a or not b:
            continue
        conn_type_a = str(rel.RelatingConnectionType) if rel.RelatingConnectionType else None
        conn_type_b = str(rel.RelatedConnectionType) if rel.RelatedConnectionType else None
        conn_map[a.id()].append({
            "id": b.id(),
            "global_id": b.GlobalId,
            "ifc_class": b.is_a(),
            "name": _d(b.Name) if b.Name else None,
            "connection_type": conn_type_a,
        })
        conn_map[b.id()].append({
            "id": a.id(),
            "global_id": a.GlobalId,
            "ifc_class": a.is_a(),
            "name": _d(a.Name) if a.Name else None,
            "connection_type": conn_type_b,
        })
    return dict(conn_map)


def _build_system_map(f):
    """Build element_id → system info from IfcRelAssignsToGroup where group is a system."""
    system_map: dict[int, dict] = {}
    for rel in _safe_by_type(f, "IfcRelAssignsToGroup"):
        group = rel.RelatingGroup
        if not group:
            continue
        is_system = group.is_a("IfcSystem") or group.is_a("IfcDistributionSystem")
        if not is_system:
            continue
        info = {
            "id": group.id(),
            "global_id": group.GlobalId,
            "ifc_class": group.is_a(),
            "name": _d(group.Name) if group.Name else None,
        }
        ot = getattr(group, "ObjectType", None)
        if ot:
            info["object_type"] = _d(ot)
        pt = getattr(group, "PredefinedType", None)
        if pt:
            info["predefined_type"] = str(pt)
        for obj in rel.RelatedObjects:
            if obj.is_a("IfcProduct"):
                system_map[obj.id()] = info
    return system_map


def _build_port_map(f):
    """Build element_id → [port + connection info] from IfcRelNests / IfcRelConnectsPortToElement + IfcRelConnectsPorts."""
    port_to_owner: dict[int, int] = {}
    owner_ports: dict[int, list] = defaultdict(list)

    def _register_port(owner, port):
        if port.id() in port_to_owner:
            return
        port_to_owner[port.id()] = owner.id()
        fd = getattr(port, "FlowDirection", None)
        owner_ports[owner.id()].append({
            "id": port.id(),
            "flow_direction": str(fd) if fd else None,
            "connected_to": None,
        })

    for rel in _safe_by_type(f, "IfcRelNests"):
        owner = rel.RelatingObject
        if not owner or not owner.is_a("IfcProduct"):
            continue
        for port in rel.RelatedObjects:
            if not port.is_a("IfcDistributionPort"):
                continue
            _register_port(owner, port)

    for rel in _safe_by_type(f, "IfcRelConnectsPortToElement"):
        port = rel.RelatingPort
        elem = rel.RelatedElement
        if not port or not elem:
            continue
        if not port.is_a("IfcDistributionPort") or not elem.is_a("IfcProduct"):
            continue
        _register_port(elem, port)

    for rel in _safe_by_type(f, "IfcRelConnectsPorts"):
        p1 = rel.RelatingPort
        p2 = rel.RelatedPort
        if not p1 or not p2:
            continue
        owner1 = port_to_owner.get(p1.id())
        owner2 = port_to_owner.get(p2.id())

        def _port_target(target_owner_id, target_port_id, model):
            if target_owner_id is None:
                return None
            elem = model.by_id(target_owner_id)
            return {
                "id": elem.id(),
                "global_id": elem.GlobalId,
                "ifc_class": elem.is_a(),
                "name": _d(elem.Name) if elem.Name else None,
            }

        t2 = _port_target(owner2, p2.id(), f)
        t1 = _port_target(owner1, p1.id(), f)

        if owner1 is not None:
            for entry in owner_ports.get(owner1, []):
                if entry["id"] == p1.id():
                    entry["connected_to"] = t2
                    break
        if owner2 is not None:
            for entry in owner_ports.get(owner2, []):
                if entry["id"] == p2.id():
                    entry["connected_to"] = t1
                    break

    return dict(owner_ports)


def _build_coverings_map(f):
    """Build element_id → [covering info] and covering_id → host info from IfcRelCoversBldgElements."""
    coverings_map: dict[int, list] = defaultdict(list)
    covering_to_host: dict[int, dict] = {}
    for rel in _safe_by_type(f, "IfcRelCoversBldgElements"):
        host = rel.RelatingBuildingElement
        if not host:
            continue
        host_info = {
            "id": host.id(),
            "ifc_class": host.is_a(),
            "name": _d(host.Name) if host.Name else None,
            "global_id": host.GlobalId,
        }
        for cov in (rel.RelatedCoverings or []):
            coverings_map[host.id()].append({
                "id": cov.id(),
                "global_id": cov.GlobalId,
                "ifc_class": cov.is_a(),
                "name": _d(cov.Name) if cov.Name else None,
            })
            covering_to_host[cov.id()] = host_info
    return dict(coverings_map), covering_to_host


def _build_services_map(f):
    """Build spatial_element_id → [system info] from IfcRelServicesBuildings."""
    services_map: dict[int, list] = defaultdict(list)
    for rel in _safe_by_type(f, "IfcRelServicesBuildings"):
        system = rel.RelatingSystem
        spatial = rel.RelatedBuildings
        if not system:
            continue
        info = {
            "id": system.id(),
            "global_id": system.GlobalId,
            "ifc_class": system.is_a(),
            "name": _d(system.Name) if system.Name else None,
        }
        pt = getattr(system, "PredefinedType", None)
        if pt:
            info["predefined_type"] = str(pt)
        for bldg in (spatial or []):
            services_map[bldg.id()].append(info)
    return dict(services_map)


# ---------------------------------------------------------------------------
# Element serialisation
# ---------------------------------------------------------------------------


def _serialize_element(elem, *, layer_map=None, host_map=None, geometry_key=None,
                       space_boundary_map=None, wall_connection_map=None,
                       system_map=None, port_map=None, coverings_map=None,
                       covering_to_host_map=None):
    d = {
        "id": elem.id(),
        "global_id": elem.GlobalId,
        "ifc_class": elem.is_a(),
        "name": _d(elem.Name),
        "description": _d(elem.Description),
    }
    tag = getattr(elem, "Tag", None)
    if tag:
        d["tag"] = tag
    pt = getattr(elem, "PredefinedType", None)
    if pt:
        d["predefined_type"] = str(pt)

    ti = _get_type_info(elem)
    if ti:
        d["type"] = ti
    mi = _material_info(elem)
    if mi:
        d["material"] = mi
    ci = _get_classification_info(elem)
    if ci:
        d["classification"] = ci

    if layer_map and elem.id() in layer_map:
        d["layer"] = layer_map[elem.id()]

    if host_map and elem.id() in host_map:
        d["host_element"] = host_map[elem.id()]

    oi = _opening_info(elem)
    if oi:
        d["openings"] = oi

    if space_boundary_map and elem.id() in space_boundary_map:
        d["bounded_spaces"] = space_boundary_map[elem.id()]

    if wall_connection_map and elem.id() in wall_connection_map:
        d["connected_elements"] = wall_connection_map[elem.id()]

    if system_map and elem.id() in system_map:
        d["system"] = system_map[elem.id()]

    if port_map and elem.id() in port_map:
        d["ports"] = port_map[elem.id()]

    if coverings_map and elem.id() in coverings_map:
        d["coverings"] = coverings_map[elem.id()]

    if covering_to_host_map and elem.id() in covering_to_host_map:
        d["covered_element"] = covering_to_host_map[elem.id()]

    pl = _placement(elem)
    if pl:
        d["placement"] = pl
    bb = _bbox(elem)
    if bb:
        d["bounding_box"] = bb

    st = _extract_styles(elem)
    if st:
        d["styles"] = st

    if geometry_key:
        d["geometry_key"] = geometry_key

    psets = _strip_id(element_util.get_psets(elem, psets_only=True, should_inherit=False))
    if psets:
        d["property_sets"] = psets
    qtos = _strip_id(element_util.get_psets(elem, qtos_only=True, should_inherit=False))
    if qtos:
        d["quantities"] = qtos
    return d


def _serialize_space(space, *, layer_map=None, geometry_key=None,
                     space_boundary_map=None, wall_connection_map=None,
                     system_map=None, port_map=None):
    d = _serialize_element(space, layer_map=layer_map, geometry_key=geometry_key,
                           space_boundary_map=space_boundary_map,
                           wall_connection_map=wall_connection_map,
                           system_map=system_map, port_map=port_map)
    ln = getattr(space, "LongName", None)
    if ln:
        d["long_name"] = _d(ln)
    ct = getattr(space, "CompositionType", None)
    if ct:
        d["composition_type"] = str(ct)

    area = None
    for qset in (d.get("quantities") or {}).values():
        if isinstance(qset, dict):
            for key in ("GrossFloorArea", "NetFloorArea", "Area", "NetArea", "GrossArea"):
                if key in qset and isinstance(qset[key], (int, float)):
                    area = qset[key]
                    break
            if area is not None:
                break
    if area is not None:
        d["area"] = area

    boundaries = []
    for rel in getattr(space, "BoundedBy", []):
        be = rel.RelatedBuildingElement
        boundaries.append({
            "element_class": be.is_a() if be else None,
            "element_name": _d(be.Name) if be and be.Name else None,
            "element_id": be.id() if be else None,
            "physical_or_virtual": str(rel.PhysicalOrVirtualBoundary) if rel.PhysicalOrVirtualBoundary else None,
            "internal_or_external": str(rel.InternalOrExternalBoundary) if rel.InternalOrExternalBoundary else None,
        })
    if boundaries:
        d["boundaries"] = boundaries
    return d


# ---------------------------------------------------------------------------
# Type serialisation
# ---------------------------------------------------------------------------


def _serialize_type(tobj):
    d = {
        "id": tobj.id(),
        "global_id": tobj.GlobalId,
        "ifc_class": tobj.is_a(),
        "name": _d(tobj.Name),
        "description": _d(tobj.Description),
    }
    pt = getattr(tobj, "PredefinedType", None)
    if pt:
        d["predefined_type"] = str(pt)

    # Collect instance IDs via both IFC4 and IFC2x3 inverse attributes
    instance_ids = []
    for rel in getattr(tobj, "Types", []):
        instance_ids.extend(o.id() for o in getattr(rel, "RelatedObjects", []))
    for rel in getattr(tobj, "ObjectTypeOf", []):
        instance_ids.extend(o.id() for o in getattr(rel, "RelatedObjects", []))
    d["instance_count"] = len(instance_ids)
    if instance_ids:
        d["element_ids"] = instance_ids

    mi = _material_info(tobj)
    if mi:
        d["material"] = mi

    # Door / window specifics
    op = getattr(tobj, "OperationType", None)
    if op:
        d["operation_type"] = str(op)
    ptp = getattr(tobj, "ParameterTakesPrecedence", None)
    if ptp is not None:
        d["parameter_takes_precedence"] = ptp

    psets = _strip_id(element_util.get_psets(tobj, psets_only=True, should_inherit=False))
    if psets:
        d["property_sets"] = psets

    return d


# ---------------------------------------------------------------------------
# Meta extraction
# ---------------------------------------------------------------------------


def _header(f):
    h = {"schema": f.schema}
    try:
        hdr = f.wrapped_data.header()
        fn = hdr.file_name_py()
        names = fn.get_attribute_names()
        for i, name in enumerate(names):
            val = fn.get_argument(i)
            if isinstance(val, tuple):
                val = list(val)
            if isinstance(val, str):
                val = _d(val)
            h[name] = val
    except Exception:
        pass
    try:
        hdr = f.wrapped_data.header()
        fd = hdr.file_description_py()
        names = fd.get_attribute_names()
        for i, name in enumerate(names):
            val = fd.get_argument(i)
            if isinstance(val, tuple):
                val = [_d(v) if isinstance(v, str) else v for v in val]
            h[name] = val
    except Exception:
        pass
    return h


def _units(f):
    u = {}
    for ua in f.by_type("IfcUnitAssignment"):
        for unit in ua.Units:
            if unit.is_a("IfcSIUnit"):
                key = str(unit.UnitType) if unit.UnitType else "unknown"
                prefix = str(unit.Prefix) + " " if unit.Prefix else ""
                u[key] = f"{prefix}{unit.Name}".strip()
            elif unit.is_a("IfcConversionBasedUnit"):
                key = str(unit.UnitType) if unit.UnitType else "unknown"
                u[key] = _d(unit.Name) if unit.Name else str(unit)
            elif unit.is_a("IfcMonetaryUnit"):
                u["MONETARYUNIT"] = unit.Currency if hasattr(unit, "Currency") else str(unit)
            elif unit.is_a("IfcDerivedUnit"):
                key = str(unit.UnitType) if unit.UnitType else "unknown"
                elements = []
                for de in unit.Elements:
                    uname = ""
                    if de.Unit.is_a("IfcSIUnit"):
                        prefix = str(de.Unit.Prefix) + " " if de.Unit.Prefix else ""
                        uname = f"{prefix}{de.Unit.Name}".strip()
                    elements.append({"unit": uname, "exponent": de.Exponent})
                u[key] = {"derived_elements": elements}
    return u


def _project(f):
    projects = f.by_type("IfcProject")
    if not projects:
        return {}
    p = projects[0]
    d = {
        "global_id": p.GlobalId,
        "name": _d(p.Name),
        "description": _d(p.Description),
        "phase": getattr(p, "Phase", None),
    }
    oh = p.OwnerHistory
    if oh:
        owner = {}
        if oh.OwningUser:
            person = oh.OwningUser.ThePerson
            if person:
                owner["person"] = {
                    "family_name": _d(person.FamilyName) if person.FamilyName else None,
                    "given_name": _d(person.GivenName) if person.GivenName else None,
                }
            org = oh.OwningUser.TheOrganization
            if org:
                owner["organization"] = _d(org.Name) if org.Name else None
        app = oh.OwningApplication
        if app:
            owner["application"] = {
                "name": _d(app.ApplicationFullName),
                "version": app.Version,
                "identifier": app.ApplicationIdentifier,
                "developer": _d(app.ApplicationDeveloper.Name) if app.ApplicationDeveloper else None,
            }
        d["owner_history"] = owner
    psets = _strip_id(element_util.get_psets(p, psets_only=True, should_inherit=False))
    if psets:
        d["property_sets"] = psets
    qtos = _strip_id(element_util.get_psets(p, qtos_only=True, should_inherit=False))
    if qtos:
        d["quantities"] = qtos
    return d


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

_TYPE_CLASSES = [
    # IFC4 / IFC4X3
    "IfcWallType", "IfcDoorType", "IfcWindowType", "IfcSlabType",
    "IfcRoofType", "IfcStairType", "IfcStairFlightType", "IfcColumnType",
    "IfcBeamType", "IfcMemberType", "IfcCoveringType", "IfcRailingType",
    "IfcCurtainWallType", "IfcFurnishingElementType", "IfcFurnitureType",
    "IfcBuildingElementProxyType", "IfcSpaceType",
    "IfcPlateType", "IfcFootingType",
    # MEP
    "IfcPipeSegmentType", "IfcPipeFittingType", "IfcDuctSegmentType",
    "IfcDuctFittingType", "IfcSanitaryTerminalType", "IfcFlowTerminalType",
    "IfcFlowSegmentType", "IfcFlowFittingType", "IfcFlowControllerType",
    "IfcCableCarrierFittingType", "IfcCableCarrierSegmentType",
    # IFC2X3 equivalents
    "IfcDoorStyle", "IfcWindowStyle",
]


def _safe_by_type(f, cls):
    """Schema-safe by_type: returns [] if the entity class doesn't exist in this schema."""
    try:
        return f.by_type(cls)
    except RuntimeError:
        return []


def _expand_nested(elements):
    """Expand element list by adding children aggregated/nested under products."""
    expanded = list(elements)
    seen = {e.id() for e in expanded}
    queue = list(elements)
    while queue:
        parent = queue.pop()
        for rel in getattr(parent, "IsDecomposedBy", []):
            for child in rel.RelatedObjects:
                if child.id() not in seen and child.is_a("IfcProduct") and not child.is_a("IfcSpace"):
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


def convert(ifc_path: str, output_dir: str) -> None:
    print(f"Loading {ifc_path} …")
    f = ifcopenshell.open(ifc_path)

    os.makedirs(output_dir, exist_ok=True)
    meta = os.path.join(output_dir, "__meta__")

    # ── 1. metadata ──────────────────────────────────────────────────────
    print("  [1/8] metadata")
    _write(_header(f), os.path.join(meta, "header.json"))
    _write(_units(f), os.path.join(meta, "units.json"))
    _write(_project(f), os.path.join(meta, "project.json"))

    # ── 2. types ─────────────────────────────────────────────────────────
    print("  [2/8] types")
    types_dir = os.path.join(output_dir, "__types__")
    n_types = 0
    for cls in _TYPE_CLASSES:
        for tobj in _safe_by_type(f, cls):
            td = _serialize_type(tobj)
            fn = _safe(tobj.Name or f"unnamed_{tobj.id()}") + ".json"
            _write(td, os.path.join(types_dir, cls, fn))
            n_types += 1
    # Also catch any IfcTypeProduct not in the predefined list
    for tobj in f.by_type("IfcTypeProduct"):
        cls = tobj.is_a()
        if cls not in _TYPE_CLASSES:
            td = _serialize_type(tobj)
            fn = _safe(tobj.Name or f"unnamed_{tobj.id()}") + ".json"
            _write(td, os.path.join(types_dir, cls, fn))
            n_types += 1
    print(f"        {n_types} type definitions")

    # ── 3. materials ─────────────────────────────────────────────────────
    print("  [3/8] materials")
    mat_dir = os.path.join(output_dir, "__materials__")
    n_mat = 0
    for mat in f.by_type("IfcMaterial"):
        _write(
            {"name": _d(mat.Name), "category": getattr(mat, "Category", None)},
            os.path.join(mat_dir, _safe(mat.Name) + ".json"),
        )
        n_mat += 1
    for mls in f.by_type("IfcMaterialLayerSet"):
        data = {
            "name": _d(mls.LayerSetName) if mls.LayerSetName else None,
            "type": "IfcMaterialLayerSet",
            "layers": [
                {
                    "material": _d(la.Material.Name) if la.Material else None,
                    "thickness": la.LayerThickness,
                }
                for la in mls.MaterialLayers
            ],
        }
        name = mls.LayerSetName or f"layerset_{mls.id()}"
        _write(data, os.path.join(mat_dir, _safe(name) + ".json"))
        n_mat += 1
    print(f"        {n_mat} materials")

    # ── 4. relationship maps ─────────────────────────────────────────────
    print("  [4/8] relationship maps")
    layer_map = _build_layer_map(f)
    host_map = _build_host_map(f)
    space_boundary_map = _build_space_boundary_map(f)
    wall_connection_map = _build_wall_connection_map(f)
    system_map = _build_system_map(f)
    port_map = _build_port_map(f)
    services_map = _build_services_map(f)
    coverings_map, covering_to_host_map = _build_coverings_map(f)
    print(f"        {len(layer_map)} layer, {len(host_map)} host, "
          f"{len(space_boundary_map)} boundary, {len(wall_connection_map)} wall-conn, "
          f"{len(system_map)} system, {len(port_map)} port, {len(services_map)} services, "
          f"{len(coverings_map)} coverings")

    # ── 5. systems ───────────────────────────────────────────────────────
    print("  [5/8] systems")
    systems_dir = os.path.join(output_dir, "__systems__")
    n_systems = 0
    for rel in _safe_by_type(f, "IfcRelAssignsToGroup"):
        group = rel.RelatingGroup
        if not group:
            continue
        if not (group.is_a("IfcSystem") or group.is_a("IfcDistributionSystem")):
            continue
        members = [o for o in rel.RelatedObjects if o.is_a("IfcProduct")]
        member_classes: dict[str, int] = {}
        for m in members:
            member_classes[m.is_a()] = member_classes.get(m.is_a(), 0) + 1
        sd = {
            "id": group.id(),
            "global_id": group.GlobalId,
            "ifc_class": group.is_a(),
            "name": _d(group.Name) if group.Name else None,
            "member_count": len(members),
            "member_classes": member_classes,
            "member_ids": [m.id() for m in members],
        }
        ot = getattr(group, "ObjectType", None)
        if ot:
            sd["object_type"] = _d(ot)
        pt = getattr(group, "PredefinedType", None)
        if pt:
            sd["predefined_type"] = str(pt)
        fn = _safe(group.Name or f"system_{group.id()}") + ".json"
        _write(sd, os.path.join(systems_dir, fn))
        n_systems += 1
    print(f"        {n_systems} systems")

    # ── 6. spatial hierarchy ─────────────────────────────────────────────
    print("  [6/8] spatial hierarchy")

    written_ids: set[int] = set()
    total_elements = 0

    def _get_contained(spatial):
        """Get elements contained in a spatial structure element."""
        return [
            e for rel in getattr(spatial, "ContainsElements", [])
            for e in rel.RelatedElements
        ]

    def _get_spaces(spatial):
        """Get IfcSpace children from both IsDecomposedBy and ContainsElements."""
        spaces = [
            c for rel in getattr(spatial, "IsDecomposedBy", [])
            for c in rel.RelatedObjects if c.is_a("IfcSpace")
        ]
        space_ids = {s.id() for s in spaces}
        for rel in getattr(spatial, "ContainsElements", []):
            for e in rel.RelatedElements:
                if e.is_a("IfcSpace") and e.id() not in space_ids:
                    spaces.append(e)
                    space_ids.add(e.id())
        return spaces

    def _get_storeys(spatial):
        """Get IfcBuildingStorey children, sorted by elevation."""
        storeys = [
            c for rel in getattr(spatial, "IsDecomposedBy", [])
            for c in rel.RelatedObjects if c.is_a("IfcBuildingStorey")
        ]
        storeys.sort(key=lambda s: getattr(s, "Elevation", 0) or 0)
        return storeys

    def _write_elements_grouped(elements, parent_dir, geom_store):
        """Write elements grouped by IFC class, accumulating geometry."""
        nonlocal total_elements
        class_counter: dict[str, int] = defaultdict(int)
        for elem in elements:
            try:
                cls = elem.is_a()
                class_counter[cls] += 1
                ename = _safe(_d(elem.Name) if elem.Name else f"unnamed_{elem.id()}")
                base = f"{class_counter[cls]:04d}__{ename}"
                geom = _extract_geometry(elem)
                if geom:
                    geom_store[str(elem.id())] = geom
                geom_key = str(elem.id()) if geom else None
                ed = _serialize_element(elem, layer_map=layer_map, host_map=host_map,
                                       geometry_key=geom_key,
                                       space_boundary_map=space_boundary_map,
                                       wall_connection_map=wall_connection_map,
                                       system_map=system_map, port_map=port_map,
                                       coverings_map=coverings_map,
                                       covering_to_host_map=covering_to_host_map)
                _write(ed, os.path.join(parent_dir, cls, f"{base}.json"))
            except Exception as exc:
                print(f"        WARNING: failed to serialize {elem.is_a()} #{elem.id()}: {exc}")
            written_ids.add(elem.id())
            total_elements += 1

    def _write_storey(storey, sdir):
        raw_elements = _get_contained(storey)
        elements = [e for e in _expand_nested(raw_elements) if not e.is_a("IfcSpace")]
        spaces = _get_spaces(storey)

        seen_ids = {e.id() for e in elements}
        for sp in spaces:
            for e in _expand_nested(_get_contained(sp)):
                if not e.is_a("IfcSpace") and e.id() not in seen_ids:
                    elements.append(e)
                    seen_ids.add(e.id())

        ecounts: dict[str, int] = {}
        for e in elements:
            ecounts[e.is_a()] = ecounts.get(e.is_a(), 0) + 1

        sdata = {
            "id": storey.id(),
            "global_id": storey.GlobalId,
            "name": _d(storey.Name),
            "description": _d(storey.Description),
            "elevation": storey.Elevation if hasattr(storey, "Elevation") else None,
        }
        if hasattr(storey, "LongName") and storey.LongName:
            sdata["long_name"] = _d(storey.LongName)
        spsets = _strip_id(element_util.get_psets(storey, psets_only=True, should_inherit=False))
        if spsets:
            sdata["property_sets"] = spsets
        sqtos = _strip_id(element_util.get_psets(storey, qtos_only=True, should_inherit=False))
        if sqtos:
            sdata["quantities"] = sqtos
        sdata["element_summary"] = ecounts
        sdata["space_count"] = len(spaces)
        _write(sdata, os.path.join(sdir, "__storey__.json"))

        storey_geom: dict[str, dict] = {}

        for sp in spaces:
            sp_num = _safe(sp.Name or f"unnamed_{sp.id()}")
            sp_long = _safe(_d(sp.LongName) if sp.LongName else (_d(sp.Name) if sp.Name else ""))
            base = f"{sp_num}__{sp_long}"
            geom = _extract_geometry(sp)
            if geom:
                storey_geom[str(sp.id())] = geom
            geom_key = str(sp.id()) if geom else None
            sp_data = _serialize_space(sp, layer_map=layer_map, geometry_key=geom_key,
                                       space_boundary_map=space_boundary_map,
                                       wall_connection_map=wall_connection_map,
                                       system_map=system_map, port_map=port_map)
            _write(sp_data, os.path.join(sdir, "spaces", f"{base}.json"))
            written_ids.add(sp.id())

        _write_elements_grouped(elements, sdir, storey_geom)

        if storey_geom:
            _write(storey_geom, os.path.join(sdir, "__geometry__.json"))

        print(f"        {_safe(storey.Name or 'unnamed')}: {len(elements)} elements, {len(spaces)} spaces")

    def _walk_spatial(entity, base_dir):
        """Recursively walk the IFC spatial decomposition tree."""
        nonlocal total_elements
        for rel in getattr(entity, "IsDecomposedBy", []):
            for child in rel.RelatedObjects:
                if child.is_a("IfcSite"):
                    site_dir = os.path.join(
                        base_dir, "Site__" + _safe(child.Name) if child.Name else "Site__unnamed"
                    )
                    site_data = _serialize_element(child)
                    if hasattr(child, "RefLatitude") and child.RefLatitude:
                        site_data["latitude"] = list(child.RefLatitude)
                    if hasattr(child, "RefLongitude") and child.RefLongitude:
                        site_data["longitude"] = list(child.RefLongitude)
                    if hasattr(child, "RefElevation") and child.RefElevation is not None:
                        site_data["ref_elevation"] = child.RefElevation
                    _write(site_data, os.path.join(site_dir, "__site__.json"))

                    site_elements = _expand_nested(_get_contained(child))
                    if site_elements:
                        site_geom: dict[str, dict] = {}
                        _write_elements_grouped(
                            [e for e in site_elements if not e.is_a("IfcSpace")],
                            site_dir, site_geom,
                        )
                        if site_geom:
                            _write(site_geom, os.path.join(site_dir, "__geometry__.json"))

                    _walk_spatial(child, site_dir)

                elif child.is_a("IfcBuilding"):
                    bldg_dir = os.path.join(
                        base_dir, "Building__" + _safe(child.Name) if child.Name else "Building__unnamed"
                    )
                    bldg_data = _serialize_element(child)
                    if hasattr(child, "LongName") and child.LongName:
                        bldg_data["long_name"] = _d(child.LongName)
                    if hasattr(child, "ElevationOfRefHeight") and child.ElevationOfRefHeight is not None:
                        bldg_data["elevation_of_ref_height"] = child.ElevationOfRefHeight
                    if hasattr(child, "ElevationOfTerrain") and child.ElevationOfTerrain is not None:
                        bldg_data["elevation_of_terrain"] = child.ElevationOfTerrain
                    storeys = _get_storeys(child)
                    bldg_data["storeys"] = [
                        {"name": _d(s.Name), "description": _d(s.Description), "elevation": s.Elevation}
                        for s in storeys
                    ]
                    if child.id() in services_map:
                        bldg_data["serviced_by"] = services_map[child.id()]
                    _write(bldg_data, os.path.join(bldg_dir, "__building__.json"))

                    bldg_elements = _expand_nested(_get_contained(child))
                    if bldg_elements:
                        bldg_geom: dict[str, dict] = {}
                        _write_elements_grouped(bldg_elements, bldg_dir, bldg_geom)
                        if bldg_geom:
                            _write(bldg_geom, os.path.join(bldg_dir, "__geometry__.json"))

                    _walk_spatial(child, bldg_dir)

                elif child.is_a("IfcBuildingStorey"):
                    sname = _safe(child.Name or f"storey_{child.id()}")
                    _write_storey(child, os.path.join(base_dir, sname))

                elif child.is_a("IfcSpace"):
                    sp_num = _safe(child.Name or f"unnamed_{child.id()}")
                    sp_long = _safe(_d(child.LongName) if hasattr(child, "LongName") and child.LongName else "")
                    base_name = f"{sp_num}__{sp_long}" if sp_long else sp_num
                    geom = _extract_geometry(child)
                    geom_key = str(child.id()) if geom else None
                    sp_data = _serialize_space(child, layer_map=layer_map, geometry_key=geom_key,
                                               space_boundary_map=space_boundary_map,
                                               wall_connection_map=wall_connection_map,
                                               system_map=system_map, port_map=port_map)
                    _write(sp_data, os.path.join(base_dir, "spaces", f"{base_name}.json"))
                    written_ids.add(child.id())
                    total_elements += 1

                    sp_elems = [e for e in _expand_nested(_get_contained(child))
                                if not e.is_a("IfcSpace")]
                    if sp_elems:
                        sp_geom: dict[str, dict] = {}
                        _write_elements_grouped(sp_elems, base_dir, sp_geom)
                        if sp_geom:
                            geom_path = os.path.join(base_dir, "__geometry__.json")
                            if os.path.exists(geom_path):
                                with open(geom_path) as gf:
                                    existing = json.load(gf)
                                existing.update(sp_geom)
                                _write(existing, geom_path)
                            else:
                                _write(sp_geom, geom_path)

                else:
                    _walk_spatial(child, base_dir)

    projects = f.by_type("IfcProject")
    if projects:
        _walk_spatial(projects[0], output_dir)

    # ── 6. uncontained elements and spaces ──────────────────────────────
    port_ids = {pid for entries in port_map.values() for e in entries for pid in [e["id"]]}
    orphaned = [
        p for p in f.by_type("IfcProduct")
        if not p.is_a("IfcOpeningElement") and not p.is_a("IfcProject")
        and not p.is_a("IfcSite") and not p.is_a("IfcBuilding")
        and not p.is_a("IfcBuildingStorey")
        and not p.is_a("IfcDistributionPort")
        and not p.is_a("IfcVirtualElement")
        and p.id() not in written_ids
    ]
    if orphaned:
        undir = os.path.join(output_dir, "__uncontained__")
        un_geom: dict[str, dict] = {}
        class_counter: dict[str, int] = defaultdict(int)
        for elem in orphaned:
            geom = _extract_geometry(elem)
            if geom:
                un_geom[str(elem.id())] = geom
            geom_key = str(elem.id()) if geom else None
            if elem.is_a("IfcSpace"):
                sp_num = _safe(elem.Name or f"unnamed_{elem.id()}")
                sp_long = _safe(_d(elem.LongName) if hasattr(elem, "LongName") and elem.LongName else (_d(elem.Name) if elem.Name else ""))
                base = f"{sp_num}__{sp_long}"
                sp_data = _serialize_space(elem, layer_map=layer_map, geometry_key=geom_key,
                                           space_boundary_map=space_boundary_map,
                                           wall_connection_map=wall_connection_map,
                                           system_map=system_map, port_map=port_map)
                _write(sp_data, os.path.join(undir, "spaces", f"{base}.json"))
                sp_contained = [e for e in _expand_nested(_get_contained(elem))
                                if not e.is_a("IfcSpace") and e.id() not in written_ids]
                for sc in sp_contained:
                    sc_geom = _extract_geometry(sc)
                    if sc_geom:
                        un_geom[str(sc.id())] = sc_geom
                    sc_geom_key = str(sc.id()) if sc_geom else None
                    sc_cls = sc.is_a()
                    class_counter[sc_cls] += 1
                    sc_name = _safe(_d(sc.Name) if sc.Name else f"unnamed_{sc.id()}")
                    sc_base = f"{class_counter[sc_cls]:04d}__{sc_name}"
                    sc_ed = _serialize_element(sc, layer_map=layer_map, host_map=host_map,
                                               geometry_key=sc_geom_key,
                                               space_boundary_map=space_boundary_map,
                                               wall_connection_map=wall_connection_map,
                                               system_map=system_map, port_map=port_map,
                                               coverings_map=coverings_map,
                                               covering_to_host_map=covering_to_host_map)
                    _write(sc_ed, os.path.join(undir, sc_cls, f"{sc_base}.json"))
                    written_ids.add(sc.id())
                    total_elements += 1
            else:
                cls = elem.is_a()
                class_counter[cls] += 1
                ename = _safe(_d(elem.Name) if elem.Name else f"unnamed_{elem.id()}")
                base = f"{class_counter[cls]:04d}__{ename}"
                ed = _serialize_element(elem, layer_map=layer_map, host_map=host_map,
                                        geometry_key=geom_key,
                                        space_boundary_map=space_boundary_map,
                                        wall_connection_map=wall_connection_map,
                                        system_map=system_map, port_map=port_map,
                                        coverings_map=coverings_map,
                                        covering_to_host_map=covering_to_host_map)
                _write(ed, os.path.join(undir, cls, f"{base}.json"))
            written_ids.add(elem.id())
            total_elements += 1
        if un_geom:
            _write(un_geom, os.path.join(undir, "__geometry__.json"))
        print(f"        __uncontained__: {len(orphaned)} elements")

    print(f"  [7/8] done — {total_elements} elements total")

    # ── 8. summary ───────────────────────────────────────────────────────
    n_files = sum(len(fs) for _, _, fs in os.walk(output_dir))
    n_dirs = sum(1 for _ in os.walk(output_dir))
    n_geom = sum(1 for _, _, fs in os.walk(output_dir) for f_ in fs if f_ == "__geometry__.json")
    print(f"  [8/8] summary")
    print(f"        output:   {output_dir}")
    print(f"        files:    {n_files} ({n_geom} consolidated geometry files)")
    print(f"        dirs:     {n_dirs}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert an IFC file to a filesystem representation for agent-based querying.",
    )
    ap.add_argument("ifc_file", help="Path to the .ifc file")
    ap.add_argument("-o", "--output", help="Output directory (default: <stem>_fs)")
    args = ap.parse_args()

    out = args.output or (Path(args.ifc_file).stem + "_fs")
    convert(args.ifc_file, out)
