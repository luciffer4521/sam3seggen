"""X-Part first; swap a large, box-escaping solid for that instance's HoloPart draw.

HoloPart is only loaded when at least one instance qualifies. Pairing is by the
``00_name`` prefix written into ``xpart_instances.glb`` / ``open_instances.glb``.
"""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh

from xpart_complete import box_escape, group_solids, load_part_nodes

ESCAPE_LIMIT = 0.5
AREA_LARGE = 0.08
AXIS_LARGE = 0.55


def source_extent(boxes):
    lows = np.min([np.array(row["box"])[0] for row in boxes], axis=0)
    highs = np.max([np.array(row["box"])[1] for row in boxes], axis=0)
    return highs - lows


def instance_index(name):
    prefix = str(name).split("_", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def index_by_instance(nodes):
    mapping = {}
    for name, mesh in nodes:
        index = instance_index(name)
        if index is not None:
            mapping[index] = (name, mesh)
    if mapping:
        return mapping
    return {index: node for index, node in enumerate(nodes)}


def is_large(row, src_ext, area_large=AREA_LARGE, axis_large=AXIS_LARGE):
    box = np.array(row["box"])
    axis = float(np.max((box[1] - box[0]) / np.maximum(src_ext, 1e-9)))
    return row["area_share"] >= area_large or axis >= axis_large, axis


def decide_backend(row, generated_bounds, src_ext, escape_limit=ESCAPE_LIMIT,
                   area_large=AREA_LARGE, axis_large=AXIS_LARGE):
    """One instance: HoloPart only if it is large *and* the X-Part solid left its box."""
    if generated_bounds is None:
        escape = float("inf")
    else:
        escape = box_escape(generated_bounds, np.array(row["box"]))
    large, axis = is_large(row, src_ext, area_large, axis_large)
    use_holo = large and escape > escape_limit
    return {
        "name": row["name"],
        "instance": row.get("instance"),
        "escape": escape if np.isfinite(escape) else None,
        "area_share": row["area_share"],
        "max_axis_of_source": axis,
        "large": large,
        "backend": "holopart" if use_holo else "xpart",
    }


def decisions_from_instances(instances_glb, boxes, escape_limit=ESCAPE_LIMIT,
                             area_large=AREA_LARGE, axis_large=AXIS_LARGE):
    src_ext = source_extent(boxes)
    indexed = index_by_instance(load_part_nodes(instances_glb))
    decisions = []
    for row in boxes:
        inst = row.get("instance")
        node = indexed.get(inst) if inst is not None else None
        bounds = None if node is None else node[1].bounds
        decision = decide_backend(
            row, bounds, src_ext, escape_limit, area_large, axis_large)
        decision["xpart_node"] = None if node is None else node[0]
        decisions.append(decision)
    return decisions


def pick_solids(xpart_nodes, holo_nodes, boxes, decisions):
    xpart = index_by_instance(xpart_nodes)
    holo = index_by_instance(holo_nodes) if holo_nodes else {}
    names, solids = [], []
    for row, decision in zip(boxes, decisions):
        inst = row.get("instance")
        xnode = xpart.get(inst)
        hnode = holo.get(inst)
        if decision["backend"] == "holopart" and hnode is not None:
            mesh = hnode[1]
        elif xnode is not None:
            mesh = xnode[1]
        elif hnode is not None:
            mesh = hnode[1]
        else:
            continue
        names.append(row["name"])
        solids.append(mesh)
    return names, solids


def write_assembled(out_dir, names, solids, grouped_name="xpart_parts.glb"):
    instances = trimesh.Scene()
    for index, (name, mesh) in enumerate(zip(names, solids)):
        instances.add_geometry(mesh, geom_name=f"{index:02d}_{name}")
    instances_path = os.path.join(out_dir, "hybrid_instances.glb")
    instances.export(instances_path)
    grouped = trimesh.Scene()
    for name, mesh in group_solids(names, solids):
        grouped.add_geometry(mesh, geom_name=name)
    grouped_path = os.path.join(out_dir, grouped_name)
    grouped.export(grouped_path)
    return grouped_path, instances_path


def apply_hybrid(out_dir, holopart_glb=None, escape_limit=ESCAPE_LIMIT,
                 area_large=AREA_LARGE, axis_large=AXIS_LARGE):
    """Read X-Part artifacts, optionally a HoloPart glb, write decisions + assembled parts.

    Returns (decisions, swapped). ``holopart_glb`` may be omitted when nothing needs it.
    """
    boxes_path = os.path.join(out_dir, "boxes.json")
    instances_glb = os.path.join(out_dir, "xpart_instances.glb")
    with open(boxes_path, "r", encoding="utf-8") as handle:
        boxes = json.load(handle)
    decisions = decisions_from_instances(
        instances_glb, boxes, escape_limit, area_large, axis_large)
    holo_nodes = load_part_nodes(holopart_glb) if holopart_glb else []
    holo_index = index_by_instance(holo_nodes) if holo_nodes else {}
    missing = [row["name"] for row in decisions
               if row["backend"] == "holopart" and row.get("instance") not in holo_index]
    if missing:
        raise SystemExit(f"HoloPart missing instance(s): {missing}")
    names, solids = pick_solids(
        load_part_nodes(instances_glb), holo_nodes, boxes, decisions)
    write_assembled(out_dir, names, solids)
    with open(os.path.join(out_dir, "decisions.json"), "w", encoding="utf-8") as handle:
        json.dump(decisions, handle, indent=2)
    swapped = sum(1 for row in decisions if row["backend"] == "holopart")
    for row in decisions:
        mark = "HOLOPART" if row["backend"] == "holopart" else "xpart"
        escape = "   n/a" if row["escape"] is None else f"{row['escape']:6.1%}"
        print(f"  {row['name']:<18} escape={escape}  area={row['area_share']:.1%}  "
              f"axis={row['max_axis_of_source']:.2f}  -> {mark}")
    print(f"[hybrid] {swapped}/{len(decisions)} solids from HoloPart")
    return decisions, swapped
