import json
import os
import tempfile
import unittest

import numpy as np
import trimesh

from hybrid_complete import (
    AREA_LARGE, AXIS_LARGE, ESCAPE_LIMIT, apply_hybrid, decide_backend,
    index_by_instance, pick_solids, source_extent,
)
from pipeline import COMPLETE_MODES, DEFAULT_COMPLETE


def box_row(name, instance, low, high, area_share):
    return {
        "name": name,
        "instance": instance,
        "area_share": area_share,
        "box": [list(low), list(high)],
    }


def cube(center, size, name):
    mesh = trimesh.creation.box(extents=[size, size, size])
    mesh.apply_translation(center)
    return name, mesh


class HybridDecisionTest(unittest.TestCase):
    def test_hybrid_is_the_default_complete_mode(self):
        self.assertEqual(DEFAULT_COMPLETE, "hybrid")
        self.assertEqual(COMPLETE_MODES, ("off", "boxes", "full", "hybrid"))

    def test_a_large_solid_that_left_its_box_is_holopart(self):
        # Robot torso: chest box is already ~90% of the source width, X-Part drew a body.
        src = source_extent([box_row("torso", 0, [-0.5, -0.5, -0.2], [0.5, 0.5, 0.2], 0.4)])
        row = box_row("torso", 0, [-0.45, -0.05, -0.09], [0.26, 0.42, 0.19], 0.35)
        generated = np.array([[-0.5, -0.5, -0.2], [0.5, 0.5, 0.2]])
        decision = decide_backend(row, generated, src)
        self.assertTrue(decision["large"])
        self.assertGreater(decision["escape"], ESCAPE_LIMIT)
        self.assertEqual(decision["backend"], "holopart")

    def test_a_large_solid_that_fits_stays_xpart(self):
        src = source_extent([box_row("leg", 0, [-0.1, -0.5, -0.1], [0.1, 0.0, 0.1], 0.2)])
        row = box_row("leg", 0, [-0.1, -0.5, -0.1], [0.1, 0.0, 0.1], 0.2)
        generated = np.array([[-0.11, -0.51, -0.11], [0.11, 0.0, 0.11]])
        decision = decide_backend(row, generated, src)
        self.assertTrue(decision["large"])
        self.assertLess(decision["escape"], ESCAPE_LIMIT)
        self.assertEqual(decision["backend"], "xpart")

    def test_a_small_runaway_hand_stays_xpart(self):
        src = source_extent([box_row("body", 0, [-1, -1, -1], [1, 1, 1], 0.9)])
        row = box_row("hand", 1, [0.7, 0.0, 0.0], [0.85, 0.1, 0.1], 0.02)
        generated = np.array([[0.5, -0.2, -0.2], [1.2, 0.4, 0.4]])
        decision = decide_backend(row, generated, src)
        self.assertFalse(decision["large"])
        self.assertGreater(decision["escape"], ESCAPE_LIMIT)
        self.assertLess(row["area_share"], AREA_LARGE)
        self.assertLess(decision["max_axis_of_source"], AXIS_LARGE)
        self.assertEqual(decision["backend"], "xpart")


class HybridAssembleTest(unittest.TestCase):
    def test_pairing_is_by_instance_prefix_not_dict_order(self):
        nodes = [cube([0, 0, 0], 0.2, "01_arm"), cube([1, 0, 0], 0.4, "00_head")]
        indexed = index_by_instance(nodes)
        self.assertEqual(indexed[0][0], "00_head")
        self.assertEqual(indexed[1][0], "01_arm")

    def test_assemble_swaps_only_the_marked_instance(self):
        boxes = [
            box_row("head", 0, [-0.1, 0.4, -0.1], [0.1, 0.5, 0.1], 0.1),
            box_row("torso", 1, [-0.4, -0.1, -0.2], [0.4, 0.4, 0.2], 0.4),
        ]
        xpart = [cube([0, 0.45, 0], 0.08, "00_head"), cube([0, 0, 0], 2.0, "01_torso")]
        holo = [cube([0, 0.45, 0], 0.08, "00_head"), cube([0, 0.15, 0], 0.6, "01_torso")]
        src = source_extent(boxes)
        decisions = [
            decide_backend(boxes[0], xpart[0][1].bounds, src),
            decide_backend(boxes[1], xpart[1][1].bounds, src),
        ]
        self.assertEqual(decisions[0]["backend"], "xpart")
        self.assertEqual(decisions[1]["backend"], "holopart")
        names, solids = pick_solids(xpart, holo, boxes, decisions)
        self.assertEqual(names, ["head", "torso"])
        np.testing.assert_allclose(solids[0].bounds, xpart[0][1].bounds)
        np.testing.assert_allclose(solids[1].bounds, holo[1][1].bounds)

    def test_apply_hybrid_writes_decisions_without_holopart_when_nothing_escapes(self):
        boxes = [box_row("head", 0, [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1], 0.5)]
        directory = tempfile.mkdtemp()
        scene = trimesh.Scene()
        scene.add_geometry(cube([0, 0, 0], 0.2, "mesh")[1], geom_name="00_head")
        scene.export(os.path.join(directory, "xpart_instances.glb"))
        with open(os.path.join(directory, "boxes.json"), "w", encoding="utf-8") as handle:
            json.dump(boxes, handle)
        decisions, swapped = apply_hybrid(directory)
        self.assertEqual(swapped, 0)
        self.assertEqual(decisions[0]["backend"], "xpart")
        self.assertTrue(os.path.isfile(os.path.join(directory, "decisions.json")))
        self.assertTrue(os.path.isfile(os.path.join(directory, "xpart_parts.glb")))


if __name__ == "__main__":
    unittest.main()
