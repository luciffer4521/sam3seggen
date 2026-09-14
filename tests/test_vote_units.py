import unittest
from unittest import mock

import numpy as np

from data_toolkit import unit_vote
from data_toolkit.lift_sam3 import MaskSet
from data_toolkit.unit_vote import fold_fragment_units


def two_boxes():
    """Two disjoint boxes carrying one atom label, so split_units would find two units."""
    import trimesh

    mesh = trimesh.util.concatenate(
        trimesh.creation.box(), trimesh.creation.box().apply_translation([5, 0, 0]))
    return mesh, np.zeros(len(mesh.faces), dtype=np.int64)


def one_view_masks(resolution, concept="head"):
    """A single view whose whole silhouette is claimed by one concept."""
    covered = np.ones((1, 1, resolution, resolution), dtype=bool)
    return MaskSet(
        masks=covered,
        foreground=covered[:, 0].copy(),
        scores=np.ones((1, 1)),
        concepts=[concept], owners=[concept], views=["v0"],
        part_order=[concept], unassigned_to=None,
    )


class VoteUnitsTest(unittest.TestCase):
    """The vote must name the partition the split exported, not one it recomputed."""

    def setUp(self):
        self.mesh, self.atoms = two_boxes()
        self.resolution = 8
        # Every pixel belongs to face 1, so the raster never needs a GPU here.
        self.raster = np.ones((1, self.resolution, self.resolution), dtype=np.int32)

    def run_vote(self, units):
        with mock.patch.object(unit_vote, "rasterize_face_ids", return_value=self.raster), \
             mock.patch.object(unit_vote, "split_units",
                               side_effect=AssertionError("split_units was recomputed")) as split:
            if units is None:
                split.side_effect = None
                split.return_value = np.zeros(len(self.mesh.faces), dtype=np.int64)
            _, _, out = unit_vote.vote(
                self.mesh, self.atoms, one_view_masks(self.resolution),
                cameras=[None], camera_angle_x=0.7, resolution=self.resolution,
                part_order=["head"], units=units)
        return out, split

    def test_given_units_are_used_verbatim(self):
        units = np.arange(len(self.mesh.faces)) // 6
        out, split = self.run_vote(units)
        np.testing.assert_array_equal(out, units)
        split.assert_not_called()

    def test_units_are_still_computed_when_none_are_given(self):
        out, split = self.run_vote(None)
        split.assert_called_once()
        self.assertEqual(len(np.unique(out)), 1)


class HangOffTest(unittest.TestCase):
    """A sliver no mask claimed should join the part it hangs off, not the dump bucket."""

    def assign(self, votes, neighbors, faces, unassigned_to="torso", hang_share=0.1):
        names = ["head", "torso"]
        n = len(votes)
        unit_of_face = np.concatenate(
            [np.full(count, unit, dtype=np.int64) for unit, count in enumerate(faces)])
        mesh = mock.Mock()
        mesh.triangles_center = np.zeros((len(unit_of_face), 3))
        assignment, _ = unit_vote.assign_units(
            mesh, unit_of_face, names, np.asarray(votes), np.asarray(votes, dtype=float),
            seen=np.ones(n, dtype=int), unassigned_to=unassigned_to,
            neighbors=neighbors, hang_share=hang_share)
        return assignment

    def test_whiskers_that_only_touch_the_head_join_the_head(self):
        # unit 0 = head (voted), unit 1 = whiskers (4% of the head, only touches it)
        self.assertEqual(
            self.assign([[1, 0], [0, 0]], [[1], [0]], [1000, 40]),
            ["head", "head"])

    def test_a_body_that_also_touches_the_head_stays_with_unassigned(self):
        # The neck is a shared edge; dumping every unvoted neighbour of the head
        # onto it would swallow the torso. The body also touches the hands (unit 2).
        self.assertEqual(
            self.assign([[1, 0], [0, 0], [0, 0]], [[1], [0, 2], [1]], [1000, 400, 200]),
            ["head", "torso", "torso"])

    def test_a_large_piece_that_only_touches_the_head_is_still_a_part(self):
        # A torso whose only neighbour is the head -- SAM3 missed it entirely --
        # must not be absorbed just because it hangs off something named.
        self.assertEqual(
            self.assign([[1, 0], [0, 0]], [[1], [0]], [1000, 400]),
            ["head", "torso"])


class FoldFragmentsTest(unittest.TestCase):
    """A path of faces: two large same-name parts, a same-name chip, a named speck."""

    def fold(self, faces, names, max_share=0.05):
        units = np.concatenate(
            [np.full(count, unit, dtype=np.int64) for unit, count in enumerate(faces)])
        adjacency = np.stack([np.arange(len(units) - 1), np.arange(1, len(units))], axis=1)
        areas = np.ones(len(units), dtype=np.float64)
        return fold_fragment_units(units, areas, adjacency, names, max_share=max_share)

    def test_same_name_chip_folds_large_twins_stay_apart(self):
        # 40 + 2 + 40 faces, all named body: the chip joins a neighbour, the two
        # large bodies stay two nodes (merge=name would weld all three).
        compact, names, absorbed = self.fold([40, 2, 40], ["body", "body", "body"])
        self.assertEqual(absorbed, 1)
        self.assertEqual(len(np.unique(compact)), 2)
        self.assertEqual(names, ["body", "body"])

    def test_a_small_uniquely_named_part_is_kept(self):
        compact, names, absorbed = self.fold([50, 2, 50], ["body", "button", "body"])
        self.assertEqual(absorbed, 0)
        self.assertEqual(len(np.unique(compact)), 3)
        self.assertEqual(names, ["body", "button", "body"])

    def test_an_unnamed_speck_joins_its_neighbour(self):
        compact, names, absorbed = self.fold([50, 2, 50], ["body", None, "leg"])
        self.assertEqual(absorbed, 1)
        self.assertEqual(len(np.unique(compact)), 2)
        self.assertEqual(set(names), {"body", "leg"})

    def test_a_tighter_share_keeps_the_same_chip(self):
        # 2 / 82 ≈ 2.4%. Folds at 5%, stays at 1%.
        _, _, absorbed_loose = self.fold([40, 2, 40], ["body", "body", "body"], max_share=0.05)
        _, _, absorbed_tight = self.fold([40, 2, 40], ["body", "body", "body"], max_share=0.01)
        self.assertEqual(absorbed_loose, 1)
        self.assertEqual(absorbed_tight, 0)

    def test_zero_share_folds_nothing(self):
        compact, names, absorbed = self.fold([40, 2, 40], ["body", "body", "body"], max_share=0)
        self.assertEqual(absorbed, 0)
        self.assertEqual(len(np.unique(compact)), 3)
        self.assertEqual(names, ["body", "body", "body"])


if __name__ == "__main__":
    unittest.main()
