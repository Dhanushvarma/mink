"""Tests for barriers/point_obstacle_barrier.py."""

import numpy as np
from absl.testing import absltest

from mink import PointObstacleBarrier
from mink.barriers import point_capsule_distances

from .barrier_utils import (
    ROBOT_GEOMS,
    arm_model,
    configuration_at,
    finite_difference_jacobian,
)


class TestPointObstacleBarrier(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = arm_model()

    def setUp(self):
        self.configuration = configuration_at(self.model, [0.3, -0.5, 0.2])
        self.barrier = PointObstacleBarrier(
            self.model, ROBOT_GEOMS, activation_distance=0.3, min_spacing=0.0
        )

    def test_no_obstacles_no_rows(self):
        rows = self.barrier.compute_rows(self.configuration)
        self.assertEqual(rows.size, 0)
        self.assertEqual(rows.jacobian.shape, (0, self.model.nv))

    def test_distance_and_witnesses(self):
        centre, radius = np.array([0.15, 0.2, 0.0]), 0.02
        self.barrier.set_obstacles([centre], radius)
        rows = self.barrier.compute_rows(self.configuration)
        a, b = self.barrier.capsules.segments(self.configuration.data)
        dist, _ = point_capsule_distances(centre, a, b, self.barrier.capsules.radius)
        expected = np.sort(dist[0] - radius)
        expected = expected[expected < 0.3]
        np.testing.assert_allclose(np.sort(rows.distance), expected, atol=1e-12)
        for d, (robot, obstacle) in zip(rows.distance, rows.witness):
            self.assertAlmostEqual(np.linalg.norm(obstacle - robot), d, places=12)
            self.assertAlmostEqual(np.linalg.norm(obstacle - centre), radius)

    def test_jacobian_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        centres = rng.uniform([-0.1, -0.3, -0.2], [0.7, 0.3, 0.2], size=(8, 3))
        self.barrier.set_obstacles(centres, 0.01)
        self.barrier.max_rows_per_geom = 1
        rows = self.barrier.compute_rows(self.configuration)
        self.assertGreater(rows.size, 0)
        capsules = self.barrier.capsules

        def distances(configuration):
            a, b = capsules.segments(configuration.data)
            return point_capsule_distances(centres, a, b, capsules.radius)[0] - 0.01

        at_q0 = distances(self.configuration)
        for k in range(rows.size):
            # The (obstacle, geom) pair this row was built from.
            i, g = np.unravel_index(
                np.argmin(np.abs(at_q0 - rows.distance[k])), at_q0.shape
            )
            numeric = finite_difference_jacobian(
                lambda c, i=i, g=g: distances(c)[i, g], self.configuration
            )
            np.testing.assert_allclose(rows.jacobian[k], numeric, atol=1e-5)

    def test_activation_distance(self):
        self.barrier.set_obstacles([[0.0, 0.0, 5.0]])
        self.assertEqual(self.barrier.compute_rows(self.configuration).size, 0)

    def test_rows_per_geom_and_spacing(self):
        # A dense cluster beside the first link: many points, few distinct rows.
        x = np.linspace(0.05, 0.25, 21)
        cloud = np.stack([x, np.full_like(x, 0.1), np.zeros_like(x)], axis=1)
        self.configuration.update(np.zeros(3))
        barrier = PointObstacleBarrier(
            self.model, ["cap1"], max_rows_per_geom=3, min_spacing=0.05
        )
        barrier.set_obstacles(cloud)
        rows = barrier.compute_rows(self.configuration)
        self.assertEqual(rows.size, 3)
        picked = rows.witness[:, 1]
        gaps = np.linalg.norm(picked[:, None] - picked[None], axis=2)
        self.assertGreaterEqual(gaps[np.triu_indices(3, 1)].min(), 0.05 - 1e-12)
        barrier.min_spacing = 0.0
        barrier.max_rows_per_geom = 50
        barrier.set_obstacles(cloud)
        self.assertEqual(barrier.compute_rows(self.configuration).size, 21)

    def test_rows_are_cached_until_something_changes(self):
        self.barrier.set_obstacles([[0.15, 0.2, 0.0]])
        rows = self.barrier.compute_rows(self.configuration)
        self.assertIs(self.barrier.compute_rows(self.configuration), rows)
        self.barrier.set_obstacles([[0.15, 0.25, 0.0]])
        moved = self.barrier.compute_rows(self.configuration)
        self.assertIsNot(moved, rows)
        self.configuration.update(self.configuration.q + 0.01)
        self.assertIsNot(self.barrier.compute_rows(self.configuration), moved)

    def test_set_obstacles_validates(self):
        with self.assertRaises(ValueError):
            self.barrier.set_obstacles([[0.0, 0.0, 0.0]], radii=-0.1)
        self.barrier.set_obstacles(np.zeros((4, 3)), radii=0.05)
        np.testing.assert_allclose(self.barrier.radii, np.full(4, 0.05))
        self.barrier.clear_obstacles()
        self.assertEqual(len(self.barrier.centers), 0)
        with self.assertRaises(ValueError):
            PointObstacleBarrier(self.model, ROBOT_GEOMS, max_rows_per_geom=0)
        with self.assertRaises(ValueError):
            PointObstacleBarrier(self.model, ROBOT_GEOMS, min_spacing=-1.0)


if __name__ == "__main__":
    absltest.main()
