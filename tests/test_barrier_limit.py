"""Tests for limits/barrier_limit.py."""

import numpy as np
from absl.testing import absltest

from mink import (
    SE3,
    BarrierLimit,
    CollisionAvoidanceLimit,
    ConfigurationLimit,
    FrameTask,
    GeomPairBarrier,
    PointObstacleBarrier,
    solve_ik,
)
from mink.exceptions import LimitDefinitionError

from .barrier_utils import ROBOT_GEOMS, arm_model, configuration_at


def _inequalities(limit, configuration, dt) -> tuple[np.ndarray, np.ndarray]:
    G, h = limit.compute_qp_inequalities(configuration, dt)
    assert G is not None and h is not None
    return G, h


class TestBarrierLimit(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = arm_model()

    def setUp(self):
        self.configuration = configuration_at(self.model, [0.3, -0.5, 0.2])
        self.barrier = PointObstacleBarrier(self.model, ROBOT_GEOMS)

    def test_inactive_without_rows(self):
        limit = BarrierLimit(self.barrier, minimum_distance=0.02, gain=10.0)
        self.assertTrue(
            limit.compute_qp_inequalities(self.configuration, 0.01).inactive
        )

    def test_constraint_values(self):
        self.barrier.set_obstacles([[0.15, 0.12, 0.0], [0.15, 0.04, 0.0]])
        rows = self.barrier.compute_rows(self.configuration)
        dt, gain, d_min = 0.01, 10.0, 0.02
        limit = BarrierLimit(self.barrier, minimum_distance=d_min, gain=gain)
        G, h = _inequalities(limit, self.configuration, dt)
        np.testing.assert_allclose(G, -rows.jacobian)
        margin = rows.distance - d_min
        expected = np.where(
            margin > 0, np.minimum(dt * gain * margin, margin), dt * gain * margin
        )
        np.testing.assert_allclose(h, expected)
        self.assertTrue(np.any(h < 0.0))  # One obstacle is inside the minimum distance.
        limit.relaxation = 0.5
        _, half = _inequalities(limit, self.configuration, dt)
        np.testing.assert_allclose(
            half, np.where(expected < 0, 0.5 * expected, expected)
        )
        limit.relaxation = 1.0
        _, relaxed = _inequalities(limit, self.configuration, dt)
        np.testing.assert_allclose(relaxed, np.maximum(expected, 0.0))

    def test_bound_is_capped_at_the_margin(self):
        self.barrier.set_obstacles([[0.15, 0.2, 0.0]])
        limit = BarrierLimit(self.barrier, minimum_distance=0.0, gain=1000.0)
        _, h = _inequalities(limit, self.configuration, 0.01)
        rows = self.barrier.compute_rows(self.configuration)
        np.testing.assert_allclose(h, rows.distance)

    def test_gain_function(self):
        self.barrier.set_obstacles([[0.15, 0.2, 0.0]])
        limit = BarrierLimit(
            self.barrier, minimum_distance=0.0, gain=2.0, gain_function=np.sqrt
        )
        _, h = _inequalities(limit, self.configuration, 0.01)
        rows = self.barrier.compute_rows(self.configuration)
        np.testing.assert_allclose(h, 0.01 * 2.0 * np.sqrt(rows.distance))

    def test_invalid_parameters(self):
        with self.assertRaises(LimitDefinitionError):
            BarrierLimit(self.barrier, gain=0.0)
        with self.assertRaises(LimitDefinitionError):
            BarrierLimit(
                self.barrier, minimum_distance=self.barrier.activation_distance
            )

    def test_matches_collision_avoidance_limit_outside_the_minimum(self):
        pairs = [(list(ROBOT_GEOMS), ["obstacle"])]
        dt, gain, d_min, d_act = 0.01, 20.0, 0.02, 0.5
        barrier = GeomPairBarrier(self.model, pairs, activation_distance=d_act)
        ours = BarrierLimit(barrier, minimum_distance=d_min, gain=gain)
        theirs = CollisionAvoidanceLimit(
            self.model,
            pairs,
            gain=gain * dt,
            minimum_distance_from_collisions=d_min,
            collision_detection_distance=d_act,
        )
        G1, h1 = _inequalities(ours, self.configuration, dt)
        G2, h2 = _inequalities(theirs, self.configuration, dt)
        finite = np.isfinite(h2)
        order = np.argsort(
            [theirs.geom_id_pairs.index(p) for p in barrier.geom_id_pairs]
        )
        self.assertTrue(
            np.all(barrier.compute_rows(self.configuration).distance > d_min)
        )
        np.testing.assert_allclose(h1[order], h2[finite], atol=1e-12)
        np.testing.assert_allclose(G1[order], G2[finite], atol=1e-9)

    def test_ik_never_crosses_the_minimum_distance(self):
        # Pull the tip straight through a sphere obstacle that the barrier sees as a
        # point with a radius; the arm must stop outside the minimum distance.
        centre, radius, d_min = np.array([0.55, 0.0, 0.0]), 0.05, 0.02
        configuration = configuration_at(self.model, [0.6, 0.0, 0.0])
        barrier = PointObstacleBarrier(self.model, ROBOT_GEOMS)
        barrier.set_obstacles([centre], radius)
        limits = [
            ConfigurationLimit(self.model),
            BarrierLimit(barrier, minimum_distance=d_min, gain=20.0),
        ]
        task = FrameTask("tip", "site", position_cost=1.0, orientation_cost=0.0)
        task.set_target(SE3.from_translation(np.array([0.7, -0.05, 0.0])))
        dt, closest = 0.01, np.inf
        for _ in range(400):
            v = solve_ik(configuration, [task], dt, "daqp", 1e-3, limits=limits)
            configuration.integrate_inplace(v, dt)
            rows = barrier.compute_rows(configuration)
            if rows.size:
                closest = min(closest, rows.distance.min())
        # Linearisation leaves second-order slack only.
        self.assertGreater(closest, d_min - 2e-3)
        self.assertLess(closest, d_min + 0.02)


if __name__ == "__main__":
    absltest.main()
