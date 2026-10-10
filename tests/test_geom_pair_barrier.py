"""Tests for barriers/geom_pair_barrier.py."""

import mujoco
import numpy as np
from absl.testing import absltest

from mink import GeomPairBarrier

from .barrier_utils import (
    ROBOT_GEOMS,
    arm_model,
    configuration_at,
    finite_difference_jacobian,
)


class TestGeomPairBarrier(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = arm_model()

    def setUp(self):
        self.configuration = configuration_at(self.model, [0.4, -0.3, 0.2])
        self.barrier = GeomPairBarrier(
            self.model, [(list(ROBOT_GEOMS), ["obstacle"])], activation_distance=0.5
        )

    def test_pairs_include_the_mocap_obstacle(self):
        obstacle = self.model.geom("obstacle").id
        self.assertLen(self.barrier.geom_id_pairs, 3)
        self.assertTrue(all(obstacle in pair for pair in self.barrier.geom_id_pairs))

    def test_distances_match_mj_geom_distance(self):
        rows = self.barrier.compute_rows(self.configuration)
        self.assertEqual(rows.size, 3)
        data = self.configuration.data
        for (g1, g2), d, witness in zip(
            self.barrier.geom_id_pairs, rows.distance, rows.witness
        ):
            fromto = np.zeros(6)
            expected = mujoco.mj_geomDistance(self.model, data, g1, g2, 0.5, fromto)
            self.assertAlmostEqual(d, expected, places=9)
            np.testing.assert_allclose(witness.ravel(), fromto, atol=1e-9)

    def test_jacobian_matches_finite_differences(self):
        rows = self.barrier.compute_rows(self.configuration)
        for k, (g1, g2) in enumerate(self.barrier.geom_id_pairs):

            def distance(configuration, g1=g1, g2=g2):
                return mujoco.mj_geomDistance(
                    self.model, configuration.data, g1, g2, 10.0, None
                )

            numeric = finite_difference_jacobian(distance, self.configuration)
            np.testing.assert_allclose(rows.jacobian[k], numeric, atol=1e-5)

    def test_penetration_flips_the_normal(self):
        # Put the obstacle into the first link: the Jacobian must still be d(dist)/dq.
        data = self.configuration.data
        data.mocap_pos[0] = data.geom_xpos[self.model.geom("cap1").id] + [0, 0.03, 0]
        self.configuration.update()
        rows = self.barrier.compute_rows(self.configuration)
        cap1 = self.model.geom("cap1").id
        k = [i for i, pair in enumerate(self.barrier.geom_id_pairs) if cap1 in pair][0]
        self.assertLess(rows.distance[k], 0.0)
        g1, g2 = self.barrier.geom_id_pairs[k]
        numeric = finite_difference_jacobian(
            lambda c: mujoco.mj_geomDistance(self.model, c.data, g1, g2, 10.0, None),
            self.configuration,
        )
        np.testing.assert_allclose(rows.jacobian[k], numeric, atol=1e-5)

    def test_moving_the_mocap_obstacle_updates_rows(self):
        rows = self.barrier.compute_rows(self.configuration)
        self.configuration.data.mocap_pos[0] = [5.0, 5.0, 5.0]
        self.configuration.update()
        moved = self.barrier.compute_rows(self.configuration)
        self.assertIsNot(moved, rows)
        self.assertEqual(moved.size, 0)


if __name__ == "__main__":
    absltest.main()
