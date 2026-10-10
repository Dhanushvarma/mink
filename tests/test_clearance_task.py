"""Tests for tasks/clearance_task.py."""

import numpy as np
from absl.testing import absltest

from mink import ClearanceTask, PointObstacleBarrier, solve_ik
from mink.exceptions import InvalidDamping, InvalidGain, TaskDefinitionError

from .barrier_utils import arm_model, configuration_at


class TestClearanceTask(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = arm_model()

    def setUp(self):
        self.configuration = configuration_at(self.model, [0.0, 0.0, 0.0])
        self.barrier = PointObstacleBarrier(
            self.model, ["cap1"], activation_distance=0.3, min_spacing=0.0
        )

    def _residual_at(self, gap: float, task: ClearanceTask):
        # cap1 lies on the x axis with radius 0.03: put a point `gap` above it.
        self.barrier.set_obstacles([[0.15, 0.03 + gap, 0.0]])
        return task.compute_qp_residual(self.configuration)

    def test_inactive_beyond_the_preferred_distance(self):
        task = ClearanceTask(self.barrier, preferred_distance=0.1, cost=1.0)
        jac, err, mu = self._residual_at(0.12, task)
        self.assertEqual(jac.shape, (0, self.model.nv))
        self.assertEqual(err.shape, (0,))
        self.assertEqual(mu, 0.0)

    def test_weight_ramps_smoothly(self):
        task = ClearanceTask(
            self.barrier, preferred_distance=0.1, cost=2.0, gain=0.5, ramp_width=0.04
        )
        rows_jac = None
        for gap, weight in ((0.1 - 1e-9, 0.0), (0.08, 1.0), (0.06, 2.0), (0.02, 2.0)):
            jac, err, _ = self._residual_at(gap, task)
            rows_jac = self.barrier.compute_rows(self.configuration).jacobian[0]
            np.testing.assert_allclose(jac[0], weight * rows_jac, atol=1e-6)
            np.testing.assert_allclose(err[0], weight * 0.5 * (0.1 - gap), atol=1e-6)

    def test_objective_matches_residual(self):
        task = ClearanceTask(self.barrier, 0.1, cost=1.5, gain=0.3, lm_damping=0.1)
        jac, err, mu = self._residual_at(0.03, task)
        H, c = task.compute_qp_objective(self.configuration)
        np.testing.assert_allclose(H, jac.T @ jac + mu * np.eye(self.model.nv))
        np.testing.assert_allclose(c, -err @ jac)
        self.assertGreater(mu, 0.0)

    def test_moves_the_robot_away(self):
        task = ClearanceTask(self.barrier, preferred_distance=0.1, cost=1.0, gain=0.5)
        self.barrier.set_obstacles([[0.25, 0.06, 0.0]])
        before = self.barrier.compute_rows(self.configuration).distance.min()
        v = solve_ik(self.configuration, [task], 0.01, "daqp", 1e-6, limits=[])
        self.configuration.integrate_inplace(v, 0.01)
        after = self.barrier.compute_rows(self.configuration).distance.min()
        self.assertGreater(after, before + 1e-3)

    def test_invalid_parameters(self):
        with self.assertRaises(InvalidGain):
            ClearanceTask(self.barrier, 0.1, gain=1.5)
        with self.assertRaises(InvalidDamping):
            ClearanceTask(self.barrier, 0.1, lm_damping=-1.0)
        with self.assertRaises(TaskDefinitionError):
            ClearanceTask(self.barrier, 0.1, cost=-1.0)
        with self.assertRaises(TaskDefinitionError):
            ClearanceTask(self.barrier, 0.1, ramp_width=0.0)
        with self.assertRaises(TaskDefinitionError):
            ClearanceTask(self.barrier, preferred_distance=0.5)


if __name__ == "__main__":
    absltest.main()
