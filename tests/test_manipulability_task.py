"""Tests for manipulability_task.py."""

from pathlib import Path

import mujoco
import numpy as np
from absl.testing import absltest, parameterized

import mink
from mink import Configuration
from mink.exceptions import InvalidFrame, TaskDefinitionError
from mink.tasks import ManipulabilityTask

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
_UR5E_XML = _EXAMPLES_DIR / "universal_robots_ur5e" / "ur5e.xml"
_TIDYBOT_XML = _EXAMPLES_DIR / "stanford_tidybot" / "scene.xml"

# One joint of every MuJoCo type on the way to the site, plus a side branch.
_ALL_JOINT_TYPES_XML = """
<mujoco>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <freejoint/>
      <geom size=".1"/>
      <body name="b1" pos=".2 .1 0">
        <joint type="ball"/>
        <geom size=".05"/>
        <body name="b2" pos=".3 0 .1">
          <joint type="hinge" axis="0 1 0"/>
          <joint type="slide" axis="1 0 0"/>
          <geom size=".05"/>
          <body name="b3" pos=".1 .2 0">
            <joint type="ball" pos=".05 0 0"/>
            <geom size=".05"/>
            <body name="b4" pos=".2 0 0">
              <joint type="hinge" axis="1 1 0"/>
              <geom size=".05"/>
              <site name="ee" pos=".1 .05 .02"/>
            </body>
          </body>
          <body name="side" pos="0 .3 0">
            <joint type="hinge" axis="0 0 1"/>
            <geom size=".05"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_METRICS_AND_AXES = [
    (metric, axes)
    for metric in ("yoshikawa", "icn")
    for axes in ("all", "translation", "rotation")
]


def _finite_difference_gradient(
    configuration: Configuration, task: ManipulabilityTask, eps: float = 1e-6
) -> np.ndarray:
    """Central differences of the measure along each tangent direction."""
    model = configuration.model
    q = configuration.q
    gradient = np.zeros(model.nv)
    for j in range(model.nv):
        dq = np.zeros(model.nv)
        dq[j] = eps
        values = []
        for sign in (1.0, -1.0):
            q_perturbed = q.copy()
            mujoco.mj_integratePos(model, q_perturbed, sign * dq, 1.0)
            configuration.update(q_perturbed)
            values.append(task.compute_manipulability(configuration))
        gradient[j] = (values[0] - values[1]) / (2.0 * eps)
    configuration.update(q)
    return gradient


def _random_configuration(model: mujoco.MjModel, seed: int) -> Configuration:
    rng = np.random.default_rng(seed)
    q = model.qpos0.copy()
    mujoco.mj_integratePos(model, q, rng.normal(size=model.nv), 1.0)
    return Configuration(model, q)


class TestManipulabilityTask(parameterized.TestCase):
    """Test consistency of the manipulability task."""

    @classmethod
    def setUpClass(cls):
        cls.ur5e = mujoco.MjModel.from_xml_path(_UR5E_XML.as_posix())
        cls.tidybot = mujoco.MjModel.from_xml_path(_TIDYBOT_XML.as_posix())
        cls.all_joints = mujoco.MjModel.from_xml_string(_ALL_JOINT_TYPES_XML)

    def setUp(self):
        self.configuration = Configuration(self.ur5e)
        self.configuration.update_from_keyframe("home")

    def test_cost_negative(self):
        with self.assertRaises(TaskDefinitionError):
            ManipulabilityTask(self.ur5e, "attachment_site", "site", cost=-1.0)

    def test_set_cost_negative(self):
        task = ManipulabilityTask(self.ur5e, "attachment_site", "site", cost=1.0)
        with self.assertRaises(TaskDefinitionError):
            task.set_cost(-1.0)

    def test_metric_invalid(self):
        with self.assertRaises(TaskDefinitionError):
            ManipulabilityTask(
                self.ur5e, "attachment_site", "site", cost=1.0, metric="volume"
            )

    def test_axes_invalid(self):
        with self.assertRaises(TaskDefinitionError):
            ManipulabilityTask(
                self.ur5e, "attachment_site", "site", cost=1.0, axes="planar"
            )

    def test_frame_invalid(self):
        with self.assertRaises(InvalidFrame):
            ManipulabilityTask(self.ur5e, "not_a_site", "site", cost=1.0)

    def test_dof_index_out_of_range(self):
        with self.assertRaises(TaskDefinitionError):
            ManipulabilityTask(
                self.ur5e, "attachment_site", "site", cost=1.0, dof_indices=range(1, 7)
            )

    def test_dof_index_duplicate(self):
        with self.assertRaises(TaskDefinitionError):
            ManipulabilityTask(
                self.ur5e,
                "attachment_site",
                "site",
                cost=1.0,
                dof_indices=[0, 1, 2, 3, 4, 5, 5],
            )

    def test_dof_index_does_not_move_frame(self):
        # DOF 10 is a gripper finger joint, downstream of the pinch site.
        with self.assertRaises(TaskDefinitionError):
            ManipulabilityTask(
                self.tidybot, "pinch_site", "site", cost=1.0, dof_indices=range(3, 11)
            )

    def test_too_few_dofs(self):
        with self.assertRaises(TaskDefinitionError):
            ManipulabilityTask(
                self.ur5e, "attachment_site", "site", cost=1.0, dof_indices=range(5)
            )
        # Three DOFs are enough for three rows.
        ManipulabilityTask(
            self.ur5e,
            "attachment_site",
            "site",
            cost=1.0,
            dof_indices=range(3),
            axes="translation",
        )

    def test_default_dof_indices_are_the_dofs_that_move_the_frame(self):
        task = ManipulabilityTask(self.tidybot, "pinch_site", "site", cost=1.0)
        self.assertEqual(task.dof_indices, list(range(10)))

    @parameterized.parameters(*_METRICS_AND_AXES)
    def test_manipulability_value(self, metric: str, axes: str):
        task = ManipulabilityTask(
            self.ur5e, "attachment_site", "site", cost=1.0, metric=metric, axes=axes
        )
        jac = self.configuration.get_frame_jacobian("attachment_site", "site")
        rows = {"all": slice(0, 6), "translation": slice(0, 3), "rotation": slice(3, 6)}
        jac = jac[rows[axes]]
        if metric == "yoshikawa":
            expected = np.sqrt(np.linalg.det(jac @ jac.T))
        else:
            expected = 1.0 / np.linalg.cond(jac)
        self.assertAlmostEqual(
            task.compute_manipulability(self.configuration), expected, places=10
        )

    @parameterized.parameters(*_METRICS_AND_AXES)
    def test_gradient_matches_finite_differences(self, metric: str, axes: str):
        task = ManipulabilityTask(
            self.ur5e, "attachment_site", "site", cost=1.0, metric=metric, axes=axes
        )
        configuration = _random_configuration(self.ur5e, seed=0)
        np.testing.assert_allclose(
            task.compute_gradient(configuration),
            _finite_difference_gradient(configuration, task),
            atol=1e-6,
        )

    @parameterized.parameters("body", "geom")
    def test_gradient_matches_finite_differences_other_frame_types(
        self, frame_type: str
    ):
        task = ManipulabilityTask(self.ur5e, "wrist_3_link", frame_type, cost=1.0)
        configuration = _random_configuration(self.ur5e, seed=1)
        np.testing.assert_allclose(
            task.compute_gradient(configuration),
            _finite_difference_gradient(configuration, task),
            atol=1e-6,
        )

    @parameterized.parameters(
        *[
            (metric, dof_indices)
            for metric in ("yoshikawa", "icn")
            # Every DOF on the chain, then a subset that leaves the free joint
            # out and splits the first ball joint.
            for dof_indices in (None, (7, 8, 9, 10, 11, 12, 13, 14))
        ]
    )
    def test_gradient_matches_finite_differences_all_joint_types(
        self, metric: str, dof_indices
    ):
        task = ManipulabilityTask(
            self.all_joints,
            "ee",
            "site",
            cost=1.0,
            dof_indices=dof_indices,
            metric=metric,
        )
        configuration = _random_configuration(self.all_joints, seed=2)
        gradient = task.compute_gradient(configuration)
        np.testing.assert_allclose(
            gradient, _finite_difference_gradient(configuration, task), atol=1e-6
        )
        # The side branch does not move the site.
        self.assertEqual(gradient[15], 0.0)

    def test_gradient_is_finite_at_a_singularity(self):
        # A straight elbow is singular.
        q = self.configuration.q
        q[2] = 0.0
        self.configuration.update(q)
        task = ManipulabilityTask(self.ur5e, "attachment_site", "site", cost=1.0)
        self.assertAlmostEqual(task.compute_manipulability(self.configuration), 0.0)
        gradient = task.compute_gradient(self.configuration)
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertGreater(np.linalg.norm(gradient), 0.0)

    @parameterized.parameters(
        *[
            (metric, dof_indices)
            for metric in ("yoshikawa", "icn")
            for dof_indices in (None, tuple(range(3, 10)))
        ]
    )
    def test_gradient_has_no_planar_base_component(self, metric: str, dof_indices):
        """Moving a planar base rotates the Jacobian, which keeps singular values."""
        task = ManipulabilityTask(
            self.tidybot,
            "pinch_site",
            "site",
            cost=1.0,
            dof_indices=dof_indices,
            metric=metric,
        )
        configuration = Configuration(self.tidybot)
        configuration.update_from_keyframe("home")
        q = configuration.q
        q[:3] = [0.4, -0.7, 0.9]
        configuration.update(q)
        gradient = task.compute_gradient(configuration)
        np.testing.assert_allclose(gradient[:3], np.zeros(3), atol=1e-12)
        self.assertGreater(np.linalg.norm(gradient[3:10]), 1e-4)
        np.testing.assert_allclose(
            gradient, _finite_difference_gradient(configuration, task), atol=1e-6
        )

    def test_qp_objective(self):
        task = ManipulabilityTask(self.ur5e, "attachment_site", "site", cost=2.5)
        H, c = task.compute_qp_objective(self.configuration)
        np.testing.assert_array_equal(H, np.zeros((self.ur5e.nv, self.ur5e.nv)))
        np.testing.assert_allclose(
            c, -2.5 * task.compute_gradient(self.configuration), atol=1e-12
        )
        self.assertIsNone(task.compute_qp_residual(self.configuration))

    def test_qp_objective_follows_set_cost(self):
        task = ManipulabilityTask(self.ur5e, "attachment_site", "site", cost=0.0)
        _, c = task.compute_qp_objective(self.configuration)
        np.testing.assert_array_equal(c, np.zeros(self.ur5e.nv))

        task.set_cost(3.0)
        _, c = task.compute_qp_objective(self.configuration)
        np.testing.assert_allclose(
            c, -3.0 * task.compute_gradient(self.configuration), atol=1e-12
        )

    def test_solve_raises_arm_manipulability_while_holding_the_frame(self):
        configuration = Configuration(self.tidybot)
        configuration.update_from_keyframe("home")

        frame_task = mink.FrameTask(
            "pinch_site", "site", position_cost=1.0, orientation_cost=1.0
        )
        frame_task.set_target_from_configuration(configuration)
        damping_task = mink.DampingTask(self.tidybot, cost=0.1)
        task = ManipulabilityTask(
            self.tidybot, "pinch_site", "site", cost=1e-3, dof_indices=range(3, 10)
        )
        initial = task.compute_manipulability(configuration)

        dt = 1e-2
        for _ in range(300):
            vel = mink.solve_ik(
                configuration,
                [frame_task, damping_task, task],
                dt,
                "daqp",
                damping=1e-6,
            )
            configuration.integrate_inplace(vel, dt)

        self.assertGreater(task.compute_manipulability(configuration), 1.5 * initial)
        error = frame_task.compute_error(configuration)
        self.assertLess(np.linalg.norm(error[:3]), 1e-3)
        self.assertLess(np.linalg.norm(error[3:]), 1e-3)
        # The arm cannot reconfigure at a fixed pinch pose unless the base moves.
        self.assertGreater(np.linalg.norm(configuration.q[:2]), 0.05)


if __name__ == "__main__":
    absltest.main()
