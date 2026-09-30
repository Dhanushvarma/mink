"""Tests for line_of_sight_task.py."""

from pathlib import Path

import mujoco
import numpy as np
from absl.testing import absltest, parameterized

import mink
from mink import Configuration
from mink.exceptions import (
    InvalidFrame,
    InvalidTarget,
    TargetNotSet,
    TaskDefinitionError,
)
from mink.tasks import LineOfSightTask

from .occlusion_utils import (
    ARM_GEOMS,
    MOBILE_ARM_XML,
    finite_difference_jacobian,
    random_configuration,
)

_TIDYBOT_XML = (
    Path(__file__).resolve().parents[1] / "examples" / "stanford_tidybot" / "scene.xml"
)

# The camera rides on a slide joint, and a thin static capsule "los" lies along
# the line of sight it has at slide = 0.3. mj_geomDistance to that capsule is then
# an independent measure of each occluder's clearance.
_LOS_RADIUS = 1e-3
_SLIDE = 0.3
_TARGET = np.array([1.2, -0.2, 0.1])
_ORACLE_XML = f"""
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="rig" pos="0 0 1">
      <joint name="slide" type="slide" axis="0 1 0"/>
      <geom name="rig" size=".01"/>
      <camera name="cam"/>
      <camera name="ortho" projection="orthographic"/>
    </body>
    <geom name="los" type="capsule" size="{_LOS_RADIUS}"
      fromto="0 {_SLIDE} 1 {_TARGET[0]} {_TARGET[1]} {_TARGET[2]}"/>
    <geom name="sphere" type="sphere" pos=".5 .3 .6" size=".07"/>
    <geom name="crossing" type="capsule" fromto=".2 -.4 .3 .9 .4 .9" size=".05"/>
    <geom name="end_on" type="capsule" fromto=".6 .5 .9 .6 .2 .7" size=".04"/>
    <geom name="past_target" type="capsule" fromto="1.4 -.5 0 1.6 0 .2" size=".03"/>
    <geom name="behind_camera" type="capsule" fromto="-.3 .2 1.2 -.1 .6 1.1" size=".03"/>
    <geom name="parallel" type="capsule" fromto=".1 .5 1 .7 .25 .55" size=".02"/>
    <geom name="box" type="box" pos=".7 -.3 .8" size=".05 .08 .03"/>
  </worldbody>
</mujoco>
"""
_EXACT_GEOMS = (
    "sphere",
    "crossing",
    "end_on",
    "past_target",
    "behind_camera",
    "parallel",
)

# Reach used by the solve test: the arm has to pass through the line of sight to
# get the tip to its goal unless it goes around.
_START = np.array([0.0, 0.0, 0.0, 0.0, 0.3, -0.9, 1.3, 0.3])
_TIP_GOAL = np.array([0.75, -0.1, 0.05])
_REACH_TARGET = np.array([1.3, 0.2, 0.0])


class TestLineOfSightTask(parameterized.TestCase):
    """Test consistency of the line-of-sight task."""

    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string(MOBILE_ARM_XML)
        cls.oracle = mujoco.MjModel.from_xml_string(_ORACLE_XML)
        cls.tidybot = mujoco.MjModel.from_xml_path(_TIDYBOT_XML.as_posix())

    def _task(self, **kwargs) -> LineOfSightTask:
        args = dict(camera="fixed", occluders=ARM_GEOMS, cost=1.0, distance=0.1)
        args.update(kwargs)
        return LineOfSightTask(self.model, **args)

    def test_camera_unknown(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(camera="not_a_camera")
        with self.assertRaises(TaskDefinitionError):
            self._task(camera=self.model.ncam)

    def test_camera_orthographic(self):
        with self.assertRaises(TaskDefinitionError):
            LineOfSightTask(self.oracle, "ortho", ["sphere"], cost=1.0, distance=0.1)

    def test_occluder_unknown(self):
        with self.assertRaises(InvalidFrame):
            self._task(occluders=["not_a_geom"])

    def test_occluders_empty(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(occluders=[])

    def test_occluders_duplicate(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(occluders=["forearm", "forearm"])

    def test_occluder_unbounded(self):
        with self.assertRaises(TaskDefinitionError):
            LineOfSightTask(self.oracle, "cam", ["floor"], cost=1.0, distance=0.1)

    def test_cost_negative(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(cost=-1.0)

    def test_cost_invalid_shape(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(cost=[1.0, 2.0])

    def test_cost_per_occluder(self):
        cost = np.arange(1.0, len(ARM_GEOMS) + 1.0)
        task = self._task(cost=cost)
        np.testing.assert_array_equal(task.cost, cost)

    def test_distance_not_positive(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(distance=0.0)

    def test_target_not_set(self):
        task = self._task()
        configuration = Configuration(self.model)
        with self.assertRaises(TargetNotSet):
            task.compute_error(configuration)
        with self.assertRaises(TargetNotSet):
            task.compute_jacobian(configuration)
        with self.assertRaises(TargetNotSet):
            task.compute_clearance(configuration)

    def test_target_invalid_shape(self):
        with self.assertRaises(InvalidTarget):
            self._task().set_target([1.0, 2.0])

    def test_clearance_matches_geom_distance(self):
        """Also checks the camera pose is current after a plain update."""
        task = LineOfSightTask(self.oracle, "cam", _EXACT_GEOMS, cost=1.0, distance=0.1)
        task.set_target(_TARGET)
        configuration = Configuration(self.oracle, np.array([_SLIDE]))
        los_id = self.oracle.geom("los").id
        expected = [
            mujoco.mj_geomDistance(
                self.oracle,
                configuration.data,
                self.oracle.geom(name).id,
                los_id,
                10.0,
                None,
            )
            + _LOS_RADIUS
            for name in _EXACT_GEOMS
        ]
        np.testing.assert_allclose(
            task.compute_clearance(configuration), expected, atol=1e-5
        )

    def test_other_geom_types_use_their_bounding_sphere(self):
        task = LineOfSightTask(self.oracle, "cam", ["box"], cost=1.0, distance=0.1)
        task.set_target(_TARGET)
        configuration = Configuration(self.oracle, np.array([_SLIDE]))

        box_id = self.oracle.geom("box").id
        center = configuration.data.geom_xpos[box_id]
        camera = np.array([0.0, _SLIDE, 1.0])
        along = np.clip(
            (center - _TARGET) @ (camera - _TARGET) / np.sum((camera - _TARGET) ** 2),
            0.0,
            1.0,
        )
        to_line = np.linalg.norm(center - _TARGET - along * (camera - _TARGET))
        np.testing.assert_allclose(
            task.compute_clearance(configuration),
            [to_line - self.oracle.geom_rbound[box_id]],
            atol=1e-12,
        )

    @parameterized.parameters(
        *[(camera, seed) for camera in ("fixed", "head") for seed in range(3)]
    )
    def test_jacobian_matches_finite_differences(self, camera: str, seed: int):
        # A distance this large keeps every row active.
        task = self._task(camera=camera, distance=10.0)
        task.set_target([1.2, 0.2, 0.1])
        configuration = random_configuration(self.model, seed)
        jacobian = task.compute_jacobian(configuration)
        np.testing.assert_allclose(
            jacobian,
            finite_difference_jacobian(configuration, task.compute_clearance),
            atol=1e-6,
        )
        # Only the head camera moves with the pan joint.
        pan = self.model.joint("pan").dofadr[0]
        self.assertEqual(np.any(jacobian[:, pan] != 0.0), camera == "head")

    def test_jacobian_matches_finite_differences_on_meshes(self):
        arm_geoms = [
            geom_id
            for geom_id in mink.get_subtree_geom_ids(
                self.tidybot, self.tidybot.body("gen3/base_link").id
            )
            if self.tidybot.geom_contype[geom_id]
        ]
        task = LineOfSightTask(self.tidybot, "base", arm_geoms, cost=1.0, distance=10.0)
        task.set_target([0.9, 0.1, 0.0])
        configuration = Configuration(self.tidybot)
        configuration.update_from_keyframe("home")
        np.testing.assert_allclose(
            task.compute_jacobian(configuration),
            finite_difference_jacobian(configuration, task.compute_clearance),
            atol=1e-6,
        )

    def test_error_is_the_shortfall_in_clearance(self):
        task = self._task(distance=0.25)
        task.set_target([1.2, 0.2, 0.1])
        configuration = random_configuration(self.model, seed=0)
        clearance = task.compute_clearance(configuration)
        # The configuration has occluders on both sides of the distance.
        self.assertTrue(np.any(clearance < 0.25) and np.any(clearance > 0.25))

        error = task.compute_error(configuration)
        np.testing.assert_allclose(error, np.minimum(clearance - 0.25, 0.0))
        jacobian = task.compute_jacobian(configuration)
        np.testing.assert_array_equal(jacobian[error == 0.0], 0.0)
        self.assertTrue(np.all(np.any(jacobian[error < 0.0] != 0.0, axis=1)))

    def test_inactive_when_nothing_is_near(self):
        task = self._task(distance=0.05)
        # Far off to the side, away from the arm.
        task.set_target([0.5, -3.0, 0.2])
        configuration = Configuration(self.model, _START)
        np.testing.assert_array_equal(
            task.compute_error(configuration), np.zeros(len(ARM_GEOMS))
        )
        np.testing.assert_array_equal(
            task.compute_jacobian(configuration),
            np.zeros((len(ARM_GEOMS), self.model.nv)),
        )

    def test_residual_matches_objective(self):
        task = self._task(cost=2.0, distance=10.0, lm_damping=0.1)
        task.set_target([1.2, 0.2, 0.1])
        configuration = random_configuration(self.model, seed=1)
        objective = task.compute_qp_objective(configuration)
        W, e, mu = task.compute_qp_residual(configuration)
        H = W.T @ W + mu * np.eye(self.model.nv)
        np.testing.assert_allclose(objective.H, H, atol=1e-12)
        np.testing.assert_allclose(objective.c, -e @ W, atol=1e-12)

    def _reach(self, with_task: bool) -> tuple[float, float]:
        """Move the tip to its goal; return the lowest clearance seen and the
        final tip error."""
        configuration = Configuration(self.model, _START)
        start = configuration.data.site_xpos[self.model.site("tip").id].copy()
        tip_task = mink.FrameTask(
            "tip", "site", position_cost=1.0, orientation_cost=0.0
        )
        task = self._task()
        task.set_target(_REACH_TARGET)
        tasks = [tip_task, mink.DampingTask(self.model, cost=0.05)]
        if with_task:
            tasks.append(task)

        dt = 0.02
        lowest = np.inf
        for step in range(300):
            alpha = min((step + 1) / 150, 1.0)
            tip_task.set_target(
                mink.SE3.from_translation(start + alpha * (_TIP_GOAL - start))
            )
            vel = mink.solve_ik(configuration, tasks, dt, "daqp", damping=1e-6)
            configuration.integrate_inplace(vel, dt)
            lowest = min(lowest, task.compute_clearance(configuration).min())
        tip_error = np.linalg.norm(tip_task.compute_error(configuration)[:3])
        return lowest, float(tip_error)

    def test_solve_keeps_the_line_of_sight_clear(self):
        lowest, tip_error = self._reach(with_task=False)
        self.assertLess(lowest, 0.0)
        self.assertLess(tip_error, 1e-3)

        lowest, tip_error = self._reach(with_task=True)
        self.assertGreater(lowest, 0.05)
        self.assertLess(tip_error, 1e-3)


if __name__ == "__main__":
    absltest.main()
