"""Tests for field_of_view_clearance_task.py."""

from pathlib import Path

import mujoco
import numpy as np
from absl.testing import absltest, parameterized

import mink
from mink import Configuration
from mink.exceptions import InvalidFrame, TaskDefinitionError
from mink.tasks import FieldOfViewClearanceTask

from .occlusion_utils import (
    ARM_GEOMS,
    MOBILE_ARM_XML,
    finite_difference_jacobian,
    random_configuration,
)

_TIDYBOT_XML = (
    Path(__file__).resolve().parents[1] / "examples" / "stanford_tidybot" / "scene.xml"
)

# Cameras of each kind on one tilted body, and a sphere that can be put anywhere.
_PROBE_RADIUS = 0.03
_CAMERAS_XML = f"""
<mujoco>
  <worldbody>
    <body name="rig" pos="0.3 -0.2 0.5" euler="20 30 40">
      <geom name="rig" size=".01"/>
      <camera name="fovy" pos="0.1 0 0" euler="10 0 0" fovy="50"
        resolution="640 360"/>
      <camera name="intrinsics" pos="0 0.1 0" focalpixel="500 450"
        principalpixel="40 -25" sensorsize="0.004 0.003" resolution="800 600"/>
      <camera name="no_resolution" pos="0 0 0.1" fovy="40"/>
      <camera name="ortho" projection="orthographic"/>
    </body>
    <body name="probe" mocap="true">
      <geom name="probe" type="sphere" size="{_PROBE_RADIUS}"/>
    </body>
    <geom name="floor" type="plane" size="2 2 .1"/>
  </worldbody>
</mujoco>
"""

_START = np.array([0.0, 0.0, 0.0, 0.0, 0.3, -0.9, 1.3, 0.3])
_TIP_GOAL = np.array([0.8, 0.15, 0.1])


def _drawn_frustum_planes(
    model: mujoco.MjModel, data: mujoco.MjData, camera_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Camera position and inward normals of the side planes of the frustum that
    MuJoCo draws for a camera."""
    scene = mujoco.MjvScene(model, maxgeom=1000)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
    free_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, free_camera)
    mujoco.mjv_updateScene(
        model, data, option, None, free_camera, mujoco.mjtCatBit.mjCAT_ALL, scene
    )

    # The frustum edges are line geoms: a start point, a direction and a length.
    vertices = []
    for geom in scene.geoms[: scene.ngeom]:
        if (
            geom.objtype == mujoco.mjtObj.mjOBJ_CAMERA
            and geom.objid == camera_id
            and geom.type == mujoco.mjtGeom.mjGEOM_LINE
        ):
            start = np.array(geom.pos, dtype=float)
            direction = np.array(geom.mat, dtype=float).reshape(3, 3)[:, 2]
            vertices += [start, start + geom.size[2] * direction]
    assert vertices, "MuJoCo drew no frustum for this camera"

    position = data.cam_xpos[camera_id].copy()
    forward = -data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
    rays = np.unique(np.round(np.array(vertices) - position, 6), axis=0)
    # Keep the four corners of the far face and order them around the view axis.
    depth = rays @ forward
    corners = rays[depth > 0.5 * depth.max()]
    assert len(corners) == 4
    right = data.cam_xmat[camera_id].reshape(3, 3)[:, 0]
    up = data.cam_xmat[camera_id].reshape(3, 3)[:, 1]
    corners = corners[np.argsort(np.arctan2(corners @ up, corners @ right))]

    center = corners.mean(axis=0)
    normals = []
    for i in range(4):
        normal = np.cross(corners[i], corners[(i + 1) % 4])
        normal /= np.linalg.norm(normal)
        normals.append(normal if normal @ center > 0.0 else -normal)
    return position, np.array(normals + [forward])


class TestFieldOfViewClearanceTask(parameterized.TestCase):
    """Test consistency of the field-of-view clearance task."""

    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string(MOBILE_ARM_XML)
        cls.cameras = mujoco.MjModel.from_xml_string(_CAMERAS_XML)
        cls.tidybot = mujoco.MjModel.from_xml_path(_TIDYBOT_XML.as_posix())

    def _task(self, **kwargs) -> FieldOfViewClearanceTask:
        args = dict(camera="fixed", occluders=ARM_GEOMS, cost=1.0)
        args.update(kwargs)
        return FieldOfViewClearanceTask(self.model, **args)

    def test_camera_unknown(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(camera="not_a_camera")

    def test_camera_orthographic(self):
        with self.assertRaises(TaskDefinitionError):
            FieldOfViewClearanceTask(self.cameras, "ortho", ["probe"], cost=1.0)

    def test_camera_without_resolution(self):
        with self.assertRaises(TaskDefinitionError):
            FieldOfViewClearanceTask(self.cameras, "no_resolution", ["probe"], cost=1.0)

    def test_occluder_unknown(self):
        with self.assertRaises(InvalidFrame):
            self._task(occluders=["not_a_geom"])

    def test_occluders_empty(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(occluders=[])

    def test_occluder_unbounded(self):
        with self.assertRaises(TaskDefinitionError):
            FieldOfViewClearanceTask(self.cameras, "fovy", ["floor"], cost=1.0)

    def test_cost_negative(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(cost=-1.0)

    def test_cost_invalid_shape(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(cost=[1.0, 2.0])

    def test_margin_negative(self):
        with self.assertRaises(TaskDefinitionError):
            self._task(margin=-0.1)

    @parameterized.parameters("fovy", "intrinsics")
    def test_clearance_matches_the_frustum_mujoco_draws(self, camera: str):
        task = FieldOfViewClearanceTask(self.cameras, camera, ["probe"], cost=1.0)
        configuration = Configuration(self.cameras)
        data = mujoco.MjData(self.cameras)
        mujoco.mj_forward(self.cameras, data)
        position, normals = _drawn_frustum_planes(
            self.cameras, data, self.cameras.camera(camera).id
        )

        rng = np.random.default_rng(0)
        signs = set()
        for _ in range(200):
            point = position + rng.uniform(-1.0, 1.0, size=3)
            configuration.data.mocap_pos[0] = point
            configuration.update()
            expected = np.max(-normals @ (point - position)) - _PROBE_RADIUS
            clearance = task.compute_clearance(configuration)
            np.testing.assert_allclose(clearance, [expected], atol=1e-5)
            signs.add(bool(clearance[0] < 0.0))
        # The samples cover both sides of the view boundary.
        self.assertEqual(signs, {True, False})

    def test_capsule_is_out_of_view_when_both_ends_are(self):
        task = self._task(occluders=["forearm"])
        configuration = Configuration(self.model, _START)
        clearance = task.compute_clearance(configuration)[0]
        self.assertGreater(clearance, 0.0)

        # Lower the arm until the forearm points into the view.
        q = _START.copy()
        q[5] = 0.2
        configuration.update(q)
        self.assertLess(task.compute_clearance(configuration)[0], 0.0)

    @parameterized.parameters(
        *[(camera, seed) for camera in ("fixed", "head") for seed in range(3)]
    )
    def test_jacobian_matches_finite_differences(self, camera: str, seed: int):
        # A margin this large keeps every row active.
        task = self._task(camera=camera, margin=10.0)
        configuration = random_configuration(self.model, seed)
        jacobian = task.compute_jacobian(configuration)
        np.testing.assert_allclose(
            jacobian,
            finite_difference_jacobian(configuration, task.compute_clearance),
            atol=1e-6,
        )
        # A camera fixed to the base sees the arm move only through arm joints.
        base = [self.model.joint(name).dofadr[0] for name in ("x", "y", "yaw")]
        np.testing.assert_allclose(jacobian[:, base], 0.0, atol=1e-12)
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
        task = FieldOfViewClearanceTask(
            self.tidybot, "base", arm_geoms, cost=1.0, margin=10.0
        )
        configuration = Configuration(self.tidybot)
        configuration.update_from_keyframe("home")
        np.testing.assert_allclose(
            task.compute_jacobian(configuration),
            finite_difference_jacobian(configuration, task.compute_clearance),
            atol=1e-6,
        )

    def test_error_is_the_shortfall_in_clearance(self):
        task = self._task()
        configuration = random_configuration(self.model, seed=1)
        clearance = task.compute_clearance(configuration)
        # The configuration has occluders both in and out of view.
        self.assertTrue(np.any(clearance < 0.0) and np.any(clearance > 0.0))

        error = task.compute_error(configuration)
        np.testing.assert_allclose(error, np.minimum(clearance, 0.0))
        jacobian = task.compute_jacobian(configuration)
        np.testing.assert_array_equal(jacobian[error == 0.0], 0.0)

    def test_inactive_when_nothing_is_in_view(self):
        task = self._task()
        configuration = Configuration(self.model, _START)
        self.assertGreater(task.compute_clearance(configuration).min(), 0.0)
        np.testing.assert_array_equal(
            task.compute_error(configuration), np.zeros(len(ARM_GEOMS))
        )
        np.testing.assert_array_equal(
            task.compute_jacobian(configuration),
            np.zeros((len(ARM_GEOMS), self.model.nv)),
        )

    def test_residual_matches_objective(self):
        task = self._task(cost=2.0, margin=10.0, lm_damping=0.1)
        configuration = random_configuration(self.model, seed=1)
        objective = task.compute_qp_objective(configuration)
        W, e, mu = task.compute_qp_residual(configuration)
        H = W.T @ W + mu * np.eye(self.model.nv)
        np.testing.assert_allclose(objective.H, H, atol=1e-12)
        np.testing.assert_allclose(objective.c, -e @ W, atol=1e-12)

    def _reach(self, with_task: bool) -> tuple[int, float]:
        """Move the tip to a goal inside the view; return how many occluders end
        up in view and the final tip error."""
        configuration = Configuration(self.model, _START)
        start = configuration.data.site_xpos[self.model.site("tip").id].copy()
        tip_task = mink.FrameTask(
            "tip", "site", position_cost=1.0, orientation_cost=0.0
        )
        task = self._task()
        tasks = [tip_task, mink.DampingTask(self.model, cost=0.05)]
        if with_task:
            tasks.append(task)

        dt = 0.02
        for step in range(400):
            alpha = min((step + 1) / 150, 1.0)
            tip_task.set_target(
                mink.SE3.from_translation(start + alpha * (_TIP_GOAL - start))
            )
            vel = mink.solve_ik(configuration, tasks, dt, "daqp", damping=1e-6)
            configuration.integrate_inplace(vel, dt)
        in_view = int(np.sum(task.compute_clearance(configuration) < -1e-3))
        tip_error = np.linalg.norm(tip_task.compute_error(configuration)[:3])
        return in_view, float(tip_error)

    def test_solve_moves_the_arm_out_of_view(self):
        in_view, tip_error = self._reach(with_task=False)
        self.assertGreater(in_view, 0)
        self.assertLess(tip_error, 1e-3)

        in_view, tip_error = self._reach(with_task=True)
        self.assertEqual(in_view, 0)
        self.assertLess(tip_error, 1e-3)


if __name__ == "__main__":
    absltest.main()
