"""TidyBot keeping its arm out of the way of its base camera.

Drag the target: the pinch site follows it. Press 1 to switch the line-of-sight
task on and off, 2 for the field-of-view task, SPACE to pause. The line from the
base camera to the target is green while the arm leaves it clear and red while a
link blocks it.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.contrib.keyboard_teleop import keycodes

_HERE = Path(__file__).parent
_XML = _HERE / "stanford_tidybot" / "scene.xml"

_CAMERA = "base"
# The links near the hand are left out: they have to come close to the target.
_OCCLUDER_BODIES = [
    "half_arm_1_link",
    "half_arm_2_link",
    "forearm_link",
    "spherical_wrist_1_link",
]


@dataclass
class KeyCallback:
    line_of_sight: bool = True
    field_of_view: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == keycodes.KEY_1:
            self.line_of_sight = not self.line_of_sight
        elif key == keycodes.KEY_2:
            self.field_of_view = not self.field_of_view
        elif key == keycodes.KEY_SPACE:
            self.pause = not self.pause


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Joints we wish to control.
    # fmt: off
    joint_names = [
        # Base joints.
        "joint_x", "joint_y", "joint_th",
        # Arm joints.
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
    ]
    # fmt: on
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    posture_cost = np.zeros((model.nv,))
    posture_cost[3:] = 1e-3
    posture_task = mink.PostureTask(model, cost=posture_cost)

    # The arm links are meshes, so each one is treated as its bounding sphere.
    occluders = [
        geom_id
        for body_name in _OCCLUDER_BODIES
        for geom_id in mink.get_body_geom_ids(model, model.body(body_name).id)
        if model.geom_contype[geom_id]
    ]
    # A low gain spreads the correction over many steps at this control rate.
    line_of_sight_task = mink.LineOfSightTask(
        model, _CAMERA, occluders, cost=0.3, distance=0.1, gain=0.05
    )
    field_of_view_task = mink.FieldOfViewClearanceTask(
        model, _CAMERA, occluders, cost=0.1, gain=0.05
    )

    limits = [
        mink.ConfigurationLimit(model),
    ]

    # IK settings.
    solver = "daqp"

    key_callback = KeyCallback()
    camera_id = model.camera(_CAMERA).id

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        step = 0
        while viewer.is_running():
            # Update task targets.
            T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            end_effector_task.set_target(T_wt)
            line_of_sight_task.set_target(T_wt.translation())

            tasks = [end_effector_task, posture_task]
            if key_callback.line_of_sight:
                tasks.append(line_of_sight_task)
            if key_callback.field_of_view:
                tasks.append(field_of_view_task)

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)

            if not key_callback.pause:
                data.ctrl[actuator_ids] = configuration.q[dof_ids]
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Draw the line of sight.
            clearance = line_of_sight_task.compute_clearance(configuration).min()
            rgba = np.array([0.1, 0.8, 0.1, 1.0] if clearance > 0 else [1, 0, 0, 1])
            line = viewer.user_scn.geoms[0]
            mujoco.mjv_initGeom(
                line,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).flatten(),
                rgba.astype(np.float32),
            )
            mujoco.mjv_connector(
                line,
                mujoco.mjtGeom.mjGEOM_LINE,
                3.0,
                data.cam_xpos[camera_id],
                T_wt.translation(),
            )
            viewer.user_scn.ngeom = 1

            step += 1
            if step % 200 == 0:
                in_view = np.sum(
                    field_of_view_task.compute_clearance(configuration) < 0.0
                )
                print(
                    f"line of sight {'on' if key_callback.line_of_sight else 'off'}, "
                    f"field of view {'on' if key_callback.field_of_view else 'off'}: "
                    f"clearance {clearance:+.3f} m, {in_view} occluders in view"
                )

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
