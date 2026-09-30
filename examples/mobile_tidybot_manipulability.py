"""TidyBot holding an end-effector pose while maximizing arm manipulability.

Press ENTER to switch the manipulability task on and off, SPACE to pause. With the
task on and the target left alone, the base drives so the arm can move to a better
conditioned posture.
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


@dataclass
class KeyCallback:
    maximize_manipulability: bool = True
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == keycodes.KEY_ENTER:
            self.maximize_manipulability = not self.maximize_manipulability
        elif key == keycodes.KEY_SPACE:
            self.pause = not self.pause


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Joints we wish to control.
    # fmt: off
    base_joint_names = ["joint_x", "joint_y", "joint_th"]
    arm_joint_names = [
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
    ]
    # fmt: on
    joint_names = base_joint_names + arm_joint_names
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    base_dof_ids = [model.joint(name).dofadr[0] for name in base_joint_names]
    arm_dof_ids = [model.joint(name).dofadr[0] for name in arm_joint_names]

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    posture_cost = np.zeros((model.nv,))
    posture_cost[arm_dof_ids] = 1e-3
    posture_task = mink.PostureTask(model, cost=posture_cost)

    # The manipulability task has no Hessian, so the base needs its own damping.
    base_damping_cost = np.zeros((model.nv,))
    base_damping_cost[base_dof_ids] = 0.1
    damping_task = mink.DampingTask(model, base_damping_cost)

    # Arm columns only: with the base included, the measure stops seeing the arm
    # lose reach.
    manipulability_task = mink.ManipulabilityTask(
        model,
        frame_name="pinch_site",
        frame_type="site",
        cost=1e-4,
        dof_indices=arm_dof_ids,
    )

    tasks = [end_effector_task, posture_task, damping_task]

    limits = [
        mink.ConfigurationLimit(model),
    ]

    # IK settings.
    solver = "daqp"

    key_callback = KeyCallback()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        step = 0
        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            end_effector_task.set_target(T_wt)

            # Compute velocity and integrate into the next configuration.
            if key_callback.maximize_manipulability:
                vel = mink.solve_ik(
                    configuration,
                    [*tasks, manipulability_task],
                    rate.dt,
                    solver,
                    damping=1e-3,
                    limits=limits,
                )
            else:
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
                )
            configuration.integrate_inplace(vel, rate.dt)

            if not key_callback.pause:
                data.ctrl[actuator_ids] = configuration.q[dof_ids]
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            step += 1
            if step % 200 == 0:
                manipulability = manipulability_task.compute_manipulability(
                    configuration
                )
                state = "on" if key_callback.maximize_manipulability else "off"
                print(f"task {state}, arm manipulability: {manipulability:.4f}")

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
