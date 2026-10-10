"""TidyBot keeping its arm clear of a ball it only knows from range readings.

Drag the target: the pinch site follows it. Drag the ball into the arm's way: every
arm link keeps at least 3 cm from it. The ball is seen as noisy points on its surface,
as a range sensor would report them, and fed to a PointObstacleBarrier on capsules
fitted to the arm's meshes. Press 1 to switch the BarrierLimit on and off, 2 for the
ClearanceTask, SPACE to pause. A line is drawn from each link to its nearest points,
green while clear, turning red at the minimum distance.
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

# The shoulder only spins about its own axis, so it cannot move away from anything.
_ARM_BODIES = [
    "half_arm_1_link",
    "half_arm_2_link",
    "forearm_link",
    "spherical_wrist_1_link",
    "spherical_wrist_2_link",
    "bracelet_link",
]
_BALL_RADIUS = 0.06
_MINIMUM_DISTANCE = 0.03
_PREFERRED_DISTANCE = 0.12


@dataclass
class KeyCallback:
    barrier: bool = True
    clearance: bool = True
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == keycodes.KEY_1:
            self.barrier = not self.barrier
        elif key == keycodes.KEY_2:
            self.clearance = not self.clearance
        elif key == keycodes.KEY_SPACE:
            self.pause = not self.pause


def construct_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(_XML.as_posix())
    meshes = spec.compile()
    # Capsules enclosing each arm link's collision mesh, kept out of collisions.
    for name in _ARM_BODIES:
        p0, p1, radius = mink.fit_body_capsule(meshes, name, geom_group=3)
        spec.body(name).add_geom(
            name=f"{name}_capsule",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=np.concatenate([p0, p1]),
            size=[radius, 0, 0],
            group=4,
            contype=0,
            conaffinity=0,
            rgba=[0.3, 0.6, 1.0, 0.3],
        )
    ball = spec.worldbody.add_body(name="ball", mocap=True, pos=[0.45, 0.35, 0.6])
    ball.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[_BALL_RADIUS, 0, 0],
        contype=0,
        conaffinity=0,
        rgba=[0.9, 0.5, 0.3, 0.6],
    )
    return spec.compile()


def sense_ball(center: np.ndarray, rng: np.random.Generator, n: int = 64) -> np.ndarray:
    """Points spread over the ball's surface, with 5 mm of range noise."""
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azimuth = np.pi * (1.0 + 5.0**0.5) * i
    directions = np.stack(
        [
            np.cos(azimuth) * np.sin(polar),
            np.sin(azimuth) * np.sin(polar),
            np.cos(polar),
        ],
        axis=1,
    )
    return center + _BALL_RADIUS * directions + rng.normal(0.0, 0.005, (n, 3))


if __name__ == "__main__":
    model = construct_model()
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

    barrier = mink.PointObstacleBarrier(
        model, [f"{name}_capsule" for name in _ARM_BODIES], activation_distance=0.25
    )
    barrier_limit = mink.BarrierLimit(barrier, _MINIMUM_DISTANCE, gain=20.0)
    # A low gain spreads the correction over many steps at this control rate.
    clearance_task = mink.ClearanceTask(
        barrier, _PREFERRED_DISTANCE, cost=0.3, gain=0.05
    )

    configuration_limit = mink.ConfigurationLimit(model)

    # IK settings.
    solver = "daqp"
    rng = np.random.default_rng(0)
    ball_mocap = model.body("ball").mocapid[0]

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
            # Update task targets and the sensed obstacle points.
            T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            end_effector_task.set_target(T_wt)
            barrier.set_obstacles(sense_ball(data.mocap_pos[ball_mocap], rng), 0.01)

            tasks = [end_effector_task, posture_task]
            if key_callback.clearance:
                tasks.append(clearance_task)
            limits = [configuration_limit]
            if key_callback.barrier:
                limits.append(barrier_limit)

            # Compute velocity and integrate into the next configuration. If the ball
            # is pushed in faster than the arm can back away, hold still instead.
            try:
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
                )
            except mink.NoSolutionFound:
                barrier_limit.relaxation = 1.0
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, damping=1e-3, limits=limits
                )
                barrier_limit.relaxation = 0.0
            configuration.integrate_inplace(vel, rate.dt)

            if not key_callback.pause:
                data.ctrl[actuator_ids] = configuration.q[dof_ids]
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Draw each barrier row from the arm to the nearest sensed point.
            rows = barrier.compute_rows(configuration)
            viewer.user_scn.ngeom = 0
            for distance, (robot, obstacle) in zip(rows.distance, rows.witness):
                if viewer.user_scn.ngeom == viewer.user_scn.maxgeom:
                    break
                margin = (distance - _MINIMUM_DISTANCE) / (
                    _PREFERRED_DISTANCE - _MINIMUM_DISTANCE
                )
                x = float(np.clip(margin, 0.0, 1.0))
                rgba = np.array([1.0 - x, 0.2 + 0.6 * x, 0.1, 1.0], dtype=np.float32)
                line = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    line,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).flatten(),
                    rgba,
                )
                mujoco.mjv_connector(
                    line, mujoco.mjtGeom.mjGEOM_LINE, 3.0, robot, obstacle
                )
                viewer.user_scn.ngeom += 1

            step += 1
            if step % 200 == 0 and rows.size:
                print(
                    f"barrier {'on' if key_callback.barrier else 'off'}, "
                    f"clearance task {'on' if key_callback.clearance else 'off'}: "
                    f"closest link {rows.distance.min():+.3f} m from the ball"
                )

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
