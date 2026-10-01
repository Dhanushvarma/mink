"""Shared model and helpers for the self-occlusion task tests."""

from typing import Callable

import mujoco
import numpy as np

from mink import Configuration

# A planar mobile base carrying a four-joint arm and two cameras: "fixed" on the
# base, looking forward and 30 degrees down, and "head" on a pan joint.
MOBILE_ARM_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="y" type="slide" axis="0 1 0"/>
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom name="base" type="box" size=".2 .2 .1"/>
      <camera name="fixed" pos="0.25 0 0.15" xyaxes="0 -1 0 0.5 0 0.866"
        fovy="60" resolution="640 480"/>
      <body name="head" pos="-0.1 0 0.7">
        <joint name="pan" type="hinge" axis="0 0 1"/>
        <geom name="head" type="sphere" size=".05"/>
        <camera name="head" pos="0.05 0 0" xyaxes="0 -1 0 0.5 0 0.866"
          fovy="45" resolution="640 480"/>
      </body>
      <body name="upper_arm" pos="0 0 0.2">
        <joint name="shoulder_pan" type="hinge" range="-2.5 2.5" axis="0 0 1"/>
        <joint name="shoulder_lift" type="hinge" range="-2 2" axis="0 1 0"/>
        <geom name="upper_arm" type="capsule" fromto="0 0 0 0.35 0 0" size=".04"/>
        <body name="forearm" pos="0.35 0 0">
          <joint name="elbow" type="hinge" range="-2.5 2.5" axis="0 1 0"/>
          <geom name="forearm" type="capsule" fromto="0 0 0 0.35 0 0" size=".03"/>
          <body name="hand" pos="0.35 0 0">
            <joint name="wrist" type="hinge" range="-2 2" axis="0 1 0"/>
            <geom name="wrist" type="cylinder" fromto="0 0 0 0.06 0 0" size=".03"/>
            <geom name="palm" type="box" pos="0.09 0 0" size=".03 .04 .02"/>
            <geom name="fingertip" type="sphere" pos="0.14 0 0" size=".02"/>
            <site name="tip" pos="0.16 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

ARM_GEOMS = ("upper_arm", "forearm", "wrist", "palm", "fingertip")


def random_configuration(model: mujoco.MjModel, seed: int) -> Configuration:
    rng = np.random.default_rng(seed)
    q = model.qpos0.copy()
    mujoco.mj_integratePos(model, q, rng.uniform(-0.8, 0.8, size=model.nv), 1.0)
    return Configuration(model, q)


def finite_difference_jacobian(
    configuration: Configuration,
    function: Callable[[Configuration], np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """Central differences of a vector function along each tangent direction."""
    model = configuration.model
    q = configuration.q
    columns = []
    for j in range(model.nv):
        dq = np.zeros(model.nv)
        dq[j] = eps
        values = []
        for sign in (1.0, -1.0):
            q_perturbed = q.copy()
            mujoco.mj_integratePos(model, q_perturbed, sign * dq, 1.0)
            configuration.update(q_perturbed)
            values.append(function(configuration).copy())
        columns.append((values[0] - values[1]) / (2.0 * eps))
    configuration.update(q)
    return np.array(columns).T
