"""Shared fixtures for the distance barrier tests."""

import mujoco
import numpy as np

from mink import Configuration

# A three-link arm of capsule and sphere geoms, and a mocap sphere obstacle.
ARM_XML = """
<mujoco>
  <worldbody>
    <body name="link1">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom name="cap1" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom name="cap2" type="capsule" fromto="0 0 0 0.25 0 0" size="0.025"/>
        <body name="link3" pos="0.25 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom name="ball3" type="sphere" size="0.04" pos="0.06 0 0"/>
          <site name="tip" pos="0.1 0 0"/>
        </body>
      </body>
    </body>
    <body name="obstacle" mocap="true" pos="0.3 0.25 0">
      <geom name="obstacle" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

ROBOT_GEOMS = ("cap1", "cap2", "ball3")


def arm_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(ARM_XML)


def configuration_at(model: mujoco.MjModel, q) -> Configuration:
    configuration = Configuration(model)
    configuration.update(np.asarray(q, dtype=float))
    return configuration


def finite_difference_jacobian(distance_fn, configuration: Configuration, eps=1e-6):
    """Central differences of a scalar function of the configuration, shape (nv,)."""
    q0 = configuration.q
    jac = np.zeros(configuration.model.nv)
    for i in range(configuration.model.nv):
        dq = np.zeros(configuration.model.nv)
        dq[i] = eps
        configuration.update(q0 + dq)
        plus = distance_fn(configuration)
        configuration.update(q0 - dq)
        minus = distance_fn(configuration)
        jac[i] = (plus - minus) / (2 * eps)
    configuration.update(q0)
    return jac
