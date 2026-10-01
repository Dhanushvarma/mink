"""Camera and occluder geometry shared by the self-occlusion tasks."""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from ..configuration import Configuration, _resolve_frame_id
from ..exceptions import TaskDefinitionError

_EPS = 1e-12


def resolve_camera_id(model: mujoco.MjModel, camera: str | int) -> int:
    """Resolve a camera name or id to the id of a perspective camera."""
    if isinstance(camera, str):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if camera_id == -1:
            names = [model.camera(i).name for i in range(model.ncam)]
            raise TaskDefinitionError(
                f"camera '{camera}' does not exist in the model. Available camera "
                f"names: {names}"
            )
    else:
        camera_id = int(camera)
        if not 0 <= camera_id < model.ncam:
            raise TaskDefinitionError(
                f"camera id {camera_id} is out of range for this model; expected "
                f"0 <= id < {model.ncam}."
            )
    if model.cam_projection[camera_id] != mujoco.mjtProjection.mjPROJ_PERSPECTIVE:
        raise TaskDefinitionError(
            f"camera '{model.camera(camera_id).name}' is not a perspective camera."
        )
    return camera_id


def camera_pose(
    configuration: Configuration, camera_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the world position and rotation matrix of a camera."""
    data = configuration.data
    # Configuration.update stops at mj_comPos, which leaves cam_xpos stale.
    mujoco.mj_camlight(configuration.model, data)
    return data.cam_xpos[camera_id].copy(), data.cam_xmat[camera_id].reshape(3, 3)


class Occluders:
    """A set of geoms, each reduced to an axis segment and a radius.

    Spheres and capsules are exact. A cylinder is treated as the capsule with the
    same radius and half-length. Any other type is replaced by its bounding sphere.
    """

    def __init__(self, model: mujoco.MjModel, geoms: Sequence[str | int]):
        geom_ids = [_resolve_frame_id(model, geom, "geom") for geom in geoms]
        if not geom_ids:
            raise TaskDefinitionError("At least one occluder geom is required.")
        if len(geom_ids) != len(set(geom_ids)):
            raise TaskDefinitionError(f"Duplicate occluder geoms found: {geom_ids}.")

        half_length = np.zeros(len(geom_ids))
        radius = np.zeros(len(geom_ids))
        for i, geom_id in enumerate(geom_ids):
            geom_type = model.geom_type[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius[i] = model.geom_size[geom_id, 0]
            elif geom_type in (
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
            ):
                radius[i], half_length[i] = model.geom_size[geom_id, :2]
            elif model.geom_rbound[geom_id] > 0.0:
                radius[i] = model.geom_rbound[geom_id]
            else:
                # Planes and height fields have no bounding sphere.
                raise TaskDefinitionError(
                    f"geom {geom_id} has an unbounded type and cannot be an occluder."
                )

        self.geom_ids = np.array(geom_ids)
        self.body_ids = model.geom_bodyid[self.geom_ids]
        self.half_length = half_length
        self.radius = radius

    def __len__(self) -> int:
        return len(self.geom_ids)

    def segments(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Return the two end points of every occluder axis, each of shape (n, 3)."""
        center = data.geom_xpos[self.geom_ids]
        # Third column of each rotation matrix: the geom's local z axis.
        axis = data.geom_xmat[self.geom_ids][:, 2::3]
        offset = self.half_length[:, None] * axis
        return center - offset, center + offset


def closest_points_to_segment(
    start: np.ndarray, end: np.ndarray, other_start: np.ndarray, other_end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest points between n segments and one other segment.

    Args:
        start: First end points of the n segments, of shape (n, 3).
        end: Second end points of the n segments, of shape (n, 3).
        other_start: First end point of the other segment, of shape (3,).
        other_end: Second end point of the other segment, of shape (3,).

    Returns:
        Parameters ``s`` and ``u`` in [0, 1], each of shape (n,). The closest
        points are ``start + s * (end - start)`` on each of the n segments and
        ``other_start + u * (other_end - other_start)`` on the other one.
    """
    d1 = end - start
    d2 = other_end - other_start
    r = start - other_start
    a = np.einsum("ij,ij->i", d1, d1)
    e = float(d2 @ d2)
    b = d1 @ d2
    c = np.einsum("ij,ij->i", d1, r)
    f = r @ d2

    # A point has no direction, so its parameter is pinned to 0.
    is_point = a <= _EPS
    safe_a = np.where(is_point, 1.0, a)

    def clamp_s(numerator: np.ndarray) -> np.ndarray:
        return np.where(is_point, 0.0, np.clip(numerator / safe_a, 0.0, 1.0))

    if e <= _EPS:
        return clamp_s(-c), np.zeros_like(a)

    denom = a * e - b * b
    # For parallel segments any s is a closest point; take s = 0.
    parallel = is_point | (denom <= _EPS * a * e)
    s = np.where(
        parallel,
        0.0,
        np.clip((b * f - c * e) / np.where(parallel, 1.0, denom), 0.0, 1.0),
    )
    u = (b * s + f) / e
    # Where u leaves [0, 1], clamp it and recompute s for the clamped point.
    s = np.where(u < 0.0, clamp_s(-c), s)
    s = np.where(u > 1.0, clamp_s(b - c), s)
    return s, np.clip(u, 0.0, 1.0)
