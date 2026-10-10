"""Capsule geometry: robot geoms as capsules, point distances, and enclosing fits.

A capsule is the set of points within a radius of a segment. Spheres are capsules
with a zero-length segment; capsule and cylinder geoms use their axis and radius
(a cylinder is enclosed by the capsule); any other geom is replaced by its bounding
sphere. Meshes are better covered by fitting a capsule to their vertices with
:func:`fit_body_capsule` and adding it to the model as a proxy geom.
"""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from ..configuration import _resolve_frame_id


class GeomCapsules:
    """A fixed set of geoms, posed as capsules from any ``MjData`` of the model.

    Attributes:
        geom_ids: Ids of the geoms, shape (L,).
        body_ids: Id of each geom's body, shape (L,).
        radius: Capsule radius of each geom in [m], shape (L,).
    """

    def __init__(self, model: mujoco.MjModel, geoms: Sequence[str | int]):
        """Constructor.

        Args:
            model: MuJoCo model.
            geoms: Names or ids of the geoms.
        """
        self.geom_ids = np.array(
            [_resolve_frame_id(model, g, "geom") for g in geoms], dtype=int
        )
        self.body_ids = model.geom_bodyid[self.geom_ids].copy()
        types = model.geom_type[self.geom_ids]
        size = model.geom_size[self.geom_ids]
        is_sphere = types == mujoco.mjtGeom.mjGEOM_SPHERE
        is_axial = (types == mujoco.mjtGeom.mjGEOM_CAPSULE) | (
            types == mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        self.radius = np.where(
            is_sphere | is_axial, size[:, 0], model.geom_rbound[self.geom_ids]
        )
        self._half_length = np.where(is_axial, size[:, 1], 0.0)

    def __len__(self) -> int:
        return len(self.geom_ids)

    def segments(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Axis end points in the world frame, two arrays of shape (L, 3)."""
        pos = data.geom_xpos[self.geom_ids]
        axis = data.geom_xmat[self.geom_ids].reshape(-1, 3, 3)[:, :, 2]
        offset = axis * self._half_length[:, None]
        return pos - offset, pos + offset


def closest_points_on_segments(
    points: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Closest point of every segment to every point.

    Args:
        points: Query points, shape (M, 3).
        a: First end points of the segments, shape (L, 3).
        b: Second end points of the segments, shape (L, 3).

    Returns:
        Closest points, shape (M, L, 3).
    """
    ab = b - a
    length_sq = np.einsum("lj,lj->l", ab, ab)
    rel = points[:, None, :] - a[None, :, :]
    safe = np.where(length_sq > 0.0, length_sq, 1.0)
    t = np.where(length_sq > 0.0, np.einsum("mlj,lj->ml", rel, ab) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return a[None] + t[..., None] * ab[None]


def point_capsule_distances(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, radius: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance from every point to every capsule surface.

    Args:
        points: Query points, shape (M, 3).
        a: First axis end points, shape (L, 3).
        b: Second axis end points, shape (L, 3).
        radius: Capsule radii, shape (L,).

    Returns:
        Distances of shape (M, L), negative inside a capsule, and the closest axis
        points of shape (M, L, 3).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    closest = closest_points_on_segments(points, a, b)
    dist = np.linalg.norm(points[:, None, :] - closest, axis=2) - radius[None, :]
    return dist, closest


def _circle_through(points: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Smallest circle with 1, 2 or 3 points on its boundary (2D)."""
    if len(points) == 1:
        return points[0], 0.0
    if len(points) == 2:
        centre = 0.5 * (points[0] + points[1])
        return centre, float(np.linalg.norm(points[0] - centre))
    a, b, c = points
    den = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(den) < 1e-15:
        # Collinear: the two farthest points span the circle.
        pairs = [(a, b), (a, c), (b, c)]
        p, q = max(pairs, key=lambda pq: float(np.linalg.norm(pq[0] - pq[1])))
        return _circle_through([p, q])
    sa, sb, sc = a @ a, b @ b, c @ c
    ux = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / den
    uy = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / den
    centre = np.array([ux, uy])
    return centre, float(np.linalg.norm(a - centre))


def min_enclosing_circle(points: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    """Smallest circle enclosing 2D points (Welzl's algorithm, iterative form).

    Args:
        points: Points, shape (N, 2).
        seed: Seed of the random point order, which only affects run time.

    Returns:
        Centre, shape (2,), and radius.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.random.default_rng(seed).permutation(len(pts))]
    n, eps = len(pts), 1e-10

    def first_outside(centre, r, lo, hi) -> int:
        if hi <= lo:
            return -1
        out = np.nonzero(np.linalg.norm(pts[lo:hi] - centre, axis=1) > r + eps)[0]
        return lo + int(out[0]) if out.size else -1

    centre, r = pts[0], 0.0
    i = first_outside(centre, r, 1, n)
    while i >= 0:  # pts[i] is on the boundary of the circle enclosing pts[: i + 1].
        centre, r = pts[i], 0.0
        j = first_outside(centre, r, 0, i)
        while j >= 0:  # So is pts[j].
            centre, r = _circle_through([pts[i], pts[j]])
            k = first_outside(centre, r, 0, j)
            while k >= 0:
                centre, r = _circle_through([pts[i], pts[j], pts[k]])
                k = first_outside(centre, r, k + 1, j)
            j = first_outside(centre, r, j + 1, i)
        i = first_outside(centre, r, i + 1, n)
    return centre, r


def fit_capsule(
    vertices: np.ndarray, axis: np.ndarray | Sequence[float] | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Smallest-radius capsule along an axis that encloses all vertices.

    The radius is that of the minimal circle enclosing the vertices projected across
    the axis; the segment is then the shortest one whose end caps still cover every
    vertex. An object shorter than it is wide gets a sphere.

    Args:
        vertices: Points to enclose, shape (N, 3).
        axis: Capsule axis direction. Defaults to the first principal axis.

    Returns:
        Axis end points, two arrays of shape (3,), and the radius.
    """
    v = np.asarray(vertices, dtype=float)
    if axis is None:
        a = np.linalg.svd(v - v.mean(axis=0), full_matrices=False)[2][0]
    else:
        a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    t = v @ a
    across = np.stack([v @ u, v @ w], axis=1)
    centre, r = min_enclosing_circle(across)
    rho = np.linalg.norm(across - centre, axis=1)
    reach = np.sqrt(np.maximum(r * r - rho * rho, 0.0))
    t_lo, t_hi = float(np.min(t + reach)), float(np.max(t - reach))
    origin = centre[0] * u + centre[1] * w
    if t_lo <= t_hi:
        return origin + t_lo * a, origin + t_hi * a, float(r)
    mid = origin + 0.5 * (t_lo + t_hi) * a
    return mid, mid.copy(), float(np.max(np.linalg.norm(v - mid, axis=1)))


def fit_body_capsule(
    model: mujoco.MjModel,
    body: str | int,
    geom_group: int | None = None,
    axis: np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Capsule enclosing the mesh geoms of a body, in the body frame.

    Args:
        model: MuJoCo model.
        body: Name or id of the body.
        geom_group: Only use meshes in this geom group (e.g. the collision group).
            Defaults to all mesh geoms of the body.
        axis: Capsule axis in the body frame. Defaults to the first principal axis.

    Returns:
        Axis end points in the body frame, two arrays of shape (3,), and the radius.
    """
    body_id = _resolve_frame_id(model, body, "body")
    vertices = []
    for g in np.nonzero(model.geom_bodyid == body_id)[0]:
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if geom_group is not None and model.geom_group[g] != geom_group:
            continue
        mesh = model.geom_dataid[g]
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        local = model.mesh_vert[adr : adr + num] @ rot.reshape(3, 3).T
        vertices.append(local + model.geom_pos[g])
    if not vertices:
        raise ValueError(f"body {model.body(body_id).name!r} has no mesh geoms to fit")
    return fit_capsule(np.concatenate(vertices), axis)
