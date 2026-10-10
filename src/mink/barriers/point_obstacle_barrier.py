"""Distances from robot geoms to obstacle spheres supplied as data."""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from .barrier import BarrierRows, DistanceBarrier
from .capsules import GeomCapsules, point_capsule_distances


class PointObstacleBarrier(DistanceBarrier):
    r"""Distances from robot geoms to obstacles given as spheres, such as sensor returns.

    Obstacles are spheres with centres :math:`c_i` and radii :math:`\rho_i`, set with
    :meth:`set_obstacles`; a radius can carry the uncertainty of a measured point. The
    robot is a set of geoms seen as capsules (see :class:`~mink.GeomCapsules`). For a
    robot capsule with axis segment :math:`[a, b]` and radius :math:`r`, let
    :math:`p` be the point of the segment closest to :math:`c_i`. Then

    .. math::

        d = \|c_i - p\| - r - \rho_i,
        \qquad
        \frac{\partial d}{\partial q} = -n^T J_s,

    where :math:`n = (c_i - p) / \|c_i - p\|`, :math:`s = p + r n` is the witness
    point on the robot and :math:`J_s` its translational Jacobian. Obstacles are taken
    as static over a step; sliding of :math:`p` along the axis is orthogonal to
    :math:`n`, so the Jacobian is exact to first order.

    Each robot geom keeps at most ``max_rows_per_geom`` rows: the closest obstacles
    within ``activation_distance``, picked greedily at least ``min_spacing`` apart, so
    that one surface seen through many sensor zones gives a few distinct rows rather
    than many identical ones.

    Attributes:
        model: MuJoCo model.
        capsules: The robot geoms as capsules.
        max_rows_per_geom: Maximum number of rows per robot geom.
        min_spacing: Minimum distance in [m] between the obstacle centres of the rows
            of one geom.

    Example:

    .. code-block:: python

        barrier = PointObstacleBarrier(model, geoms=["upper_arm", "forearm"])
        barrier.set_obstacles(points, radii=0.02)
        limit = BarrierLimit(barrier, minimum_distance=0.03, gain=10.0)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geoms: Sequence[str | int],
        activation_distance: float = 0.25,
        max_rows_per_geom: int = 3,
        min_spacing: float = 0.03,
    ):
        """Constructor.

        Args:
            model: MuJoCo model.
            geoms: Names or ids of the robot geoms to keep away from obstacles.
            activation_distance: Distance in [m] below which an obstacle produces a
                row.
            max_rows_per_geom: Maximum number of rows per robot geom.
            min_spacing: Minimum distance in [m] between the obstacle centres of the
                rows of one geom.
        """
        if max_rows_per_geom < 1:
            raise ValueError("max_rows_per_geom must be at least 1")
        if min_spacing < 0.0:
            raise ValueError("min_spacing must be non-negative")
        super().__init__(activation_distance)
        self.model = model
        self.capsules = GeomCapsules(model, geoms)
        self.max_rows_per_geom = max_rows_per_geom
        self.min_spacing = min_spacing
        self._centers = np.zeros((0, 3))
        self._radii = np.zeros(0)
        self._jacp = np.zeros((3, model.nv))

    @property
    def centers(self) -> np.ndarray:
        """Obstacle centres in the world frame, shape (M, 3)."""
        return self._centers

    @property
    def radii(self) -> np.ndarray:
        """Obstacle radii in [m], shape (M,)."""
        return self._radii

    def set_obstacles(
        self, centers: npt.ArrayLike, radii: npt.ArrayLike | None = None
    ) -> None:
        """Replace the obstacles.

        Args:
            centers: Sphere centres in the world frame, shape (M, 3).
            radii: Sphere radii in [m]: a scalar or shape (M,). Defaults to points.
        """
        centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        if radii is None:
            radii = np.zeros(len(centers))
        radii = np.broadcast_to(np.asarray(radii, dtype=float), (len(centers),))
        if np.any(radii < 0.0):
            raise ValueError("obstacle radii must be non-negative")
        self._centers = centers.copy()
        self._radii = radii.copy()
        self._invalidate()

    def clear_obstacles(self) -> None:
        """Remove all obstacles."""
        self.set_obstacles(np.zeros((0, 3)))

    def _pick(self, candidates: np.ndarray, dist: np.ndarray) -> list[int]:
        """Closest candidates, at least min_spacing apart, up to max_rows_per_geom."""
        picked: list[int] = []
        for i in candidates[np.argsort(dist[candidates])]:
            if picked and self.min_spacing > 0.0:
                gaps = np.linalg.norm(self._centers[picked] - self._centers[i], axis=1)
                if np.min(gaps) < self.min_spacing:
                    continue
            picked.append(int(i))
            if len(picked) == self.max_rows_per_geom:
                break
        return picked

    def _compute_rows(self, configuration: Configuration) -> BarrierRows:
        model, data = self.model, configuration.data
        if len(self._centers) == 0 or len(self.capsules) == 0:
            return BarrierRows.empty(model.nv)
        a, b = self.capsules.segments(data)
        radius = self.capsules.radius
        dist, closest = point_capsule_distances(self._centers, a, b, radius)
        dist -= self._radii[:, None]

        distances, jacobians, witnesses = [], [], []
        for g in range(len(self.capsules)):
            candidates = np.nonzero(dist[:, g] < self.activation_distance)[0]
            if candidates.size == 0:
                continue
            for i in self._pick(candidates, dist[:, g]):
                offset = self._centers[i] - closest[i, g]
                norm = float(np.linalg.norm(offset))
                if norm < 1e-9:  # Centre on the axis: no defined direction.
                    continue
                n = offset / norm
                surface = closest[i, g] + radius[g] * n
                mujoco.mj_jac(
                    model, data, self._jacp, None, surface, self.capsules.body_ids[g]
                )
                distances.append(dist[i, g])
                jacobians.append(-n @ self._jacp)
                witnesses.append((surface, self._centers[i] - self._radii[i] * n))
        if not distances:
            return BarrierRows.empty(model.nv)
        return BarrierRows(
            np.array(distances), np.array(jacobians), np.array(witnesses)
        )
