"""Exact distances between geom pairs."""

from __future__ import annotations

import mujoco
import numpy as np

from ..configuration import Configuration
from ..limits.collision_avoidance_limit import (
    CollisionAvoidanceLimit,
    CollisionPairs,
    compute_contact_normal_jacobian,
)
from .barrier import BarrierRows, DistanceBarrier


class GeomPairBarrier(DistanceBarrier):
    r"""Exact signed distances between geom pairs, from ``mj_geomDistance``.

    Pairs are formed and filtered as in :class:`~mink.CollisionAvoidanceLimit`. Each
    pair closer than ``activation_distance`` gives one row with

    .. math::

        \frac{\partial d}{\partial q} = n^T (J_2 - J_1),

    where :math:`n` is the unit vector from the witness point on the first geom to the
    one on the second (reversed under penetration) and :math:`J_1, J_2` are the
    translational Jacobians of the witness points.

    The obstacles are geoms of the model, so moving ones (mocap bodies) must be posed
    in ``configuration.data`` before solving, then ``configuration.update()``.

    Attributes:
        model: MuJoCo model.
        geom_id_pairs: The geom id pairs that are checked.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        activation_distance: float = 0.25,
    ):
        """Constructor.

        Args:
            model: MuJoCo model.
            geom_pairs: Pairs of geom groups, as for
                :class:`~mink.CollisionAvoidanceLimit`.
            activation_distance: Distance in [m] below which a pair produces a row.
        """
        super().__init__(activation_distance)
        self.model = model
        pairs = CollisionAvoidanceLimit(
            model,
            geom_pairs,
            minimum_distance_from_collisions=0.0,
            collision_detection_distance=activation_distance,
        )
        self.geom_id_pairs = sorted(pairs.geom_id_pairs)
        self._fromto = np.empty(6)
        self._normal = np.empty(3)
        self._jac1 = np.empty((3, model.nv))
        self._jac2 = np.empty((3, model.nv))

    def _compute_rows(self, configuration: Configuration) -> BarrierRows:
        model, data = self.model, configuration.data
        distances, jacobians, witnesses = [], [], []
        for g1, g2 in self.geom_id_pairs:
            dist = mujoco.mj_geomDistance(
                model, data, g1, g2, self.activation_distance, self._fromto
            )
            if dist >= self.activation_distance - 1e-12:
                continue
            row = compute_contact_normal_jacobian(
                model, data, g1, g2, self._fromto, self._normal, self._jac1, self._jac2
            )
            distances.append(dist)
            jacobians.append(row if dist >= 0.0 else -row)
            witnesses.append((self._fromto[:3].copy(), self._fromto[3:].copy()))
        if not distances:
            return BarrierRows.empty(model.nv)
        return BarrierRows(
            np.array(distances), np.array(jacobians), np.array(witnesses)
        )
