"""All distance barriers derive from the :class:`DistanceBarrier` base class."""

from __future__ import annotations

import abc
from typing import NamedTuple

import numpy as np

from ..configuration import Configuration


class BarrierRows(NamedTuple):
    r"""Signed distances :math:`d_k(q)` and their Jacobians at one configuration.

    Each row is one witness pair: a point on the robot and the closest point of an
    obstacle to it. A positive distance is a gap, a negative one a penetration.
    """

    distance: np.ndarray
    r"""Signed distances :math:`d_k` in [m], shape (k,)."""
    jacobian: np.ndarray
    r"""Jacobians :math:`\partial d_k / \partial q`, shape (k, :math:`n_v`)."""
    witness: np.ndarray
    """Witness points in the world frame, shape (k, 2, 3): ``witness[:, 0]`` lies on
    the robot (the first geom of a pair) and ``witness[:, 1]`` on the obstacle."""

    @property
    def size(self) -> int:
        """Number of rows."""
        return int(self.distance.shape[0])

    @staticmethod
    def empty(nv: int) -> BarrierRows:
        """No rows."""
        return BarrierRows(np.zeros(0), np.zeros((0, nv)), np.zeros((0, 2, 3)))


class DistanceBarrier(abc.ABC):
    r"""Signed distances between a robot and obstacles, used as barrier functions.

    A distance barrier produces a varying set of rows :math:`k`, each a signed distance
    :math:`d_k(q)` between a point on the robot and an obstacle, with its Jacobian
    :math:`\partial d_k / \partial q`. Only pairs closer than
    ``activation_distance`` produce rows. :class:`~mink.BarrierLimit` turns the rows
    into hard control-barrier constraints and :class:`~mink.ClearanceTask` into a soft
    cost; both may share one barrier.

    Rows are cached for the current configuration (joint and mocap state) and the
    barrier's own state, so a limit and a task that share a barrier compute them once
    per solve. Subclasses implement :meth:`_compute_rows` and call
    :meth:`_invalidate` whenever their state changes.

    Attributes:
        activation_distance: Distance in [m] below which a pair produces a row.
    """

    def __init__(self, activation_distance: float):
        self.activation_distance = activation_distance
        self._state_version = 0
        self._cache_key: tuple | None = None
        self._cache_rows: BarrierRows | None = None

    @abc.abstractmethod
    def _compute_rows(self, configuration: Configuration) -> BarrierRows:
        """Compute the rows at the current configuration, without caching."""
        raise NotImplementedError

    def _invalidate(self) -> None:
        """Discard cached rows; call when anything besides the configuration changes."""
        self._state_version += 1

    def compute_rows(self, configuration: Configuration) -> BarrierRows:
        """Signed distances and Jacobians of the active pairs at a configuration.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            The active rows.
        """
        data = configuration.data
        key = (
            id(data),
            data.qpos.tobytes(),
            data.mocap_pos.tobytes(),
            data.mocap_quat.tobytes(),
            self._state_version,
        )
        if key != self._cache_key or self._cache_rows is None:
            self._cache_rows = self._compute_rows(configuration)
            self._cache_key = key
        return self._cache_rows
