"""Control-barrier limit on robot-obstacle distances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from ..configuration import Configuration
from ..exceptions import LimitDefinitionError
from .limit import Constraint, Limit

if TYPE_CHECKING:
    from ..barriers.barrier import DistanceBarrier


class BarrierLimit(Limit):
    r"""Keep the distances of a :class:`~mink.DistanceBarrier` above a minimum.

    Each row :math:`k` of the barrier defines the barrier function
    :math:`h_k(q) = d_k(q) - d_{min}`, and the limit enforces the control-barrier
    condition [CBF]_

    .. math::

        \dot{h}_k \geq -\gamma\, \alpha(h_k),

    with :math:`\gamma` the gain in [1/s] and :math:`\alpha` an extended class-K
    function (the identity by default). As in pink's barriers [PinkBarriers]_, it is
    applied over the integration step:

    .. math::

        -J_k \Delta q \leq dt\, \gamma\, \alpha(h_k),
        \qquad J_k = \frac{\partial d_k}{\partial q}.

    The robot may approach an obstacle no faster than :math:`\gamma\, \alpha(h)`, and
    must back away when :math:`h < 0`. For :math:`h > 0` the bound is capped at
    :math:`h`, so a large :math:`\gamma\, dt` cannot step past :math:`d_{min}`. With
    the identity for :math:`\alpha`, this is :class:`~mink.CollisionAvoidanceLimit`
    with ``gain`` :math:`= \gamma\, dt`.

    Backing away can be infeasible together with other limits, for example when an
    obstacle closes in faster than the joints can retreat, or from two sides at once.
    Raise ``relaxation`` and solve again: negative bounds are scaled by
    :math:`1 - \text{relaxation}`, so the robot is asked to back away more slowly,
    and at 1 they are clipped to zero, :math:`\Delta q = 0` is feasible, and the
    robot holds still instead. Approach stays limited throughout.

    Attributes:
        barrier: Source of the distances and Jacobians.
        minimum_distance: Distance :math:`d_{min}` in [m] to keep.
        gain: Barrier gain :math:`\gamma` in [1/s].
        gain_function: Extended class-K function :math:`\alpha`.
        relaxation: In [0, 1]; scales negative bounds by ``1 - relaxation``.

    Example:

    .. code-block:: python

        barrier = PointObstacleBarrier(model, geoms=robot_geoms)
        limits = [ConfigurationLimit(model), BarrierLimit(barrier, 0.03, gain=10.0)]
        barrier.set_obstacles(points)
        v = solve_ik(configuration, tasks, dt, "daqp", limits=limits)
    """

    def __init__(
        self,
        barrier: DistanceBarrier,
        minimum_distance: float = 0.0,
        gain: float = 10.0,
        gain_function: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        """Constructor.

        Args:
            barrier: Source of the distances and Jacobians.
            minimum_distance: Distance :math:`d_{min}` in [m] to keep.
            gain: Barrier gain :math:`\\gamma` in [1/s], positive.
            gain_function: Extended class-K function :math:`\\alpha`, applied
                elementwise. Defaults to the identity.

        Raises:
            LimitDefinitionError: If the gain is not positive, or the barrier's
                activation distance does not exceed the minimum distance (the limit
                would only switch on once the minimum was already violated).
        """
        if gain <= 0.0:
            raise LimitDefinitionError(f"{self.__class__.__name__} gain must be > 0")
        if barrier.activation_distance <= minimum_distance:
            raise LimitDefinitionError(
                f"{self.__class__.__name__}: the barrier's activation distance "
                f"({barrier.activation_distance}) must exceed the minimum distance "
                f"({minimum_distance})"
            )
        self.barrier = barrier
        self.minimum_distance = minimum_distance
        self.gain = gain
        self.gain_function = gain_function
        self.relaxation = 0.0

    def compute_qp_inequalities(
        self, configuration: Configuration, dt: float
    ) -> Constraint:
        r"""Compute the barrier constraints :math:`G \Delta q \leq h`.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Integration timestep in [s].

        Returns:
            Pair :math:`(G, h)` with one row per active barrier row, or an inactive
            constraint when no pair is within the activation distance.
        """
        rows = self.barrier.compute_rows(configuration)
        if rows.size == 0:
            return Constraint()
        h = rows.distance - self.minimum_distance
        alpha = h if self.gain_function is None else self.gain_function(h)
        bound = dt * self.gain * np.asarray(alpha, dtype=float)
        bound = np.where(h > 0.0, np.minimum(bound, h), bound)
        if self.relaxation > 0.0:
            scale = 1.0 - min(max(self.relaxation, 0.0), 1.0)
            bound = np.where(bound < 0.0, scale * bound, bound)
        return Constraint(G=-rows.jacobian, h=bound)
