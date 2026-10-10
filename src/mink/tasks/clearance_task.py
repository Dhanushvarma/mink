"""Soft clearance from obstacles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..configuration import Configuration
from ..exceptions import InvalidDamping, InvalidGain, TaskDefinitionError
from .task import BaseTask, Objective

if TYPE_CHECKING:
    from ..barriers.barrier import DistanceBarrier


class ClearanceTask(BaseTask):
    r"""Prefer to keep the distances of a :class:`~mink.DistanceBarrier` above a target.

    Each barrier row :math:`k` closer than the preferred distance :math:`d_p`
    contributes the error :math:`e_k = d_k - d_p < 0` with Jacobian
    :math:`J_k = \partial d_k / \partial q`, weighted by

    .. math::

        w_k = c\, s\!\left(\frac{d_p - d_k}{\delta}\right),
        \qquad s(x) = 3x^2 - 2x^3 \text{ on } [0, 1],

    where :math:`c` is the cost and :math:`\delta` the ramp width, so rows enter and
    leave the cost smoothly. As for other tasks, the first-order dynamics
    :math:`J_k \Delta q = -\alpha e_k` push the robot back toward :math:`d_p`.

    Traded against an end-effector task, clearance is regained through null-space
    motion first, and the end effector gives way only as far as the costs allow.
    Pair it with a :class:`~mink.BarrierLimit` on the same barrier: the task starts
    evading early and smoothly, the limit guarantees the minimum distance. This plays
    the role of the safe-displacement term of pink's barriers [PinkBarriers]_.

    Attributes:
        barrier: Source of the distances and Jacobians.
        preferred_distance: Distance :math:`d_p` in [m] to prefer.
        cost: Cost :math:`c` per [m] of clearance error.
        gain: Task gain :math:`\alpha` in [0, 1].
        ramp_width: Width :math:`\delta` in [m] of the activation ramp.
        lm_damping: Levenberg-Marquardt damping.

    Example:

    .. code-block:: python

        barrier = PointObstacleBarrier(model, geoms=robot_geoms)
        tasks = [ee_task, ClearanceTask(barrier, preferred_distance=0.15, cost=0.5)]
    """

    def __init__(
        self,
        barrier: DistanceBarrier,
        preferred_distance: float,
        cost: float = 1.0,
        gain: float = 1.0,
        ramp_width: float = 0.05,
        lm_damping: float = 0.0,
    ):
        """Constructor.

        Args:
            barrier: Source of the distances and Jacobians.
            preferred_distance: Distance :math:`d_p` in [m] to prefer. It must not
                exceed the barrier's activation distance, beyond which there are no
                rows.
            cost: Cost :math:`c` per [m] of clearance error, non-negative.
            gain: Task gain :math:`\\alpha` in [0, 1].
            ramp_width: Width :math:`\\delta` in [m] of the activation ramp, positive.
            lm_damping: Levenberg-Marquardt damping, non-negative.
        """
        if not 0.0 <= gain <= 1.0:
            raise InvalidGain("`gain` must be in the range [0, 1]")
        if lm_damping < 0.0:
            raise InvalidDamping("`lm_damping` must be >= 0")
        if cost < 0.0:
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")
        if ramp_width <= 0.0:
            raise TaskDefinitionError(
                f"{self.__class__.__name__} ramp_width must be > 0"
            )
        if preferred_distance > barrier.activation_distance:
            raise TaskDefinitionError(
                f"{self.__class__.__name__}: preferred_distance ({preferred_distance}) "
                "must not exceed the barrier's activation distance "
                f"({barrier.activation_distance})"
            )
        self.barrier = barrier
        self.preferred_distance = preferred_distance
        self.cost = cost
        self.gain = gain
        self.ramp_width = ramp_width
        self.lm_damping = lm_damping

    def compute_qp_residual(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Weighted residual of the rows closer than the preferred distance.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Weighted Jacobian, weighted error, and the Levenberg-Marquardt term.
        """
        rows = self.barrier.compute_rows(configuration)
        active = rows.distance < self.preferred_distance
        error = rows.distance[active] - self.preferred_distance
        x = np.clip(-error / self.ramp_width, 0.0, 1.0)
        weight = self.cost * x * x * (3.0 - 2.0 * x)
        weighted_jacobian = weight[:, None] * rows.jacobian[active]
        weighted_error = weight * (-self.gain * error)
        mu = self.lm_damping * float(weighted_error @ weighted_error)
        return weighted_jacobian, weighted_error, mu

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Pair :math:`(H(q), c(q))`.
        """
        weighted_jacobian, weighted_error, mu = self.compute_qp_residual(configuration)
        H = weighted_jacobian.T @ weighted_jacobian
        H += mu * configuration._eye_nv
        c = -weighted_error @ weighted_jacobian
        return Objective(H, c)
