"""Line-of-sight task implementation."""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from ._occlusion import (
    Occluders,
    camera_pose,
    closest_points_to_segment,
    resolve_camera_id,
)
from .task import Objective, Task


class LineOfSightTask(Task):
    r"""Keep robot geoms away from the line of sight between a camera and a target.

    The line of sight is the segment from the camera position :math:`p_c` to a
    target point :math:`p_t` in the world. An occluder geom blocks the target when
    it crosses that segment. This is the self-occlusion term of [VisibilityMax]_,
    written as a soft task instead of an inequality with a slack variable.

    For each occluder, let :math:`p_a` be the point on its axis closest to the line
    of sight, :math:`p_v` the matching point on the line of sight, and :math:`r` its
    radius. Its clearance is

    .. math::

        d = \|p_a - p_v\| - r,

    and the task error is zero unless the occluder is closer than ``distance``
    :math:`D`:

    .. math::

        e = \min(d - D,\ 0).

    With :math:`\hat{n} = (p_a - p_v) / \|p_a - p_v\|` and :math:`\kappa \in [0, 1]`
    the position of :math:`p_v` along the line of sight (1 at the camera), the
    Jacobian row of an active occluder is

    .. math::

        J = \hat{n}^T \left( J_a - \kappa\, J_c \right),

    where :math:`J_a` and :math:`J_c` are the translational Jacobians of
    :math:`p_a` and of the camera. The task therefore moves the occluder, the
    camera, or both, whichever the other tasks make cheapest.

    .. note::

        The end-effector has to enter the line of sight to reach a target that
        sits on it. Leave the gripper geoms out of ``occluders``, or lower the
        cost near the goal, as [VisibilityMax]_ does with its slack weight.

    Attributes:
        camera_id: Id of the camera.
        distance: Clearance :math:`D` below which an occluder is pushed away.
        target: Target point in the world frame.

    Example:

    .. code-block:: python

        arm_geoms = mink.get_subtree_geom_ids(model, model.body("arm_base").id)
        task = LineOfSightTask(
            model, camera="head", occluders=arm_geoms, cost=10.0, distance=0.1
        )
        task.set_target(object_position)
    """

    target: np.ndarray | None

    def __init__(
        self,
        model: mujoco.MjModel,
        camera: str | int,
        occluders: Sequence[str | int],
        cost: npt.ArrayLike,
        distance: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Constructor.

        Args:
            model: MuJoCo model.
            camera: Name or id of the camera.
            occluders: Names or ids of the geoms to keep off the line of sight.
                Spheres and capsules are exact, a cylinder is treated as a
                capsule, and any other type is replaced by its bounding sphere.
            cost: Cost of the task. A scalar, or one value per occluder.
            distance: Clearance in [m] below which an occluder is pushed away.
            gain: Task gain in [0, 1] for additional low-pass filtering.
            lm_damping: Levenberg-Marquardt damping.
        """
        self._occluders = Occluders(model, occluders)
        self.k = len(self._occluders)
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)

        if distance <= 0.0:
            raise TaskDefinitionError(
                f"{self.__class__.__name__} distance must be > 0 but got {distance}"
            )
        self.camera_id = resolve_camera_id(model, camera)
        self.distance = distance
        self.target = None
        self._camera_body_id = int(model.cam_bodyid[self.camera_id])
        self._jac_point = np.empty((3, model.nv))
        self._jac_camera = np.empty((3, model.nv))

        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        """Set the cost of the task.

        Args:
            cost: A scalar, or a vector with one value per occluder.
        """
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"(aka identical cost for all occluders) or ({self.k},). Got "
                f"{cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")
        self.cost[:] = cost

    def set_target(self, target: npt.ArrayLike) -> None:
        """Set the point that should stay visible, in the world frame.

        Args:
            target: A vector of shape (3,).
        """
        target = np.atleast_1d(np.asarray(target, dtype=float))
        if target.shape != (3,):
            raise InvalidTarget(
                f"Expected target to have shape (3,) but got {target.shape}"
            )
        self.target = target.copy()

    def _closest_points(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Clearance, axis point, unit normal and line parameter of every
        occluder, plus the camera position."""
        if self.target is None:
            raise TargetNotSet(self.__class__.__name__)
        camera_position, _ = camera_pose(configuration, self.camera_id)
        start, end = self._occluders.segments(configuration.data)
        s, kappa = closest_points_to_segment(start, end, self.target, camera_position)
        point = start + s[:, None] * (end - start)
        on_line = self.target + kappa[:, None] * (camera_position - self.target)
        normal = point - on_line
        norm = np.linalg.norm(normal, axis=1)
        # An axis that touches the line of sight has no direction to move in.
        normal = normal / np.where(norm > 0.0, norm, np.inf)[:, None]
        clearance = norm - self._occluders.radius
        return clearance, point, normal, kappa, camera_position

    def compute_clearance(self, configuration: Configuration) -> np.ndarray:
        """Compute the clearance of every occluder from the line of sight.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Clearances in [m], one per occluder. A negative value means the
            occluder blocks the target.
        """
        return self._closest_points(configuration)[0]

    def _error_and_jacobian(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray]:
        clearance, point, normal, kappa, camera_position = self._closest_points(
            configuration
        )
        error = np.minimum(clearance - self.distance, 0.0)
        jacobian = np.zeros((self.k, configuration.nv))
        active = np.flatnonzero(error < 0.0)
        if active.size == 0:
            return error, jacobian

        model, data = configuration.model, configuration.data
        jac_point, jac_camera = self._jac_point, self._jac_camera
        mujoco.mj_jac(
            model, data, jac_camera, None, camera_position, self._camera_body_id
        )
        for i in active:
            body_id = self._occluders.body_ids[i]
            mujoco.mj_jac(model, data, jac_point, None, point[i], body_id)
            jacobian[i] = normal[i] @ (jac_point - kappa[i] * jac_camera)
        return error, jacobian

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the line-of-sight task error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error :math:`e(q) = \min(d - D, 0)`, one entry per occluder.
        """
        error, _ = self._error_and_jacobian(configuration)
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the line-of-sight task Jacobian.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task Jacobian :math:`J(q)`, one row per occluder. Rows of occluders
            farther than ``distance`` are zero.
        """
        _, jacobian = self._error_and_jacobian(configuration)
        return jacobian

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        Overrides the base implementation to compute the closest points once and
        reuse them for both the error and the Jacobian.
        """
        error, jacobian = self._error_and_jacobian(configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)

    def compute_qp_residual(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray, float]:
        error, jacobian = self._error_and_jacobian(configuration)
        return self._weighted_residual(error, jacobian)
