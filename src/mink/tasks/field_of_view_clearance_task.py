"""Field-of-view clearance task implementation."""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..exceptions import TaskDefinitionError
from ._occlusion import Occluders, camera_pose, resolve_camera_id
from .task import Objective, Task


def _view_plane_normals(model: mujoco.MjModel, camera_id: int) -> np.ndarray:
    """Inward unit normals of the planes that bound a camera's view, in the camera
    frame (x right, y up, looking along -z). All planes pass through the camera
    origin."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = camera_id
    zver = np.zeros(2, dtype=np.float32)
    zhor = np.zeros(2, dtype=np.float32)
    zclip = np.zeros(2, dtype=np.float32)
    mujoco.mjv_cameraFrustum(zver, zhor, zclip, model, camera)

    # Extents of the near plane, divided by its depth: tangents of the view angles.
    bottom, top = zver.astype(float) / float(zclip[0])
    if np.any(model.cam_sensorsize[camera_id] > 0.0):
        left, right = zhor.astype(float) / float(zclip[0])
    else:
        # Without intrinsics MuJoCo leaves the width to the viewport.
        width, height = model.cam_resolution[camera_id]
        if width <= 1 and height <= 1:
            raise TaskDefinitionError(
                f"camera '{model.camera(camera_id).name}' needs a resolution or "
                "intrinsics to define its field of view."
            )
        left = right = top * width / height

    normals = np.array(
        [
            [1.0, 0.0, -left],
            [-1.0, 0.0, -right],
            [0.0, 1.0, -bottom],
            [0.0, -1.0, -top],
            # The plane through the camera that faces forward. Points inside the
            # four sides are already in front of it, but it lets a capsule that
            # reaches behind the camera count as out of view.
            [0.0, 0.0, -1.0],
        ]
    )
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


class FieldOfViewClearanceTask(Task):
    r"""Keep robot geoms out of a camera's field of view.

    Unlike :class:`~.LineOfSightTask`, there is no target: the task prefers
    configurations in which the occluder geoms do not appear in the image at all.
    It applies the self-occlusion avoidance of [VisibilityMax]_ to the whole image
    instead of one line of sight. The field of view is the pyramid bounded by the four side planes of the
    camera's frustum, which meet at the camera position :math:`p_c`. It is read
    from the MuJoCo camera (``fovy`` and ``resolution``, or the intrinsics) and
    has no depth limit.

    Let :math:`\hat{n}_k` be the inward normal of side plane :math:`k`, so that
    :math:`s_k(p) = \hat{n}_k^T (p - p_c)` is how far a point is inside that plane.
    A fifth plane through the camera, facing forward, is added to the four sides.
    An occluder with axis end points :math:`a_0, a_1` and radius :math:`r` is out
    of view when it is fully behind one of the planes. Its clearance is

    .. math::

        d = \max_k \left( -\max(s_k(a_0),\ s_k(a_1)) \right) - r,

    which is negative when the occluder is in view, and the task error is

    .. math::

        e = \min(d - m,\ 0),

    with :math:`m` the ``margin``. The Jacobian row of an active occluder uses the
    plane :math:`k` and end point :math:`p` that attain the maximum:

    .. math::

        J = -\hat{n}_k^T \left( J_a(p) - J_c(p) \right),

    where :math:`J_a(p)` is the translational Jacobian of :math:`p` on the
    occluder and :math:`J_c(p)` that of the same point carried by the camera.

    .. note::

        Nothing in this task keeps anything in view. If the hand has to be in the
        image to reach its goal, the solver may turn the camera away instead. Add a
        :class:`~.LineOfSightTask` or a :class:`~.LookAtTask` for what must stay
        visible.

    Attributes:
        camera_id: Id of the camera.
        margin: Clearance :math:`m` to keep between an occluder and the view.

    Example:

    .. code-block:: python

        arm_geoms = mink.get_subtree_geom_ids(model, model.body("arm_base").id)
        task = FieldOfViewClearanceTask(
            model, camera="head", occluders=arm_geoms, cost=5.0
        )
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        camera: str | int,
        occluders: Sequence[str | int],
        cost: npt.ArrayLike,
        margin: float = 0.0,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        """Constructor.

        Args:
            model: MuJoCo model.
            camera: Name or id of the camera. It needs a resolution or intrinsics.
            occluders: Names or ids of the geoms to keep out of view. Spheres are
                exact, capsules and cylinders use their axis and radius, and any
                other type is replaced by its bounding sphere.
            cost: Cost of the task. A scalar, or one value per occluder.
            margin: Clearance in [m] to keep between an occluder and the view.
            gain: Task gain in [0, 1] for additional low-pass filtering.
            lm_damping: Levenberg-Marquardt damping.
        """
        self._occluders = Occluders(model, occluders)
        self.k = len(self._occluders)
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)

        if margin < 0.0:
            raise TaskDefinitionError(
                f"{self.__class__.__name__} margin must be >= 0 but got {margin}"
            )
        self.camera_id = resolve_camera_id(model, camera)
        self.margin = margin
        self._normals = _view_plane_normals(model, self.camera_id)
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

    def _active_planes(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Clearance of every occluder, with the end point and the world-frame
        plane normal that set it."""
        camera_position, camera_rotation = camera_pose(configuration, self.camera_id)
        normals = self._normals @ camera_rotation.T
        start, end = self._occluders.segments(configuration.data)
        inside_start = (start - camera_position) @ normals.T
        inside_end = (end - camera_position) @ normals.T

        # Per plane, the end point that is deepest inside decides; the occluder is
        # out of view as soon as one plane has it fully behind.
        deepest = np.maximum(inside_start, inside_end)
        plane = np.argmin(deepest, axis=1)
        rows = np.arange(self.k)
        clearance = -deepest[rows, plane] - self._occluders.radius
        use_end = inside_end[rows, plane] > inside_start[rows, plane]
        point = np.where(use_end[:, None], end, start)
        return clearance, point, normals[plane]

    def compute_clearance(self, configuration: Configuration) -> np.ndarray:
        """Compute the clearance of every occluder from the field of view.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Clearances in [m], one per occluder. A negative value means the
            occluder is in view.
        """
        return self._active_planes(configuration)[0]

    def _error_and_jacobian(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray]:
        clearance, point, normal = self._active_planes(configuration)
        error = np.minimum(clearance - self.margin, 0.0)
        jacobian = np.zeros((self.k, configuration.nv))

        model, data = configuration.model, configuration.data
        jac_point, jac_camera = self._jac_point, self._jac_camera
        for i in np.flatnonzero(error < 0.0):
            body_id = self._occluders.body_ids[i]
            mujoco.mj_jac(model, data, jac_point, None, point[i], body_id)
            # The plane turns with the camera, so compare against the same point
            # carried by the camera's body.
            mujoco.mj_jac(model, data, jac_camera, None, point[i], self._camera_body_id)
            jacobian[i] = -normal[i] @ (jac_point - jac_camera)
        return error, jacobian

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the field-of-view clearance task error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task error :math:`e(q) = \min(d - m, 0)`, one entry per occluder.
        """
        error, _ = self._error_and_jacobian(configuration)
        return error

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the field-of-view clearance task Jacobian.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Task Jacobian :math:`J(q)`, one row per occluder. Rows of occluders
            that are out of view by more than ``margin`` are zero.
        """
        _, jacobian = self._error_and_jacobian(configuration)
        return jacobian

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        Overrides the base implementation to compute the clearances once and reuse
        them for both the error and the Jacobian.
        """
        error, jacobian = self._error_and_jacobian(configuration)
        return self._assemble_qp(error, jacobian, configuration._eye_nv)

    def compute_qp_residual(
        self, configuration: Configuration
    ) -> tuple[np.ndarray, np.ndarray, float]:
        error, jacobian = self._error_and_jacobian(configuration)
        return self._weighted_residual(error, jacobian)
