"""Manipulability task implementation."""

from __future__ import annotations

from typing import Sequence, SupportsFloat

import mujoco
import numpy as np

from ..configuration import Configuration, _resolve_frame_id
from ..exceptions import TaskDefinitionError
from .task import BaseTask, Objective

_AXES_TO_ROWS = {
    "all": (0, 1, 2, 3, 4, 5),
    "translation": (0, 1, 2),
    "rotation": (3, 4, 5),
}
_METRICS = ("yoshikawa", "icn")


class ManipulabilityTask(BaseTask):
    r"""Move toward configurations where a frame is easier to move.

    The task looks at the frame Jacobian restricted to a set of DOFs :math:`S`,
    :math:`J_S(q)`, with singular values :math:`\sigma_1 \ge \dots \ge \sigma_k`, and
    maximizes one of two measures:

    * ``"yoshikawa"``: :math:`m(q) = \sqrt{\det(J_S J_S^T)} = \prod_i \sigma_i`
      [Yoshikawa]_.
    * ``"icn"``: the inverse condition number :math:`m(q) = \sigma_k / \sigma_1`, as
      in [M4Diffuser]_.

    It contributes a linear term only, as in [HolisticMoMa]_:

    .. math::

        -\lambda\, \nabla m(q)^T \Delta \mathbf{q},

    where :math:`\lambda` is the ``cost``. There is no target and no Hessian. The size
    of the step is set by the other tasks and by the ``damping`` of
    :func:`~mink.solve_ik`, so use it next to a :class:`~.PostureTask` or a
    :class:`~.DampingTask`.

    The gradient is analytic. With :math:`J_S = U \Sigma V^T`,

    .. math::

        \frac{\partial m}{\partial q_j}
        = \sum_i \frac{\partial m}{\partial \sigma_i}\,
          u_i^T \frac{\partial J_S}{\partial q_j} v_i,

    and :math:`\partial J_S / \partial q_j` comes from cross products of Jacobian
    columns [ManipulatorHessian]_. Hinge, slide, ball and free joints are supported.

    .. note::

        On a mobile manipulator, pass the arm DOFs as ``dof_indices``. The base can
        always move the frame, so with the base columns included the measure stops
        seeing the arm lose reach [HolisticMoMa]_. With a planar base the gradient
        has no base component either way: the base moves because the other tasks
        still have to hold while the arm reconfigures.

    Attributes:
        frame_name: Name or id of the frame, typically a body, geom or site.
        frame_type: The frame type: ``body``, ``geom`` or ``site``.
        cost: Weight :math:`\lambda` of the linear term.
        dof_indices: Sorted DOF indices whose Jacobian columns define the measure.
        metric: ``"yoshikawa"`` or ``"icn"``.
        axes: ``"all"``, ``"translation"`` or ``"rotation"``.

    Example:

    .. code-block:: python

        # Planar base on DOFs 0-2, six-DOF arm on DOFs 3-8.
        manipulability_task = ManipulabilityTask(
            model,
            frame_name="pinch",
            frame_type="site",
            cost=1.0,
            dof_indices=range(3, 9),
        )
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        frame_name: str | int,
        frame_type: str,
        cost: SupportsFloat,
        dof_indices: Sequence[int] | None = None,
        metric: str = "yoshikawa",
        axes: str = "all",
    ):
        """Constructor.

        Args:
            model: MuJoCo model.
            frame_name: Name or id of the frame.
            frame_type: The frame type: ``body``, ``geom`` or ``site``.
            cost: Weight of the task, a scalar >= 0.
            dof_indices: DOFs whose Jacobian columns define the measure. Defaults
                to every DOF that moves the frame.
            metric: ``"yoshikawa"`` or ``"icn"``.
            axes: Rows of the Jacobian to use: ``"all"``, ``"translation"`` (linear
                velocity) or ``"rotation"`` (angular velocity).
        """
        if metric not in _METRICS:
            raise TaskDefinitionError(
                f"{self.__class__.__name__} metric must be one of {_METRICS} but got "
                f"'{metric}'"
            )
        if axes not in _AXES_TO_ROWS:
            raise TaskDefinitionError(
                f"{self.__class__.__name__} axes must be one of "
                f"{tuple(_AXES_TO_ROWS)} but got '{axes}'"
            )
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.metric = metric
        self.axes = axes
        self._rows = np.array(_AXES_TO_ROWS[axes])
        self.set_cost(cost)

        frame_id = _resolve_frame_id(model, frame_name, frame_type)
        if frame_type == "body":
            body_id = frame_id
        elif frame_type == "geom":
            body_id = int(model.geom_bodyid[frame_id])
        else:
            body_id = int(model.site_bodyid[frame_id])

        # DOFs between the world and the frame. All others have a zero Jacobian
        # column and a zero gradient.
        chain: list[int] = []
        # Rotational DOFs of a ball or free joint share an id; every other DOF is -1.
        rotation_group = np.full(model.nv, -1)
        while body_id != 0:
            adr, num = int(model.body_dofadr[body_id]), int(model.body_dofnum[body_id])
            chain.extend(range(adr, adr + num))
            jnt_adr = int(model.body_jntadr[body_id])
            for jnt_id in range(jnt_adr, jnt_adr + int(model.body_jntnum[body_id])):
                dof_adr = int(model.jnt_dofadr[jnt_id])
                if model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_BALL:
                    rotation_group[dof_adr : dof_adr + 3] = jnt_id
                elif model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                    rotation_group[dof_adr + 3 : dof_adr + 6] = jnt_id
            body_id = int(model.body_parentid[body_id])
        chain.sort()

        if dof_indices is None:
            dof_indices = chain
        dof_indices = [int(dof) for dof in dof_indices]
        for dof in dof_indices:
            if dof < 0 or dof >= model.nv:
                raise TaskDefinitionError(
                    f"DOF index {dof} is out of range [0, {model.nv})."
                )
        if len(dof_indices) != len(set(dof_indices)):
            raise TaskDefinitionError(f"Duplicate DOF indices found: {dof_indices}.")
        not_in_chain = sorted(set(dof_indices) - set(chain))
        if not_in_chain:
            raise TaskDefinitionError(
                f"DOF indices {not_in_chain} do not move {frame_type} '{frame_name}'."
            )
        if len(dof_indices) < len(self._rows):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} with axes '{axes}' needs at least "
                f"{len(self._rows)} DOFs but got {len(dof_indices)}."
            )
        self.dof_indices = sorted(dof_indices)

        self._chain = np.array(chain)
        self._selected = np.searchsorted(self._chain, self.dof_indices)
        # _precedes[j, i]: moving chain DOF j carries the axis of selected DOF i
        # with it. True for j at or before i in the chain, and for any two
        # rotational DOFs of one ball or free joint, whose axes are fixed in the
        # child body.
        j = self._chain[:, None]
        i = np.array(self.dof_indices)[None, :]
        same_group = (rotation_group[j] == rotation_group[i]) & (rotation_group[j] >= 0)
        self._precedes = ((j <= i) | same_group).astype(float)

    def set_cost(self, cost: SupportsFloat) -> None:
        """Set the weight of the task.

        Args:
            cost: A scalar >= 0.
        """
        cost = float(cost)
        if cost < 0.0:
            raise TaskDefinitionError(f"{self.__class__.__name__} cost should be >= 0")
        self.cost = cost

    def _selected_jacobian(self, jac: np.ndarray) -> np.ndarray:
        return jac[np.ix_(self._rows, self.dof_indices)]

    def compute_manipulability(self, configuration: Configuration) -> float:
        """Compute the manipulability measure at the current configuration.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            The measure :math:`m(q)` selected by ``metric``.
        """
        jac = configuration._get_frame_jacobian_world_aligned(
            self.frame_name, self.frame_type
        )
        sigma = np.linalg.svd(self._selected_jacobian(jac), compute_uv=False)
        if self.metric == "yoshikawa":
            return float(np.prod(sigma))
        return float(sigma[-1] / sigma[0]) if sigma[0] > 0.0 else 0.0

    def compute_gradient(self, configuration: Configuration) -> np.ndarray:
        r"""Compute the gradient of the manipulability measure.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Gradient :math:`\nabla m(q)` in the tangent space, of shape
            (:math:`n_v`,).
        """
        jac = configuration._get_frame_jacobian_world_aligned(
            self.frame_name, self.frame_type
        )
        gradient = np.zeros(configuration.nv)
        U, sigma, Vt = np.linalg.svd(self._selected_jacobian(jac), full_matrices=False)

        # weights[i] = dm / dsigma_i. Written without a division by sigma_i so the
        # Yoshikawa gradient stays finite at a singularity.
        weights = np.zeros_like(sigma)
        if self.metric == "yoshikawa":
            for i in range(sigma.shape[0]):
                weights[i] = np.prod(np.delete(sigma, i))
        else:
            if sigma[0] <= 0.0:
                return gradient
            weights[-1] += 1.0 / sigma[0]
            weights[0] -= sigma[-1] / sigma[0] ** 2

        # dm/dq_j = <dJ_S/dq_j, G> with G = U diag(weights) V^T.
        G = np.zeros((6, len(self.dof_indices)))
        G[self._rows] = (U * weights) @ Vt
        G_v, G_w = G[:3].T, G[3:].T

        v, w = jac[:3, self._chain].T, jac[3:, self._chain].T
        v_s, w_s = v[self._selected], w[self._selected]

        # Column i of dJ/dq_j is [w_j x v_i; w_j x w_i] where j precedes i, and
        # [w_i x v_j; 0] elsewhere. The triple products are regrouped so the sum
        # over i is a single matrix product.
        a = np.cross(v_s, G_v) + np.cross(w_s, G_w)
        b = np.cross(G_v, w_s)
        gradient[self._chain] = np.einsum("jk,jk->j", w, self._precedes @ a)
        gradient[self._chain] += np.einsum("jk,jk->j", v, (1.0 - self._precedes) @ b)
        return gradient

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        r"""Compute the matrix-vector pair :math:`(H, c)` of the QP objective.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Pair :math:`(0, -\lambda \nabla m(q))`.
        """
        nv = configuration.nv
        H = np.zeros((nv, nv))
        if self.cost == 0.0:
            return Objective(H, np.zeros(nv))
        return Objective(H, -self.cost * self.compute_gradient(configuration))
