:github_url: https://github.com/kevinzakka/mink/tree/main/docs/api/barriers.rst

.. _Barriers:

********
Barriers
********

A distance barrier turns robot-obstacle distances into rows of a control-barrier
function: each row is a signed distance :math:`d_k(q)` with its Jacobian
:math:`\partial d_k / \partial q`. :class:`~mink.BarrierLimit` enforces them as hard
constraints and :class:`~mink.ClearanceTask` as a soft cost; both can share one
barrier.

Definition
==========

.. automodule:: mink.barriers.barrier
    :members:

Point obstacles
===============

.. automodule:: mink.barriers.point_obstacle_barrier
    :members:

Geom pairs
==========

.. automodule:: mink.barriers.geom_pair_barrier
    :members:

Capsules
========

.. automodule:: mink.barriers.capsules
    :members:
