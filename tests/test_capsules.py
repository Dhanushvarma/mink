"""Tests for barriers/capsules.py."""

import itertools

import mujoco
import numpy as np
from absl.testing import absltest

from mink import GeomCapsules, fit_body_capsule, fit_capsule
from mink.barriers import min_enclosing_circle, point_capsule_distances
from mink.barriers.capsules import _circle_through


def _brute_force_circle(points: np.ndarray) -> float:
    """Radius of the smallest enclosing circle, by checking every pair and triple."""
    best = np.inf
    for k in (1, 2, 3):
        for combo in itertools.combinations(points, k):
            centre, r = _circle_through(list(combo))
            if np.all(np.linalg.norm(points - centre, axis=1) <= r + 1e-9):
                best = min(best, r)
    return best


class TestMinEnclosingCircle(absltest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            points = rng.normal(size=(12, 2))
            centre, r = min_enclosing_circle(points)
            self.assertTrue(np.all(np.linalg.norm(points - centre, axis=1) <= r + 1e-9))
            self.assertAlmostEqual(r, _brute_force_circle(points), places=9)

    def test_single_and_collinear_points(self):
        centre, r = min_enclosing_circle(np.array([[1.0, 2.0]]))
        np.testing.assert_allclose(centre, [1.0, 2.0])
        self.assertEqual(r, 0.0)
        line = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
        centre, r = min_enclosing_circle(line)
        np.testing.assert_allclose(centre, [1.5, 0.0])
        self.assertAlmostEqual(r, 1.5)


class TestFitCapsule(absltest.TestCase):
    def _sample_capsule(self, p0, p1, radius, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        t = rng.uniform(size=n)
        direction = rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        return p0 + t[:, None] * (p1 - p0) + radius * direction

    def test_recovers_a_capsule(self):
        p0, p1 = np.array([0.1, -0.2, 0.0]), np.array([0.1, 0.2, 0.05])
        points = self._sample_capsule(p0, p1, 0.04)
        a, b, r = fit_capsule(points, axis=p1 - p0)
        self.assertAlmostEqual(r, 0.04, delta=1e-3)
        ends = sorted([a, b], key=lambda p: p[1])
        np.testing.assert_allclose(ends[0], p0, atol=3e-3)
        np.testing.assert_allclose(ends[1], p1, atol=3e-3)

    def test_encloses_all_vertices(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(500, 3)) * [0.05, 0.2, 0.03]
        for axis in (None, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]):
            a, b, r = fit_capsule(points, axis=axis)
            dist, _ = point_capsule_distances(points, a[None], b[None], np.array([r]))
            self.assertLessEqual(dist.max(), 1e-9)

    def test_flat_object_gets_a_sphere(self):
        angle = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        disc = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], 1)
        a, b, r = fit_capsule(disc, axis=[0.0, 0.0, 1.0])
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(r, 1.0, places=6)

    def test_fit_body_capsule_uses_the_body_frame(self):
        xml = """
        <mujoco>
          <asset>
            <mesh name="bar" vertex="-0.2 -0.02 -0.02  0.2 -0.02 -0.02  -0.2 0.02 -0.02
                                     0.2 0.02 -0.02  -0.2 -0.02 0.02  0.2 -0.02 0.02
                                     -0.2 0.02 0.02  0.2 0.02 0.02"/>
          </asset>
          <worldbody>
            <body name="b" pos="1 2 3">
              <geom type="mesh" mesh="bar" pos="0 0 0.5" euler="0 0 90"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        a, b, r = fit_body_capsule(model, "b", axis=[0.0, 1.0, 0.0])
        # The bar is turned to lie along y and lifted by 0.5 in the body frame. Its
        # corners sit on the radius, so the segment reaches the end faces.
        np.testing.assert_allclose(sorted([a[1], b[1]]), [-0.2, 0.2], atol=1e-7)
        np.testing.assert_allclose([a[2], b[2]], [0.5, 0.5], atol=1e-7)
        self.assertAlmostEqual(r, np.hypot(0.02, 0.02), places=7)  # float32 vertices
        with self.assertRaises(ValueError):
            fit_body_capsule(model, "world")


class TestGeomCapsules(absltest.TestCase):
    def test_segments_by_geom_type(self):
        xml = """
        <mujoco>
          <worldbody>
            <geom name="s" type="sphere" size="0.1" pos="1 0 0"/>
            <geom name="c" type="capsule" size="0.05 0.2" pos="0 1 0" euler="90 0 0"/>
            <geom name="y" type="cylinder" size="0.04 0.1" pos="0 0 1"/>
            <geom name="b" type="box" size="0.1 0.2 0.3" pos="2 2 2"/>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        capsules = GeomCapsules(model, ["s", "c", "y", "b"])
        a, b = capsules.segments(data)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(np.linalg.norm(b[1] - a[1]), 0.4)
        np.testing.assert_allclose(np.abs(b[1] - a[1]), [0.0, 0.4, 0.0], atol=1e-12)
        np.testing.assert_allclose(b[2] - a[2], [0.0, 0.0, 0.2])
        np.testing.assert_allclose(a[3], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(
            capsules.radius, np.array([0.1, 0.05, 0.04, np.sqrt(0.14)])
        )

    def test_point_distances_match_sampling(self):
        a = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        b = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        radius = np.array([0.1, 0.2])
        rng = np.random.default_rng(2)
        points = rng.uniform(-1.0, 2.0, size=(30, 3))
        dist, _ = point_capsule_distances(points, a, b, radius)
        t = np.linspace(0.0, 1.0, 20001)
        for j in range(2):
            axis = a[j] + t[:, None] * (b[j] - a[j])
            for i, p in enumerate(points):
                expected = np.min(np.linalg.norm(axis - p, axis=1)) - radius[j]
                self.assertAlmostEqual(dist[i, j], expected, places=6)


if __name__ == "__main__":
    absltest.main()
