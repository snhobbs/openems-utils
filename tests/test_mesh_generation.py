import unittest
from openems_utils.mesh_generation import (
    _generate_axis_mesh,
    generate_symmetric_axis_mesh,
    generate_cartesian_meshes
)


class TestMeshGen(unittest.TestCase):
    def test_generate_axis_mesh_basic(self):
        mesh = _generate_axis_mesh([0, 10], [(2, 4)], coarse_mult=5, fine_step=1)
        self.assertEqual(sorted(mesh), [0, 2, 3, 4, 5, 10])

    def test_generate_axis_mesh_fine_only(self):
        mesh = _generate_axis_mesh([0, 2], [(0, 2)], coarse_mult=5, fine_step=0.5)
        self.assertTrue(all(abs(b - a - 0.5) < 1e-9 for a, b in zip(mesh, mesh[1:])))
        self.assertAlmostEqual(mesh[0], 0.0)
        self.assertAlmostEqual(mesh[-1], 2.0)

    def test_generate_axis_mesh_coarse_only(self):
        mesh = _generate_axis_mesh([0, 10], [], coarse_mult=2, fine_step=1)
        self.assertEqual(mesh, [0, 2, 4, 6, 8, 10])

    def test_generate_symmetric_axis_mesh(self):
        mesh = generate_symmetric_axis_mesh(lims=(-6, 6), fine_range=(-1, 1), fine_step=0.5, coarse_mult=4)
        self.assertEqual(len(mesh), 9)
        self.assertCountEqual(mesh, [-5.5, -3.5, -1.5, -1, 0, 1, 1.5, 3.5, 5.5])
        lim = 1.5 + 1e12
        self.assertTrue(all(v < lim or v > lim for v in mesh))

    def test_generate_cartesian_meshes(self):
        xlims = [-10, 10]
        ylims = [-5, 5]
        zlims = [-3, 3]
        port_start = (-2, -1, -0.5)
        port_stop = (2, 1, 0.5)

        meshx, meshy, meshz = generate_cartesian_meshes(
            [xlims, ylims, zlims],
            port_start=port_start,
            port_stop=port_stop,
            fine_step=0.5,
            coarse_mult=4
        )

        self.assertIn(0.0, meshx)
        self.assertIn(0.0, meshy)
        self.assertIn(0.0, meshz)
        self.assertTrue(all(-10 <= x <= 10 for x in meshx))
        self.assertTrue(all(-5 <= y <= 5 for y in meshy))
        self.assertTrue(all(-3 <= z <= 3 for z in meshz))


if __name__ == '__main__':
    unittest.main()
