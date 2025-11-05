"""
Unit tests for the cube kernel implementation for AMR simulations.

These tests verify that the cube kernel properly handles Cartesian geometry
of AMR cells, unlike traditional SPH kernels which are spherically symmetric.
"""

import numpy as np
import numpy.testing as npt
import pytest

import pynbody
from pynbody.sph import kernels


class TestCubeKernel:
    """Tests for the 3D cube kernel."""

    def test_kernel_creation(self):
        """Test that CubeKernel can be created and has correct attributes."""
        kernel = kernels.CubeKernel()

        assert kernel.h_power == 3
        assert kernel.uses_cartesian == True
        assert kernel.get_c_kernel_id() == 2
        # For a cube, max_d should be sqrt(3) (corner distance)
        npt.assert_allclose(kernel.max_d, np.sqrt(3.0), rtol=1e-5)

    def test_kernel_values_3d(self):
        """Test that the 3D cube kernel returns correct values."""
        kernel = kernels.CubeKernel()
        h = 1.0

        # Expected value inside cube: 1/(8h³) = 0.125
        expected_inside = 1.0 / (8.0 * h**3)

        # Test points inside the cube
        inside_points = [
            (0.0, 0.0, 0.0),  # center
            (0.5, 0.5, 0.5),  # inside
            (0.99, 0.0, 0.0), # near edge x
            (0.0, 0.99, 0.0), # near edge y
            (0.0, 0.0, 0.99), # near edge z
            (0.99, 0.99, 0.99), # near corner
        ]

        for x, y, z in inside_points:
            val = kernel.get_value_xyz(x, y, z, h)
            npt.assert_allclose(val, expected_inside, rtol=1e-10,
                              err_msg=f"Failed at point ({x}, {y}, {z})")

        # Test points outside the cube
        outside_points = [
            (1.1, 0.0, 0.0),   # outside x
            (0.0, 1.1, 0.0),   # outside y
            (0.0, 0.0, 1.1),   # outside z
            (1.1, 1.1, 1.1),   # outside corner
            (0.5, 0.5, 1.5),   # mixed
        ]

        for x, y, z in outside_points:
            val = kernel.get_value_xyz(x, y, z, h)
            npt.assert_allclose(val, 0.0, atol=1e-10,
                              err_msg=f"Failed at point ({x}, {y}, {z})")

    def test_kernel_symmetry_3d(self):
        """Test that the cube kernel is symmetric in all axes."""
        kernel = kernels.CubeKernel()
        h = 1.0

        # Test a point in the positive octant
        x, y, z = 0.5, 0.3, 0.7
        val_pos = kernel.get_value_xyz(x, y, z, h)

        # Test symmetry in all 8 octants
        for sx, sy, sz in [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),
                           (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)]:
            val = kernel.get_value_xyz(sx*x, sy*y, sz*z, h)
            npt.assert_allclose(val, val_pos, rtol=1e-10)

    def test_kernel_scaling_3d(self):
        """Test that the cube kernel scales correctly with h."""
        kernel = kernels.CubeKernel()

        # Test at different smoothing lengths
        for h in [0.5, 1.0, 2.0]:
            # Inside point
            val_inside = kernel.get_value_xyz(0.5*h, 0.5*h, 0.5*h, h)
            expected = 1.0 / (8.0 * h**3)
            npt.assert_allclose(val_inside, expected, rtol=1e-10)

            # Outside point
            val_outside = kernel.get_value_xyz(1.1*h, 0.0, 0.0, h)
            npt.assert_allclose(val_outside, 0.0, atol=1e-10)

    def test_projection_2d(self):
        """Test that the 2D projection of the cube kernel works correctly."""
        kernel_3d = kernels.CubeKernel()
        kernel_2d = kernel_3d.projection()

        assert isinstance(kernel_2d, kernels.CubeKernelProjection)
        assert kernel_2d.h_power == 2
        assert kernel_2d.uses_cartesian == True
        # For a square, max_d should be sqrt(2) (corner distance)
        npt.assert_allclose(kernel_2d.max_d, np.sqrt(2.0), rtol=1e-5)

    def test_projection_values_2d(self):
        """Test that the 2D projection returns correct values."""
        kernel_3d = kernels.CubeKernel()
        kernel_2d = kernel_3d.projection()
        h = 1.0

        # Expected value inside square: 1/(4h²) = 0.25
        expected_inside = 1.0 / (4.0 * h**2)

        # Test points inside the square (z should be ignored)
        inside_points = [
            (0.0, 0.0),
            (0.5, 0.5),
            (0.99, 0.0),
            (0.0, 0.99),
            (0.99, 0.99),
        ]

        for x, y in inside_points:
            val = kernel_2d.get_value_xyz(x, y, 0.0, h)
            npt.assert_allclose(val, expected_inside, rtol=1e-10,
                              err_msg=f"Failed at point ({x}, {y})")

            # Test with different z values (should be ignored)
            val_z = kernel_2d.get_value_xyz(x, y, 5.0, h)
            npt.assert_allclose(val_z, expected_inside, rtol=1e-10)

        # Test points outside the square
        outside_points = [
            (1.1, 0.0),
            (0.0, 1.1),
            (1.1, 1.1),
        ]

        for x, y in outside_points:
            val = kernel_2d.get_value_xyz(x, y, 0.0, h)
            npt.assert_allclose(val, 0.0, atol=1e-10,
                              err_msg=f"Failed at point ({x}, {y})")

    def test_create_kernel_by_name(self):
        """Test that cube kernel can be created using string specification."""
        # Test various string specifications
        kernel1 = kernels.create_kernel('cube')
        assert isinstance(kernel1, kernels.CubeKernel)

        kernel2 = kernels.create_kernel('CubeKernel')
        assert isinstance(kernel2, kernels.CubeKernel)

        kernel3 = kernels.create_kernel('CUBE')
        assert isinstance(kernel3, kernels.CubeKernel)


class TestCubeKernelRendering:
    """Tests for rendering with the cube kernel."""

    @pytest.fixture
    def amr_like_snapshot(self):
        """Create a synthetic snapshot mimicking AMR data."""
        n_cells = 1000
        np.random.seed(42)

        f = pynbody.new(n_cells)

        # Create a grid-like distribution (like AMR cells)
        # Cells arranged in a rough cubic grid
        grid_size = int(np.ceil(n_cells**(1/3)))
        positions = []
        for i in range(n_cells):
            ix = (i % grid_size) / grid_size
            iy = ((i // grid_size) % grid_size) / grid_size
            iz = (i // (grid_size**2)) / grid_size
            # Add small random offset to avoid perfect grid
            positions.append([ix + np.random.normal(0, 0.01),
                            iy + np.random.normal(0, 0.01),
                            iz + np.random.normal(0, 0.01)])

        f['pos'] = np.array(positions)
        f['pos'].units = 'kpc'

        # Set uniform masses (like AMR cells)
        f['mass'] = np.ones(n_cells) / n_cells
        f['mass'].units = 'Msol'

        # Set smoothing lengths to cell size
        f['smooth'] = np.ones(n_cells) * (1.0 / grid_size) * 0.5
        f['smooth'].units = 'kpc'

        # Set density
        f['rho'] = f['mass'] / (f['smooth']**3)

        return f

    def test_render_with_cube_kernel(self, amr_like_snapshot):
        """Test that rendering with cube kernel works without errors."""
        f = amr_like_snapshot

        # Render with cube kernel
        im_cube = pynbody.sph.render_image(f, nx=50, ny=50, width=1.0,
                                           kernel='cube',
                                           approximate_fast=False)

        # Check that image has correct shape
        assert im_cube.shape == (50, 50)

        # Check that image has reasonable values (not all zeros or NaN)
        assert np.isfinite(im_cube).all()
        assert (im_cube >= 0).all()  # density should be non-negative
        assert im_cube.sum() > 0  # should have some signal

    def test_cube_vs_sph_kernel(self, amr_like_snapshot):
        """Test that cube kernel produces different results from SPH kernels."""
        f = amr_like_snapshot

        # Render with different kernels
        im_cube = pynbody.sph.render_image(f, nx=50, ny=50, width=1.0,
                                           kernel='cube',
                                           approximate_fast=False)

        im_cubic_spline = pynbody.sph.render_image(f, nx=50, ny=50, width=1.0,
                                                    kernel='cubicspline',
                                                    approximate_fast=False)

        # Images should be different (cube kernel has sharp boundaries)
        assert not np.allclose(im_cube, im_cubic_spline, rtol=0.1)

        # Both images should have reasonable values
        assert im_cube.sum() > 0
        assert im_cubic_spline.sum() > 0
        assert np.isfinite(im_cube).all()
        assert np.isfinite(im_cubic_spline).all()

    def test_cube_kernel_3d_grid(self, amr_like_snapshot):
        """Test rendering to 3D grid with cube kernel."""
        f = amr_like_snapshot

        # Render to 3D grid (x2 sets the full width, grid is cubic)
        grid = pynbody.sph.render_3d_grid(f, nx=20,
                                         x2=1.0,
                                         kernel='cube',
                                         approximate_fast=False)

        # Check shape
        assert grid.shape == (20, 20, 20)

        # Check for finite values
        assert np.isfinite(grid).all()
        assert (grid >= 0).all()

    def test_cube_kernel_2d_projection(self, amr_like_snapshot):
        """Test that 2D projection with cube kernel works correctly."""
        f = amr_like_snapshot

        # Create 2D projection (surface density)
        im_2d = pynbody.sph.render_image(f, nx=50, ny=50, width=1.0,
                                        kernel='cube',
                                        out_units='Msol kpc^-2',
                                        approximate_fast=False)

        # Check units
        assert str(im_2d.units) == 'Msol kpc**-2'

        # Check that it's a valid 2D projection
        assert im_2d.shape == (50, 50)
        assert np.isfinite(im_2d).all()
        assert (im_2d >= 0).all()


class TestCubeKernelNormalization:
    """Tests for cube kernel normalization and conservation properties."""

    def test_3d_kernel_normalization(self):
        """Test that 3D cube kernel integrates to approximately 1."""
        kernel = kernels.CubeKernel()
        h = 1.0

        # Sample the kernel on a fine grid
        n_samples = 50
        x = np.linspace(-1.5*h, 1.5*h, n_samples)
        y = np.linspace(-1.5*h, 1.5*h, n_samples)
        z = np.linspace(-1.5*h, 1.5*h, n_samples)
        dx = x[1] - x[0]

        integral = 0.0
        for xi in x:
            for yi in y:
                for zi in z:
                    integral += kernel.get_value_xyz(xi, yi, zi, h) * dx**3

        # Should integrate to approximately 1 (within discrete sampling error)
        # The sharp boundaries of the cube kernel make discrete integration less accurate
        npt.assert_allclose(integral, 1.0, rtol=0.1)

    def test_2d_kernel_normalization(self):
        """Test that 2D cube kernel integrates to approximately 1."""
        kernel_3d = kernels.CubeKernel()
        kernel_2d = kernel_3d.projection()
        h = 1.0

        # Sample the kernel on a fine grid
        n_samples = 50
        x = np.linspace(-1.5*h, 1.5*h, n_samples)
        y = np.linspace(-1.5*h, 1.5*h, n_samples)
        dx = x[1] - x[0]

        integral = 0.0
        for xi in x:
            for yi in y:
                integral += kernel_2d.get_value_xyz(xi, yi, 0.0, h) * dx**2

        # Should integrate to approximately 1 (within discrete sampling error)
        # The sharp boundaries of the cube kernel make discrete integration less accurate
        npt.assert_allclose(integral, 1.0, rtol=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
