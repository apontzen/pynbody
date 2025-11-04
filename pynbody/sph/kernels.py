"""SPH kernel details"""

from __future__ import annotations

import numpy as np
import scipy.integrate as integrate


class KernelBase:
    """Base class for SPH kernels"""

    _sample_cache = {}

    def __init__(self):
        self.h_power = 3
        # Return the power of the smoothing length which appears in
        # the denominator of the expression for the general kernel.
        # Will be 3 for 3D kernels, 2 for 2D kernels.

        self.max_d = 2
        # The maximum value of the displacement over the smoothing for
        # which the kernel is non-zero

        self.uses_cartesian = False
        # If True, the kernel depends on (x,y,z) coordinates rather than just
        # radial distance. The rendering code will call get_value_xyz instead
        # of using the precomputed samples.

    def _get_samples_from_cache(self):
        if hash(self) in KernelBase._sample_cache:
            return KernelBase._sample_cache[hash(self)]
        else:
            return None

    def get_samples(self, dtype=np.float32):
        import time
        s = time.time()

        samples = self._get_samples_from_cache()
        if samples is None:
            sample_pts = np.arange(0, 4.01, 0.02)
            samples = np.array([self.get_value(x ** 0.5) for x in sample_pts], dtype=dtype)
            KernelBase._sample_cache[hash(self)] = samples

        return samples

    def get_value(self, d, h=1) -> float:
        """Get the value of the kernel for a given smoothing length."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_value_xyz(self, x, y, z, h=1) -> float:
        """Get the value of the kernel given Cartesian coordinates.

        This is used for kernels that depend on the actual (x,y,z) position
        rather than just radial distance. By default, this falls back to
        computing the radial distance and calling get_value.

        Parameters
        ----------
        x, y, z : float
            Coordinates relative to the particle center
        h : float
            Smoothing length

        Returns
        -------
        float
            Kernel value at (x,y,z)
        """
        d = np.sqrt(x**2 + y**2 + z**2)
        return self.get_value(d, h)

    def projection(self) -> KernelBase:
        """Return a 2D projection of this kernel"""
        return Kernel2D(self)

    def __hash__(self):
        return hash(self.__class__)

    @classmethod
    def get_c_kernel_id(cls) -> int:
        """Return the C kernel id for this kernel

        This is used to select the appropriate C code for the kernel, and must match
        the kernel id defined in the Kernel::create function in kernels.hpp"""
        raise NotImplementedError("Subclasses must implement this method")

class CubicSplineKernel(KernelBase):
    """A cubic spline kernel. This is the default kernel used by pynbody."""
    def get_value(self, d, h=1):
        if d < 1:
            f = 1. - (3. / 2) * d ** 2 + (3. / 4.) * d ** 3
        elif d < 2:
            f = 0.25 * (2. - d) ** 3
        else:
            f = 0

        return f / (np.pi * h ** 3)

    @classmethod
    def get_c_kernel_id(cls):
        return 0

class WendlandC2Kernel(KernelBase):
    """A Wendland C2 (quintic) kernel. This is the default kernel used by EAGLE."""

    def get_value(self, d, h=1):
        if d < 2:
            f = (1. - (d / 2.))**4 * (2. * d + 1)
        else:
            f = 0

        return (21. * f) / (16. * np.pi * h ** 3)

    @classmethod
    def get_c_kernel_id(cls):
        return 1


class CubeKernel(KernelBase):
    """A cube/top-hat kernel appropriate for AMR simulations.

    This kernel represents a cubic cell with side length 2h, properly handling
    the Cartesian geometry of AMR cells. Unlike SPH kernels which are spherically
    symmetric, this kernel returns a constant value inside the cube defined by
    |x| < h, |y| < h, |z| < h, and zero outside.

    This is more appropriate than SPH kernels for AMR data (e.g., RAMSES simulations)
    where the data is naturally organized in cubic cells rather than smooth particles.

    Note: This kernel uses Cartesian coordinates (x,y,z) rather than just radial
    distance, so it evaluates directly rather than using precomputed lookup tables.
    """

    def __init__(self):
        super().__init__()
        self.h_power = 3
        # For a cube, max_d is the distance to the corner: sqrt(3)*h
        self.max_d = np.sqrt(3.0)  # ≈ 1.732
        self.uses_cartesian = True

    def get_value(self, d, h=1):
        """Get the value of the 3D cube kernel based on radial distance.

        Note: This method provides a spherical approximation for compatibility,
        but is not the preferred evaluation method. Use get_value_xyz for
        accurate cubic geometry.
        """
        # Approximate cube with inscribed sphere for compatibility
        if d < h:
            return 1.0 / (8.0 * h ** 3)
        else:
            return 0.0

    def get_value_xyz(self, x, y, z, h=1) -> float:
        """Get the value of the 3D cube kernel using Cartesian coordinates.

        The cube extends from -h to +h in each direction, with uniform density.

        Parameters
        ----------
        x, y, z : float
            Coordinates relative to the cell center
        h : float
            Half the cell side length

        Returns
        -------
        float
            1/(8h³) if inside the cube, 0 otherwise
        """
        if abs(x) < h and abs(y) < h and abs(z) < h:
            # Uniform density: 1 / volume, where volume = (2h)³ = 8h³
            return 1.0 / (8.0 * h ** 3)
        else:
            return 0.0

    def projection(self) -> KernelBase:
        """Return a 2D projection of the cube kernel."""
        return CubeKernelProjection()

    @classmethod
    def get_c_kernel_id(cls):
        return 2


class CubeKernelProjection(KernelBase):
    """2D projection of a cube kernel.

    Represents the projection of a cubic cell along the z-axis. The result is a square
    footprint in the xy-plane with side length 2h. The kernel returns a constant value
    inside the square defined by |x| < h and |y| < h, and zero outside.

    Note: This kernel uses Cartesian coordinates (x,y) rather than just radial
    distance, so it evaluates directly rather than using precomputed lookup tables.
    """

    def __init__(self):
        super().__init__()
        self.h_power = 2
        # For a square, max_d is the distance to the corner: sqrt(2)*h
        self.max_d = np.sqrt(2.0)  # ≈ 1.414
        self.uses_cartesian = True

    def get_value(self, d, h=1):
        """Get the value of the 2D projected cube kernel based on radial distance.

        Note: This method provides a circular approximation for compatibility,
        but is not the preferred evaluation method. Use get_value_xyz for
        accurate square geometry.
        """
        # Approximate square with inscribed circle for compatibility
        if d < h:
            return 1.0 / (4.0 * h ** 2)
        else:
            return 0.0

    def get_value_xyz(self, x, y, z, h=1) -> float:
        """Get the value of the 2D projected cube kernel using Cartesian coordinates.

        The square extends from -h to +h in x and y directions. The z coordinate
        is ignored for 2D projections.

        Parameters
        ----------
        x, y : float
            Coordinates in the image plane relative to the cell center
        z : float
            Ignored for 2D projections
        h : float
            Half the cell side length

        Returns
        -------
        float
            1/(4h²) if inside the square, 0 otherwise
        """
        if abs(x) < h and abs(y) < h:
            # Uniform surface density: 1 / area, where area = (2h)² = 4h²
            return 1.0 / (4.0 * h ** 2)
        else:
            return 0.0

    def projection(self):
        raise ValueError("Cannot project a 2D kernel")

    @classmethod
    def get_c_kernel_id(cls):
        return 2


class Kernel2D(KernelBase):
    """A 2D spline kernel, generated by numerically projecting an underlying 3D kernel"""
    def __init__(self, k_orig=CubicSplineKernel()):
        """Create a 2D kernel by projecting a 3D kernel. The 3D kernel is passed as an argument."""
        self.h_power = 2
        self.max_d = k_orig.max_d
        self.k_orig = k_orig

    def projection(self):
        raise ValueError("Cannot project a 2D kernel")

    def get_value(self, d, h=1):
        return 2 * integrate.quad(lambda z: self.k_orig.get_value(np.sqrt(z ** 2 + d ** 2), h), 0, 2*h)[0]

    def get_c_kernel_id(self):
        raise NotImplementedError("2D kernels are not supported in C")

    def __hash__(self):
        return hash((self.__class__, self.k_orig))


def create_kernel(spec) -> KernelBase:
    """Create a kernel object from a string specification, a type, an existing kernel object, or a None

    This function is used to create a kernel object from a variety of input types. It is used by the
    framework to allow the user flexibility in specifying the kernel type.

    If the input is a string, it is assumed to be the name of a kernel class, and an object of that class
    is created. You can use the name of the class with or without the 'Kernel' suffix, and the case is
    ignored. For example, 'WendlandC2Kernel', 'wendlandc2', and 'WendlandC2' all return a WendlandC2Kernel
    instance.

    If the input is a subclass of KernelBase, it is assumed to be a kernel object, and is returned as is.

    If the input is None, a default kernel is created and returned.

    Returns
    -------
    KernelBase
        A kernel object

    """
    if spec is None:
        from ..configuration import config
        return create_kernel(config['sph'].get('kernel', 'CubicSplineKernel'))
    elif isinstance(spec, type):
        return spec()
    elif isinstance(spec, KernelBase):
        return spec
    elif isinstance(spec, str):
        for subclass in KernelBase.__subclasses__():
            subclass_name = subclass.__name__
            if (subclass_name.lower() == spec.lower() or
                    subclass_name.endswith('Kernel') and subclass_name[:-6].lower() == spec.lower()):
                return subclass()
        else:
            raise ValueError("Unknown kernel '%s'" % spec)
    else:
        raise ValueError("Unknown kernel specification %r" % spec)
