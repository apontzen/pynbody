# Cube Kernel for AMR Simulations

This document describes the new cube kernel implementation for rendering AMR (Adaptive Mesh Refinement) simulation data, particularly from RAMSES simulations.

## Overview

Traditional SPH (Smoothed Particle Hydrodynamics) kernels are spherically symmetric and designed for smooth particle distributions. However, AMR simulations organize data in cubic cells aligned with the coordinate axes. The new `CubeKernel` properly handles this cubic geometry, providing more accurate visualizations of AMR data.

## Key Features

- **Cartesian Geometry**: Uses actual (x, y, z) coordinates rather than just radial distance
- **Sharp Boundaries**: Uniform value inside the cube, zero outside (no artificial smoothing)
- **Proper Normalization**: Correctly preserves mass/density when rendering
- **Efficient**: Direct evaluation in Cython for performance

## Implementation Details

### 3D Cube Kernel

For a cell with smoothing length `h`, the cube extends from `-h` to `+h` in each direction (total side length `2h`):

- **Volume**: `(2h)³ = 8h³`
- **Kernel value inside**: `1/(8h³)` (uniform)
- **Kernel value outside**: `0`

The kernel is evaluated using the condition:
```
value = 1/(8h³)  if |x| < h AND |y| < h AND |z| < h
        0         otherwise
```

### 2D Projection

When projecting along the z-axis, the result is a square footprint:

- **Area**: `(2h)² = 4h²`
- **Kernel value inside**: `1/(4h²)` (uniform)
- **Kernel value outside**: `0`

The 2D projection is evaluated using:
```
value = 1/(4h²)  if |x| < h AND |y| < h
        0         otherwise
```

## Usage

### Basic Example

```python
import pynbody
import pynbody.plot.sph as sph

# Load a RAMSES simulation
sim = pynbody.load('output_00001')

# Create an image using the cube kernel
sph.image(sim.gas, qty='rho', width='10 kpc', kernel='cube')
```

### Explicit Kernel Creation

```python
from pynbody.sph.kernels import CubeKernel

# Create kernel instance
kernel = CubeKernel()

# Use for rendering
sph.image(sim.gas, qty='rho', width='10 kpc', kernel=kernel)
```

### Comparison with SPH Kernels

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Cubic spline kernel (default for SPH)
plt.subplot(1, 3, 1)
sph.image(sim.gas, qty='rho', width='10 kpc', kernel='cubic')
plt.title('Cubic Spline Kernel (SPH)')

# Wendland C2 kernel (used by EAGLE)
plt.subplot(1, 3, 2)
sph.image(sim.gas, qty='rho', width='10 kpc', kernel='wendlandc2')
plt.title('Wendland C2 Kernel')

# Cube kernel (for AMR)
plt.subplot(1, 3, 3)
sph.image(sim.gas, qty='rho', width='10 kpc', kernel='cube')
plt.title('Cube Kernel (AMR)')

plt.tight_layout()
plt.show()
```

## When to Use the Cube Kernel

### Recommended For:
- **RAMSES simulations**: AMR data naturally organized in cubic cells
- **Enzo simulations**: Another AMR code with cubic geometry
- **Any AMR code**: Where data represents volume-averaged quantities in cubes

### Not Recommended For:
- **SPH simulations**: Use `CubicSplineKernel` or `WendlandC2Kernel` instead
- **Moving-mesh codes**: These may need specialized handling

## Technical Details

### Cartesian vs. Spherical Kernels

Traditional kernels only depend on radial distance `r = sqrt(x² + y² + z²)`. The cube kernel uses separate x, y, z coordinates:

- **Traditional**: `W(r, h)` where `r = |x|`
- **Cube kernel**: `W(x, y, z, h)` with independent axis checks

### Performance Considerations

The cube kernel evaluates slightly differently from spherical kernels:

1. **Sample Generation**: Skipped for cube kernel (direct evaluation)
2. **Cartesian Check**: Uses `if abs(x) < h and abs(y) < h and abs(z) < h`
3. **Overhead**: Minimal - comparable to other kernels

### Rendering Pipeline

The rendering code automatically detects cartesian kernels via the `uses_cartesian` flag and uses the optimized `get_kernel_cube()` function in Cython for fast evaluation.

## Validation

The cube kernel has been validated to:

1. ✓ Return uniform values inside the cube/square
2. ✓ Return zero outside the boundaries
3. ✓ Preserve normalization (integral = 1)
4. ✓ Work with 2D projections and 3D grids
5. ✓ Handle boundary cases correctly

## Implementation Files

- `pynbody/sph/kernels.py`: Python kernel classes
  - `CubeKernel`: 3D cube kernel
  - `CubeKernelProjection`: 2D projection
- `pynbody/sph/kernels.hpp`: C++ kernel (fallback)
- `pynbody/sph/_render.pyx`: Cython rendering code
  - `get_kernel_cube()`: Fast Cartesian evaluation

## Limitations

1. **Axis Alignment**: The cube is always aligned with the x, y, z axes (no rotation)
2. **Fixed Shape**: Always a cube/square (no rectangular cells)
3. **Single h**: Each cell has one smoothing length h

These limitations match the typical AMR cell geometry, so they don't affect most use cases.

## Future Enhancements

Possible future improvements:

- Support for rectangular cells (different h_x, h_y, h_z)
- Rotated kernels for tilted coordinate systems
- Adaptive kernel shapes based on cell aspect ratios

## References

- RAMSES User Guide: https://bitbucket.org/rteyssie/ramses
- Pynbody Documentation: https://pynbody.github.io/pynbody/
- AMR Methods: Berger & Colella (1989), JCP

## Questions and Support

For questions or issues with the cube kernel:

1. Check the pynbody documentation
2. Open an issue on GitHub: https://github.com/pynbody/pynbody/issues
3. Join the pynbody discussions

---

**Note**: This implementation provides a significant improvement for AMR visualization by respecting the underlying cell geometry rather than imposing spherical symmetry.
