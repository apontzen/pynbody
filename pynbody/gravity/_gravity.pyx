#cython: embedsignature=True

cimport cython

import numpy as np

from pynbody import array, config, openmp, units

cimport numpy as np

np.import_array()

DTYPE = np.double

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

cdef extern from "math.h" nogil:
      double sqrt(double)
      float sqrt(float)


def _reconcile_dtypes(arrays, allow_coerce):
    """Bring a set of named arrays onto a single floating point dtype, or explain why we can't.

    The direct summation kernel is compiled separately for single and double precision, so every
    array handed to it must share one dtype. This routine works out what that dtype should be.

    Parameters
    ----------

    arrays : dict
        Maps a human-readable name onto the array it describes. Insertion order is preserved in
        any error message, so pass the arrays in the order most helpful to a reader.

    allow_coerce : bool
        If True, the arrays are promoted to ``float64`` when any one of them is already
        ``float64``, and otherwise left as ``float32``. If False, any mismatch raises a
        ``ValueError`` instead of silently making copies.

    Returns
    -------

    arrays : dict
        The input arrays, cast where required.

    dtype : numpy.dtype
        The dtype now shared by all the returned arrays.

    """

    float32 = np.dtype(np.float32)
    float64 = np.dtype(np.float64)

    dtypes = {name: np.asarray(ar).dtype for name, ar in arrays.items()}
    distinct = set(dtypes.values())

    if distinct == {float32} or distinct == {float64}:
        return arrays, distinct.pop()

    if not allow_coerce:
        described = ", ".join(f"{name} is {dtype}" for name, dtype in dtypes.items())
        raise ValueError(
            "The arrays needed for the direct gravity calculation do not share a single "
            f"floating point dtype ({described}).\n\n"
            "The calculation is compiled separately for single and double precision, so it "
            "cannot mix the two. Either make the dtypes consistent yourself -- for example by "
            "recreating the offending array with the same dtype as f['pos'] -- or pass "
            "allow_coerce=True to have pynbody promote everything to float64 for you. Coercion "
            "is not the default because it makes temporary float64 copies of the positions and "
            "masses, which for a large snapshot can require a substantial amount of extra memory."
        )

    # Anything that is not already single precision is promoted to double, so that coercion
    # never quietly throws away precision.
    dtype = float32 if distinct == {float32} else float64

    return {name: np.asarray(ar, dtype=dtype) for name, ar in arrays.items()}, dtype


def direct(f, ipos, eps=None, int num_threads = 0, allow_coerce=False):
    global config

    if num_threads == 0 :
        num_threads = int(config["number_of_threads"])

    if num_threads < 0:
        num_threads = openmp.get_cpus()

    if num_threads > openmp.get_cpus() :
        num_threads = openmp.get_cpus()

    openmp.set_threads(num_threads)

    if eps is None:
        try:
            eps = f['eps']
        except KeyError:
            eps = f.properties['eps']

    if isinstance(eps, str):
        eps = units.Unit(eps)

    if isinstance(eps, units.UnitBase):
        eps = eps.in_units(f['pos'].units, **f.conversion_context())

    # Note that IndexedSimArray is not a subclass of SimArray, so both have to be named here for
    # the softening of a subsnap (e.g. a halo, or the result of a filter) to be converted.
    if isinstance(eps, (array.SimArray, array.IndexedSimArray)):
        eps = eps.in_units(f['pos'].units, **f.conversion_context())

    # A scalar softening carries no dtype of its own, so it is only materialised once the dtype
    # of everything else has been settled.
    eps_is_scalar = np.ndim(eps) == 0

    arrays = {'ipos': np.asarray(ipos), 'pos': np.asarray(f['pos']), 'mass': np.asarray(f['mass'])}

    if not eps_is_scalar:
        arrays['eps'] = np.asarray(eps)

        if len(arrays['eps']) != len(arrays['pos']):
            raise ValueError(
                f"The softening array has length {len(arrays['eps'])}, but the snapshot has "
                f"{len(arrays['pos'])} particles"
            )

    arrays, dtype = _reconcile_dtypes(arrays, allow_coerce)

    if eps_is_scalar:
        arrays['eps'] = np.repeat(np.asarray(eps, dtype=dtype), len(arrays['pos']))

    m_by_r, m_by_r2 = _direct(arrays['ipos'], arrays['pos'], arrays['mass'],
                              arrays['eps'] * arrays['eps'])

    pot = array.SimArray(-m_by_r,units=f['mass'].units/f['pos'].units * units.G)
    accel = array.SimArray(-m_by_r2,units=f['mass'].units/f['pos'].units**2 * units.G)

    return pot, accel


@cython.cdivision(True)
@cython.boundscheck(False)
def _direct(np.ndarray[DTYPE_t, ndim=2] ipos, np.ndarray[DTYPE_t, ndim=2] pos,
            np.ndarray[DTYPE_t, ndim=1] mass, np.ndarray[DTYPE_t, ndim=1] epssq):
    from cython.parallel cimport prange

    cdef Py_ssize_t nips = len(ipos)
    cdef np.ndarray[DTYPE_t, ndim=2] m_by_r2 = np.zeros((nips,3), dtype = ipos.dtype)
    cdef np.ndarray[DTYPE_t, ndim=1] m_by_r = np.zeros(nips, dtype = ipos.dtype)
    cdef Py_ssize_t n = len(mass)

    cdef Py_ssize_t pi, i
    cdef double dx, dy, dz, mass_i, epssq_i, drsoft, drsoft3

    for pi in prange(nips, nogil=True, schedule='static'):
        for i in range(n):
            mass_i = mass[i]
            epssq_i = epssq[i]
            dx = ipos[pi,0] - pos[i,0]
            dy = ipos[pi,1] - pos[i,1]
            dz = ipos[pi,2] - pos[i,2]
            drsoft = 1.0/sqrt(dx*dx + dy*dy + dz*dz + epssq_i)
            drsoft3 = drsoft*drsoft*drsoft
            m_by_r[pi] += mass_i * drsoft
            m_by_r2[pi,0] += mass_i*dx * drsoft3
            m_by_r2[pi,1] += mass_i*dy * drsoft3
            m_by_r2[pi,2] += mass_i*dz * drsoft3

    return m_by_r, m_by_r2
