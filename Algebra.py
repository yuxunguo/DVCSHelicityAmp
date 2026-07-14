"""Linear-algebra building blocks for helicity-amplitude calculations.

This module centralizes the numerical conventions used throughout the
repository. Four-vectors are contravariant arrays ordered as
``[E, px, py, pz]`` and are contracted with the mostly-minus metric
``diag(1, -1, -1, -1)``. Helicity labels are integer doubled-helicity values
``+1`` and ``-1``. Electron spinors are massless by default but accept an
explicit mass, while proton spinors always carry a mass argument.

The functions are intentionally small and validation-heavy because they form
the lowest layer used by both the Bethe-Heitler amplitude and spin-density
matrix scans.
"""

import numpy as np

from config import ELECTRON_MASS_GEV

# ============================================================
# Conventions
# g^{mu nu} = diag(1,-1,-1,-1)
# Four-vectors are numpy arrays [E, px, py, pz]
# h, hp, s, sp, lam = +/- 1
# Electron is massless. Proton has mass m.
# ============================================================

DEFAULT_TOL = 1e-15
HELICITIES = (-1, 1)
eta = np.array([1.0, -1.0, -1.0, -1.0])


def _as_four_vector(v, name, dtype=complex):
    """Validate and return ``v`` as a finite four-vector.

    Parameters
    ----------
    v : array-like
        Candidate vector in ``[E, px, py, pz]`` order.
    name : str
        Name used in validation error messages.
    dtype : data-type, optional
        Numpy dtype used for conversion. Complex is the default because many
        algebraic objects may carry phases.

    Returns
    -------
    numpy.ndarray
        A shape ``(4,)`` array with finite entries.
    """
    arr = np.asarray(v, dtype=dtype)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be a four-vector with shape (4,).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _as_three_vector(v, name):
    """Validate and return ``v`` as a finite Euclidean three-vector."""
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a three-vector with shape (3,).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _validate_helicity(value, name):
    """Return a checked helicity label.

    The code uses doubled helicities, so the only valid labels are ``+1`` and
    ``-1``. The return value is an ``int`` for convenient use in dictionaries
    and loop indices.
    """
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1.")
    return int(value)


def _validate_lorentz_index(value, name):
    """Return a checked Lorentz index in the range ``0`` through ``3``."""
    if value not in (0, 1, 2, 3):
        raise ValueError(f"{name} must be one of 0, 1, 2, or 3.")
    return int(value)


def _validate_scalar(value, name):
    """Return ``value`` as a finite float."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_nonnegative_scalar(value, name):
    """Return ``value`` as a finite float constrained to be non-negative."""
    value = _validate_scalar(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _validate_positive_scalar(value, name, tol=DEFAULT_TOL):
    """Return ``value`` as a finite float constrained to be positive.

    Values below or equal to ``tol`` are rejected. The tolerance avoids
    accepting numerically zero masses, energies, or denominators as physical
    positive inputs.
    """
    value = _validate_scalar(value, name)
    if value <= tol:
        raise ValueError(f"{name} must be positive.")
    return value


def _check_not_singular(value, name, tol=DEFAULT_TOL):
    """Raise ``ZeroDivisionError`` if ``value`` is numerically singular."""
    if abs(value) <= tol:
        raise ZeroDivisionError(f"{name} is singular for this kinematics.")


def _real_scalar(value, name, tol=1e-10):
    """Return a complex scalar as float when its imaginary part is negligible."""
    value = complex(value)
    scale = max(1.0, abs(value.real))
    if abs(value.imag) > tol * scale:
        raise ValueError(f"{name} has a non-negligible imaginary part.")
    return float(value.real)


def mdot(a, b):
    """Minkowski dot product a.b with metric diag(1,-1,-1,-1)."""
    a = _as_four_vector(a, "a")
    b = _as_four_vector(b, "b")
    return np.dot(a, eta * b)


def cov(v):
    """Lower-index vector v_mu = g_{mu nu} v^nu."""
    return eta * _as_four_vector(v, "v")


def spatial(v):
    """Return the spatial three-vector ``[px, py, pz]`` from a four-vector."""
    return _as_four_vector(v, "v", dtype=float)[1:4]


# ============================================================
# Gamma matrices in Dirac representation
# ============================================================

id2 = np.eye(2, dtype=complex)
zero2 = np.zeros((2, 2), dtype=complex)

sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)

gamma = [
    np.block([[id2, zero2], [zero2, -id2]]),
    np.block([[zero2, sigma1], [-sigma1, zero2]]),
    np.block([[zero2, sigma2], [-sigma2, zero2]]),
    np.block([[zero2, sigma3], [-sigma3, zero2]]),
]


def gammaU(mu):
    """Return the contravariant Dirac gamma matrix ``gamma^mu``."""
    mu = _validate_lorentz_index(mu, "mu")
    return gamma[mu]


def gammaL(mu):
    """Return the covariant Dirac gamma matrix ``gamma_mu``."""
    mu = _validate_lorentz_index(mu, "mu")
    return eta[mu] * gamma[mu]


def slash(v):
    """Slash[v] = gamma^mu v_mu."""
    vc = cov(v)
    out = np.zeros((4, 4), dtype=complex)
    for mu in range(4):
        out += vc[mu] * gammaU(mu)
    return out


def spinor_bar(u):
    """Return ubar = u.conj().T gamma^0."""
    u = np.asarray(u, dtype=complex)
    if u.shape != (4,):
        raise ValueError("u must be a spinor with shape (4,).")
    return np.conjugate(u) @ gammaU(0)


# ============================================================
# Two-component helicity spinors
# sigma.p chi_h = h |p| chi_h
# h = +/- 1
# ============================================================

def chi_helicity(p3, h, patch="auto", tol=DEFAULT_TOL):
    """Return a two-component helicity eigenspinor.

    The spinor satisfies ``sigma.p chi_h = h |p| chi_h`` with ``h = +/-1``.
    The standard spherical-coordinate expression is split into north and south
    coordinate patches to avoid singularities at the poles; ``patch="auto"``
    selects a nonsingular patch from the momentum direction.

    Parameters
    ----------
    p3 : array-like
        Spatial momentum ``[px, py, pz]``.
    h : int
        Doubled helicity label, either ``+1`` or ``-1``.
    patch : {"auto", "north", "south"}
        Coordinate patch used for the spinor representative.
    tol : float
        Zero-momentum and patch-singularity tolerance.
    """
    h = _validate_helicity(h, "h")
    p3 = _as_three_vector(p3, "p3")
    px, py, pz = p3
    pabs = np.linalg.norm(p3)

    if pabs <= tol:
        raise ValueError("Helicity spinor is undefined for zero three-momentum.")

    if not isinstance(patch, str):
        raise ValueError("patch must be 'north', 'south', or 'auto'.")
    patch = patch.lower()

    if patch == "auto":
        patch = "south" if abs(pabs + pz) <= tol else "north"

    if patch == "north":
        den = np.sqrt(2.0 * pabs * (pabs + pz))
        if abs(den) <= tol:
            raise ValueError("North patch singular. Use patch='south'.")

        if h == +1:
            return np.array([pabs + pz, px + 1j * py], dtype=complex) / den
        if h == -1:
            return np.array([-px + 1j * py, pabs + pz], dtype=complex) / den

    elif patch == "south":
        den = np.sqrt(2.0 * pabs * (pabs - pz))
        if abs(den) <= tol:
            raise ValueError("South patch singular. Use patch='north'.")

        if h == +1:
            return np.array([px - 1j * py, pabs - pz], dtype=complex) / den
        if h == -1:
            return np.array([-(pabs - pz), px + 1j * py], dtype=complex) / den

    raise ValueError("h must be +/-1 and patch must be 'north', 'south', or 'auto'.")


# ============================================================
# External spinors
# ============================================================

def electron_spinor(k, h, patch="auto", electron_mass=0.0):
    """Return a Dirac spinor for an external electron.

    Parameters
    ----------
    k : array-like
        Electron four-momentum. The energy must be positive.
    h : int
        Electron doubled-helicity label, ``+1`` or ``-1``.
    patch : str
        Patch selector passed to :func:`chi_helicity`.
    electron_mass : float, optional
        Electron mass. The default ``0.0`` preserves the massless convention.
    """
    h = _validate_helicity(h, "h")
    electron_mass = _validate_nonnegative_scalar(electron_mass, "electron_mass")
    k = _as_four_vector(k, "k", dtype=float)
    E = k[0]
    if E <= DEFAULT_TOL:
        raise ValueError("Electron energy must be positive.")
    chi = chi_helicity(k[1:4], h, patch=patch)
    pabs = np.linalg.norm(k[1:4])
    upper_norm = np.sqrt(E + electron_mass)
    if upper_norm <= DEFAULT_TOL:
        raise ValueError("Electron E + electron_mass must be positive.")
    return np.concatenate([
        upper_norm * chi,
        h * pabs / upper_norm * chi,
    ])


def proton_spinor(p, s, m, patch="auto"):
    """Return a massive Dirac spinor for an external proton.

    Parameters
    ----------
    p : array-like
        Proton four-momentum.
    s : int
        Proton spin/helicity label, ``+1`` or ``-1``.
    m : float
        Proton mass. Must be positive.
    patch : str
        Patch selector passed to :func:`chi_helicity`.
    """
    s = _validate_helicity(s, "s")
    m = _validate_positive_scalar(m, "m")
    p = _as_four_vector(p, "p", dtype=float)
    E = p[0]
    pabs = np.linalg.norm(p[1:4])
    if E + m <= DEFAULT_TOL:
        raise ValueError("Proton E + m must be positive.")
    chi = chi_helicity(p[1:4], s, patch=patch)

    return np.concatenate([np.sqrt(E + m) * chi, s * pabs / np.sqrt(E + m) * chi])


# ============================================================
# Photon helicity polarization
# epsilon^mu(q,lambda) = (0, (e_theta + i lambda e_phi)/sqrt(2))
# ============================================================

def photon_pol(q, lam, tol=DEFAULT_TOL):
    """Return a transverse real-photon helicity polarization vector.

    The convention is ``epsilon^mu(q, lambda) =
    (0, (e_theta + i lambda e_phi) / sqrt(2))`` with ``lambda = +/-1``.
    The returned vector is contravariant and has a zero time component.
    """
    lam = _validate_helicity(lam, "lam")
    q = _as_four_vector(q, "q", dtype=float)
    q3 = q[1:4]
    qabs = np.linalg.norm(q3)

    if qabs <= tol:
        raise ValueError("Photon momentum cannot be zero.")

    theta = np.arccos(np.clip(q3[2] / qabs, -1.0, 1.0))
    phi = np.arctan2(q3[1], q3[0])
    e_theta = np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta),
    ], dtype=complex)
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=complex)
    eps_spatial = (e_theta + 1j * lam * e_phi) / np.sqrt(2.0)
    return np.concatenate([[0.0 + 0.0j], eps_spatial])


def photon_pol_cov_star(epsU):
    """Return the covariant complex-conjugate photon polarization vector."""
    return cov(np.conjugate(epsU))
