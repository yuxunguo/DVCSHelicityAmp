import numpy as np

# ============================================================
# Conventions
# g^{mu nu} = diag(1,-1,-1,-1)
# Four-vectors are numpy arrays [E, px, py, pz]
# h, hp, s, sp, lam = +/- 1
# Electron is massless. Proton has mass m.
# ============================================================

DEFAULT_TOL = 1e-12
HELICITIES = (-1, 1)
eta = np.array([1.0, -1.0, -1.0, -1.0])


def _as_four_vector(v, name, dtype=complex):
    arr = np.asarray(v, dtype=dtype)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be a four-vector with shape (4,).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _as_three_vector(v, name):
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a three-vector with shape (3,).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _validate_helicity(value, name):
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1.")
    return int(value)


def _validate_lorentz_index(value, name):
    if value not in (0, 1, 2, 3):
        raise ValueError(f"{name} must be one of 0, 1, 2, or 3.")
    return int(value)


def _validate_scalar(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_nonnegative_scalar(value, name):
    value = _validate_scalar(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _validate_positive_scalar(value, name, tol=DEFAULT_TOL):
    value = _validate_scalar(value, name)
    if value <= tol:
        raise ValueError(f"{name} must be positive.")
    return value


def _check_not_singular(value, name, tol=DEFAULT_TOL):
    if abs(value) <= tol:
        raise ZeroDivisionError(f"{name} is singular for this kinematics.")


def _real_scalar(value, name, tol=1e-10):
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
    mu = _validate_lorentz_index(mu, "mu")
    return gamma[mu]


def gammaL(mu):
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

def electron_spinor(k, h, patch="auto"):
    h = _validate_helicity(h, "h")
    k = _as_four_vector(k, "k", dtype=float)
    E = k[0]
    if E <= DEFAULT_TOL:
        raise ValueError("Electron energy must be positive.")
    chi = chi_helicity(k[1:4], h, patch=patch)

    return np.concatenate([np.sqrt(E) * chi, h * np.sqrt(E) * chi])


def proton_spinor(p, s, m, patch="auto"):
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
    return cov(np.conjugate(epsU))
