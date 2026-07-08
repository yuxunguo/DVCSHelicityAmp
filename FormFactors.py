"""Electromagnetic proton form factors used by the DVCS scans.

The YAHL 2018 folder stores proton Sachs form factors as ratios to the dipole
form factor. This module reads that lookup table and converts the Sachs
``G_E``/``G_M`` values to Dirac/Pauli ``F1``/``F2`` at each kinematic ``t``.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np


YAHL_MODEL_NAME = "YAHL 2018 proton lookup"
YAHL_PROTON_LOOKUP_PATH = Path(__file__).resolve().parent / "YAHL 2018" / "proton_lookup.dat"
PROTON_MAGNETIC_MOMENT = 2.79284356


def dipole_form_factor(Q2):
    """Return the standard dipole form factor at spacelike ``Q2`` in GeV^2."""
    return 1.0 / (1.0 + Q2 / 0.71) ** 2


@lru_cache(maxsize=1)
def _load_yahl_proton_table():
    """Load YAHL proton central values from the repository lookup table."""
    if not YAHL_PROTON_LOOKUP_PATH.exists():
        raise FileNotFoundError(f"Missing YAHL proton lookup table: {YAHL_PROTON_LOOKUP_PATH}")

    table = np.loadtxt(YAHL_PROTON_LOOKUP_PATH, comments="#")
    if table.ndim != 2 or table.shape[1] < 5:
        raise ValueError(f"Unexpected YAHL proton table shape: {table.shape}")

    Q2 = table[:, 0]
    GEp_over_GD = table[:, 1]
    GMp_over_mu_GD = table[:, 4]
    if np.any(np.diff(Q2) <= 0.0):
        raise ValueError("YAHL proton lookup Q2 grid must be strictly increasing.")

    return Q2, GEp_over_GD, GMp_over_mu_GD


def yahl_sachs_form_factors(Q2):
    """Return proton ``(GE, GM)`` central values from the YAHL table."""
    Q2 = float(Q2)
    if Q2 < -1.0e-12:
        raise ValueError(f"YAHL form factors require spacelike Q2 >= 0, got {Q2:.16e}.")
    Q2 = max(0.0, Q2)

    table_Q2, table_GEp_over_GD, table_GMp_over_mu_GD = _load_yahl_proton_table()
    if Q2 > table_Q2[-1]:
        raise ValueError(
            f"YAHL proton lookup covers Q2 <= {table_Q2[-1]:.6g} GeV^2, got {Q2:.6g}."
        )

    Q2_grid = np.concatenate(([0.0], table_Q2))
    GEp_over_GD_grid = np.concatenate(([1.0], table_GEp_over_GD))
    GMp_over_mu_GD_grid = np.concatenate(([1.0], table_GMp_over_mu_GD))

    GEp_over_GD = np.interp(Q2, Q2_grid, GEp_over_GD_grid)
    GMp_over_mu_GD = np.interp(Q2, Q2_grid, GMp_over_mu_GD_grid)
    GD = dipole_form_factor(Q2)
    GE = GEp_over_GD * GD
    GM = GMp_over_mu_GD * PROTON_MAGNETIC_MOMENT * GD
    return float(GE), float(GM)


def dirac_pauli_from_sachs(Q2, GE, GM, m):
    """Convert Sachs ``GE``/``GM`` to Dirac/Pauli ``F1``/``F2``."""
    tau = float(Q2) / (4.0 * float(m) ** 2)
    denominator = 1.0 + tau
    F1 = (float(GE) + tau * float(GM)) / denominator
    F2 = (float(GM) - float(GE)) / denominator
    return float(F1), float(F2)


def yahl_dirac_pauli_from_t(t, m):
    """Return proton ``(F1, F2)`` at invariant momentum transfer ``t``."""
    Q2_transfer = -float(t)
    if Q2_transfer < 0.0 and Q2_transfer > -1.0e-12:
        Q2_transfer = 0.0
    GE, GM = yahl_sachs_form_factors(Q2_transfer)
    return dirac_pauli_from_sachs(Q2_transfer, GE, GM, m)
