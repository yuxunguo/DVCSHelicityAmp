"""Bethe-Heitler helicity amplitudes and analytic benchmark output.

This module evaluates the Bethe-Heitler contribution to exclusive
electroproduction in the conventions provided by :mod:`Algebra` and
:mod:`Kinematics`. It exposes low-level routines that operate directly on
four-momenta, user-frame convenience wrappers, and a script entry point that
writes a comparison against analytic benchmark formulae to
``Output/BHHelicityAmp.log``.

Helicity labels are doubled helicities ``+1`` and ``-1``. The final photon
polarization vector is generated from its four-momentum and contracted as
``epsilon*_mu`` inside the amplitude.
"""

from itertools import product
from textwrap import dedent

import numpy as np

from Algebra import (
    HELICITIES,
    _as_four_vector,
    _check_not_singular,
    _real_scalar,
    _validate_helicity,
    _validate_lorentz_index,
    _validate_positive_scalar,
    cov,
    electron_spinor,
    gammaL,
    gammaU,
    mdot,
    photon_pol,
    photon_pol_cov_star,
    proton_spinor,
    slash,
    spinor_bar,
)
from Kinematics import (
    kinematics_user_from_independent,
    momenta_user,
)


# ============================================================
# Leptonic BH kernel
#
# L^{mu nu} =
#   (2 k'^mu gamma^nu + gamma^mu slash(q') gamma^nu)/(2 k'.q')
# + (2 k^mu gamma^nu - gamma^nu slash(q') gamma^mu)/(-2 k.q')
# ============================================================

def lepton_kernel(mu, nu, k, kp, qout):
    """Return the leptonic Bethe-Heitler kernel ``L^{mu nu}``.

    Parameters
    ----------
    mu, nu : int
        Lorentz indices in ``0..3``. ``mu`` contracts with the real-photon
        polarization and ``nu`` contracts with the proton electromagnetic
        vertex.
    k, kp, qout : array-like
        Incoming electron, outgoing electron, and outgoing real-photon
        four-momenta.

    Returns
    -------
    numpy.ndarray
        A ``4 x 4`` Dirac matrix acting on the electron spinors.
    """
    mu = _validate_lorentz_index(mu, "mu")
    nu = _validate_lorentz_index(nu, "nu")
    k = _as_four_vector(k, "k")
    kp = _as_four_vector(kp, "kp")
    qout = _as_four_vector(qout, "qout")

    den1 = 2.0 * mdot(kp, qout)
    den2 = -2.0 * mdot(k, qout)
    _check_not_singular(den1, "final-state lepton propagator")
    _check_not_singular(den2, "initial-state lepton propagator")

    slash_qout = slash(qout)
    term1 = (2.0 * kp[mu] * gammaU(nu) + gammaU(mu) @ slash_qout @ gammaU(nu)) / den1
    term2 = (2.0 * k[mu] * gammaU(nu) - gammaU(nu) @ slash_qout @ gammaU(mu)) / den2
    return term1 + term2


# ============================================================
# Proton electromagnetic vertex
#
# Gamma_nu =
#   gamma_nu (F1+F2) - (p+p')_nu F2/(2m)
# ============================================================

def proton_vertex_lower(nu, p, pp, m, F1, F2):
    """Return the lower-index proton electromagnetic vertex ``Gamma_nu``.

    The vertex convention is
    ``Gamma_nu = gamma_nu (F1 + F2) - (p + p')_nu F2 / (2m)``.
    """
    nu = _validate_lorentz_index(nu, "nu")
    m = _validate_positive_scalar(m, "m")
    p = _as_four_vector(p, "p")
    pp = _as_four_vector(pp, "pp")
    psum_cov = cov(p + pp)

    return (
        (F1 + F2) * gammaL(nu)
        - psum_cov[nu] * F2 / (2.0 * m) * np.eye(4, dtype=complex)
    )


# ============================================================
# Full amplitude
#
# M = eps^*_mu ubar_e(k',h') L^{mu nu} u_e(k,h)
#     * 1/t *
#     ubar_p(p',s') Gamma_nu u_p(p,s)
# ============================================================

def bh_amplitude_core(
    k, kp, qout,
    p, pp,
    epsU,
    hIn, hOut,
    sIn, sOut,
    m, F1, F2,
):
    """Evaluate a fixed-helicity Bethe-Heitler amplitude.

    This is the lowest-level public amplitude routine. All momenta are supplied
    directly, and the photon polarization vector ``epsU`` is already chosen by
    the caller. The returned complex amplitude is

    ``eps*_mu ubar(k',hOut) L^{mu nu} u(k,hIn) / t
    * ubar(p',sOut) Gamma_nu u(p,sIn)``.

    Parameters are validated for four-vector shape, finite entries, positive
    proton mass, and helicity labels ``+/-1``. Singular propagator or
    momentum-transfer denominators raise ``ZeroDivisionError``.
    """
    hIn = _validate_helicity(hIn, "hIn")
    hOut = _validate_helicity(hOut, "hOut")
    sIn = _validate_helicity(sIn, "sIn")
    sOut = _validate_helicity(sOut, "sOut")
    m = _validate_positive_scalar(m, "m")
    k = _as_four_vector(k, "k", dtype=float)
    kp = _as_four_vector(kp, "kp", dtype=float)
    qout = _as_four_vector(qout, "qout", dtype=float)
    p = _as_four_vector(p, "p", dtype=float)
    pp = _as_four_vector(pp, "pp", dtype=float)
    epsU = _as_four_vector(epsU, "epsU")

    ue_in = electron_spinor(k, hIn)
    ue_out = electron_spinor(kp, hOut)
    up_in = proton_spinor(p, sIn, m)
    up_out = proton_spinor(pp, sOut, m)

    ebar = spinor_bar(ue_out)
    pbar = spinor_bar(up_out)
    eps_cov_star = photon_pol_cov_star(epsU)

    t = mdot(pp - p, pp - p)
    _check_not_singular(t, "momentum-transfer denominator t")

    had_by_nu = [
        pbar @ proton_vertex_lower(nu, p, pp, m, F1, F2) @ up_in
        for nu in range(4)
    ]
    amp = 0.0 + 0.0j
    for mu in range(4):
        for nu, had in enumerate(had_by_nu):
            lep = ebar @ lepton_kernel(mu, nu, k, kp, qout) @ ue_in
            amp += eps_cov_star[mu] * lep * had
    return amp / t


def bh_unpolarized_squared_amplitude_core(
    k, kp, qout,
    p, pp,
    m, F1, F2,
    average_initial=True,
):
    """
    Return the unpolarized Bethe-Heitler squared amplitude.

    This sums over final electron, proton, and photon helicities, and averages
    over the two incoming electron and proton helicities by default.
    """
    m = _validate_positive_scalar(m, "m")
    k = _as_four_vector(k, "k", dtype=float)
    kp = _as_four_vector(kp, "kp", dtype=float)
    qout = _as_four_vector(qout, "qout", dtype=float)
    p = _as_four_vector(p, "p", dtype=float)
    pp = _as_four_vector(pp, "pp", dtype=float)

    photon_pols = {lam: photon_pol(qout, lam) for lam in HELICITIES}
    total = 0.0
    for hIn, sIn, hOut, sOut, lam in product(HELICITIES, repeat=5):
        amp = bh_amplitude_core(
            k, kp, qout,
            p, pp,
            photon_pols[lam],
            hIn, hOut,
            sIn, sOut,
            m, F1, F2,
        )
        total += abs(amp) ** 2

    return float(total / 4.0 if average_initial else total)


def bh_amplitude_table(
    momenta,
    m,
    F1,
    F2,
    initial_states=None,
    outgoing_states=None,
):
    """Return the full Bethe-Heitler helicity-amplitude table.

    The first axis spans incoming ``(hIn, sIn)`` spin labels and the second
    axis spans outgoing ``(hOut, sOut, lambda)`` labels.  By default the
    labels are ordered lexicographically from ``HELICITIES=(-1, +1)``.
    """
    if initial_states is None:
        initial_states = tuple(product(HELICITIES, repeat=2))
    if outgoing_states is None:
        outgoing_states = tuple(product(HELICITIES, repeat=3))

    k = _as_four_vector(momenta["k"], "k", dtype=float)
    kp = _as_four_vector(momenta["kp"], "kp", dtype=float)
    qout = _as_four_vector(momenta["qout"], "qout", dtype=float)
    p = _as_four_vector(momenta["p"], "p", dtype=float)
    pp = _as_four_vector(momenta["pp"], "pp", dtype=float)
    photon_pols = {lam: photon_pol(qout, lam) for lam in HELICITIES}
    amplitudes = np.zeros((len(initial_states), len(outgoing_states)), dtype=complex)

    for in_index, (h_in, s_in) in enumerate(initial_states):
        for out_index, (h_out, s_out, lam) in enumerate(outgoing_states):
            amplitudes[in_index, out_index] = bh_amplitude_core(
                k,
                kp,
                qout,
                p,
                pp,
                photon_pols[lam],
                h_in,
                h_out,
                s_in,
                s_out,
                m,
                F1,
                F2,
            )
    return amplitudes


def bh_amplitude_user(
    pIn, pOut, qOut, theta_in, phi_in, phiOut,
    hIn, hOut,
    sIn, sOut,
    lam,
    m, F1, F2,
):
    """Evaluate a fixed-helicity amplitude from user-frame variables.

    The variables ``pIn``, ``pOut``, ``qOut``, ``theta_in``, ``phi_in``, and
    ``phiOut`` are converted with :func:`Kinematics.momenta_user`; photon
    polarization is then built from the resulting ``qout`` momentum.
    """
    mom = momenta_user(pIn, pOut, qOut, theta_in, phi_in, phiOut, m)
    return bh_amplitude_core(
        mom["k"], mom["kp"], mom["qout"],
        mom["p"], mom["pp"],
        photon_pol(mom["qout"], lam),
        hIn, hOut,
        sIn, sOut,
        m, F1, F2,
    )


def bh_unpolarized_squared_amplitude_user(
    pIn, pOut, qOut, theta_in, phi_in, phiOut,
    m, F1, F2,
    average_initial=True,
):
    """Return the unpolarized squared amplitude from user-frame variables."""
    mom = momenta_user(pIn, pOut, qOut, theta_in, phi_in, phiOut, m)
    return bh_unpolarized_squared_amplitude_core(
        mom["k"], mom["kp"], mom["qout"],
        mom["p"], mom["pp"],
        m, F1, F2,
        average_initial=average_initial,
    )


def bh_amplitude_same_electron_helicity(
    pIn, pOut, qOut, theta_in, phi_in, phiOut,
    h, sIn, sOut, lam,
    m, F1, F2,
):
    """Evaluate a user-frame amplitude with ``hIn == hOut == h``."""
    return bh_amplitude_user(
        pIn, pOut, qOut, theta_in, phi_in, phiOut,
        h, h, sIn, sOut, lam,
        m, F1, F2,
    )


def main():
    """Run the analytic benchmark sweep and write ``Output/BHHelicityAmp.log``.

    The benchmark constructs several independent user-frame kinematic points
    and compares numerical spinor sums with analytic Bethe-Heitler expressions
    for multiple ``(F1, F2)`` form-factor choices.
    """
    from pathlib import Path

    def ascii_table(headers, rows, group_by=None):
        table_rows = [[str(item) for item in row] for row in rows]
        widths = [
            max(len(str(header)), *(len(row[index]) for row in table_rows))
            for index, header in enumerate(headers)
        ]
        separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
        header_line = (
            "| "
            + " | ".join(str(header).ljust(width) for header, width in zip(headers, widths))
            + " |"
        )
        lines = [separator, header_line, separator]
        previous_group = None
        for row in table_rows:
            if group_by is not None:
                group = row[group_by]
                if previous_group is not None and group != previous_group:
                    lines.append(separator)
                previous_group = group
            lines.append(
                "| " + " | ".join(item.rjust(width) for item, width in zip(row, widths)) + " |"
            )
        lines.append(separator)
        return "\n".join(lines)

    def fmt(value, digits=8):
        return f"{value:.{digits}e}"

    def rel_diff(value, reference):
        return value / reference if reference != 0.0 else float("nan")

    def vector_row(name, vector):
        phi = fmt(vector_azimuth(vector)) if name in ("k", "p", "qout") else "-"
        return [name, *(fmt(component) for component in vector), phi]

    def initial_proton_spin_vector(p, m, spin):
        p = _as_four_vector(p, "p", dtype=float)
        spin = _validate_helicity(spin, "spin")
        p_abs = np.linalg.norm(p[1:4])
        if p_abs <= 1e-12:
            raise ValueError("Longitudinal spin vector is undefined for p=0.")
        return spin * np.concatenate([[p_abs / m], p[0] * p[1:4] / (m * p_abs)])

    def fixed_initial_m2(mom, hIn, sIn, m, F1, F2):
        photon_pols = {lam: photon_pol(mom["qout"], lam) for lam in HELICITIES}
        total = 0.0
        for hOut, sOut, lam in product(HELICITIES, repeat=3):
            amp = bh_amplitude_core(
                mom["k"], mom["kp"], mom["qout"],
                mom["p"], mom["pp"],
                photon_pols[lam],
                hIn, hOut,
                sIn, sOut,
                m, F1, F2,
            )
            total += abs(amp) ** 2
        return float(total)

    def comparison_row(case_id, index, F1, F2, numerical, analytic):
        diff = numerical - analytic
        return [
            case_id, index, f"{F1:.6g}", f"{F2:.6g}",
            fmt(numerical), fmt(analytic),
            f"{diff:.3e}", f"{rel_diff(diff, analytic):.3e}",
        ]

    def normalize_angle(angle):
        return float(angle % (2.0 * np.pi))

    def vector_azimuth(vector):
        vector = _as_four_vector(vector, "vector", dtype=float)
        return normalize_angle(np.arctan2(vector[2], vector[1]))

    def analytic_bh_terms(mom, m, h=0, S=None):
        pbar = 0.5 * (mom["pp"] + mom["p"])
        delta = mom["pp"] - mom["p"]
        t = mdot(delta, delta)
        k_dot_qout = mdot(mom["k"], mom["qout"])
        kp_dot_qout = mdot(mom["kp"], mom["qout"])
        _check_not_singular(t, "analytic benchmark t")
        _check_not_singular(k_dot_qout, "analytic benchmark k.qout")
        _check_not_singular(kp_dot_qout, "analytic benchmark kp.qout")

        pbar2 = mdot(pbar, pbar)
        k_delta = mdot(mom["k"], delta)
        kp_delta = mdot(mom["kp"], delta)
        k_pbar = mdot(mom["k"], pbar)
        kp_pbar = mdot(mom["kp"], pbar)
        delta_sum = k_delta**2 + kp_delta**2
        pbar_sum = k_pbar**2 + kp_pbar**2
        a_bh = (
            -8.0
            / (t * k_dot_qout * kp_dot_qout)
            * (pbar2 * delta_sum + t * pbar_sum)
        )
        b_bh = -4.0 / (k_dot_qout * kp_dot_qout) * delta_sum
        at_bh = 0.0
        bt_bh = 0.0

        if S is not None:
            h = float(h)
            if not np.isfinite(h):
                raise ValueError("h must be finite.")
            S = _as_four_vector(S, "S", dtype=float)
            k_S = mdot(mom["k"], S)
            kp_S = mdot(mom["kp"], S)
            S_pbar = mdot(S, pbar)
            S_delta = mdot(S, delta)

            at_bh = (
                16.0 * h
                / (m * t * k_dot_qout * kp_dot_qout)
                * (
                    t * pbar2 * (kp_delta * kp_S - k_delta * k_S)
                    + t * S_pbar * (k_delta * k_pbar - kp_delta * kp_pbar)
                    + pbar2 * S_delta * (k_delta**2 - kp_delta**2)
                )
            )
            bt_bh = (
                16.0 * h * m
                / (t * k_dot_qout * kp_dot_qout)
                * (
                    S_delta * (kp_delta**2 - k_delta**2)
                    + t * (k_delta * k_S - kp_delta * kp_S)
                )
            )

        return {
            "t": _real_scalar(t, "t"),
            "A_BH": _real_scalar(a_bh, "A_BH"),
            "B_BH": _real_scalar(b_bh, "B_BH"),
            "At_BH": _real_scalar(at_bh, "At_BH"),
            "Bt_BH": _real_scalar(bt_bh, "Bt_BH"),
        }

    def analytic_bh_m2(F1, F2, terms, m, include_tilde):
        t = terms["t"]
        _check_not_singular(t, "analytic benchmark t")
        value = (
            terms["A_BH"] * (F1**2 - t * F2**2 / (4.0 * m**2))
            + terms["B_BH"] * (F1 + F2) ** 2
        )
        if include_tilde:
            value += (
                terms["At_BH"] * (F1 * F2 + F2**2)
                + terms["Bt_BH"] * (F1 + F2) ** 2
            )
        return value / t

    input_keys = ("case", "s", "theta_in", "phi_in", "qOut", "phiOut", "m")
    kinematic_inputs = [
        dict(zip(input_keys, row))
        for row in (
            ("K1", 10.25844, 1.10, 0.20, 0.45, 2.40, 0.938),
            ("K2", 14.01544, 1.45, 0.70, 0.70, 3.10, 0.938),
            ("K3", 19.64894, 1.90, 1.10, 0.95, 3.70, 0.938),
        )
    ]
    form_factors = [(0.5, 0.0), (0.8, 0.0), (1.0, 0.2)]
    form_factors += [(1.0, -0.2), (0.7, 0.5), (0.0, 1.0)]
    log_path = Path("Output") / "BHHelicityAmp.log"

    ref_F1, ref_F2 = 1.0, 0.0
    pol_h_in, pol_s = +1, +1
    pol_h_analytic = 0.5 * pol_h_in

    def build_case(inputs):
        kin = kinematics_user_from_independent(
            inputs["s"],
            inputs["theta_in"],
            inputs["phi_in"],
            inputs["qOut"],
            inputs["phiOut"],
            inputs["m"],
            label=inputs["case"],
        )
        mom = kin["momenta"]
        spin_vector = initial_proton_spin_vector(mom["p"], kin["m"], pol_s)
        return {
            "id": inputs["case"],
            "input": inputs,
            "kin": kin,
            "mom": mom,
            "S": spin_vector,
        }

    cases = [build_case(inputs) for inputs in kinematic_inputs]

    def values_from(mapping, keys):
        return [fmt(mapping[key]) for key in keys]

    def row_from(case, source, keys):
        return [case["id"], *values_from(case[source], keys)]

    def solved_row(case):
        return [
            case["id"],
            *values_from(case["kin"], ("pIn", "pOut", "qOut", "theta_in", "phi_in", "phiOut")),
            fmt(case["kin"]["energy_residual"]),
        ]

    def kinematic_checks(case):
        kin = case["kin"]
        mom = case["mom"]
        m = kin["m"]
        conservation = mom["k"] + mom["p"] - mom["kp"] - mom["pp"] - mom["qout"]

        def row(name, target, value, unit):
            return [case["id"], name, fmt(target), fmt(value), f"{value - target:.3e}", unit]

        rows = [
            row("s", case["input"]["s"], kin["s"], "GeV^2"),
            row("energy residual", 0.0, kin["energy_residual"], "GeV"),
            row("energy balance", 0.0, conservation[0], "GeV"),
            row("|3-mom balance|", 0.0, np.linalg.norm(conservation[1:4]), "GeV"),
            row("k^2", 0.0, _real_scalar(mdot(mom["k"], mom["k"]), "k^2"), "GeV^2"),
            row("kp^2", 0.0, _real_scalar(mdot(mom["kp"], mom["kp"]), "kp^2"), "GeV^2"),
            row("qout^2", 0.0, _real_scalar(mdot(mom["qout"], mom["qout"]), "qout^2"), "GeV^2"),
            row("p^2", m**2, _real_scalar(mdot(mom["p"], mom["p"]), "p^2"), "GeV^2"),
            row("pp^2", m**2, _real_scalar(mdot(mom["pp"], mom["pp"]), "pp^2"), "GeV^2"),
        ]
        return rows

    def spin_row(case):
        S = case["S"]
        return [
            case["id"], *(fmt(component) for component in S),
            fmt(_real_scalar(mdot(S, case["mom"]["p"]), "S.p")),
            fmt(_real_scalar(mdot(S, S), "S^2")),
        ]

    independent_rows = [
        row_from(case, "input", ("s", "theta_in", "phi_in", "qOut", "phiOut", "m"))
        for case in cases
    ]
    derived_rows = [
        row_from(case, "kin", ("sqrt_s", "Q2", "xB", "t", "W2", "y"))
        for case in cases
    ]
    solved_rows = [solved_row(case) for case in cases]
    momentum_rows = [
        [case["id"], *vector_row(name, case["mom"][name])]
        for case in cases
        for name in ("k", "p", "kp", "pp", "qout", "q")
    ]
    check_rows = [
        row
        for case in cases
        for row in kinematic_checks(case)
    ]
    spin_rows = [spin_row(case) for case in cases]

    ref_case = cases[0]
    ref_mom = ref_case["mom"]
    helicity_rows = []
    ref_photon_pols = {lam: photon_pol(ref_mom["qout"], lam) for lam in HELICITIES}
    for hIn, hOut, sIn, sOut, lam in product(HELICITIES, repeat=5):
        amp = bh_amplitude_core(
            ref_mom["k"], ref_mom["kp"], ref_mom["qout"],
            ref_mom["p"], ref_mom["pp"],
            ref_photon_pols[lam],
            hIn, hOut, sIn, sOut,
            ref_case["kin"]["m"], ref_F1, ref_F2,
        )
        helicity_rows.append([
            hIn, hOut, sIn, sOut, lam,
            fmt(amp.real), fmt(amp.imag), fmt(abs(amp) ** 2),
        ])

    benchmark_rows = []
    polarized_rows = []
    for case in cases:
        terms = analytic_bh_terms(case["mom"], case["kin"]["m"])
        pol_terms = analytic_bh_terms(
            case["mom"], case["kin"]["m"],
            h=pol_h_analytic,
            S=case["S"],
        )
        for index, (F1, F2) in enumerate(form_factors, start=1):
            unpolarized_amp2 = bh_unpolarized_squared_amplitude_core(
                case["mom"]["k"], case["mom"]["kp"], case["mom"]["qout"],
                case["mom"]["p"], case["mom"]["pp"],
                case["kin"]["m"], F1, F2,
            )
            analytic_amp2 = analytic_bh_m2(
                F1, F2, terms, case["kin"]["m"],
                include_tilde=False,
            )
            benchmark_rows.append(
                comparison_row(case["id"], index, F1, F2, unpolarized_amp2, analytic_amp2)
            )

            fixed_initial_amp2 = fixed_initial_m2(
                case["mom"],
                pol_h_in, pol_s,
                case["kin"]["m"],
                F1, F2,
            )
            polarized_analytic_amp2 = analytic_bh_m2(
                F1, F2, pol_terms, case["kin"]["m"],
                include_tilde=True,
            )
            polarized_rows.append(
                comparison_row(
                    case["id"],
                    index,
                    F1,
                    F2,
                    fixed_initial_amp2,
                    polarized_analytic_amp2,
                )
            )

    table_specs = {
        "independent": (
            ["case", "s [GeV^2]", "theta_in [rad]", "phi_in [rad]",
             "qOut [GeV]", "phiOut [rad]", "m [GeV]"],
            independent_rows,
        ),
        "derived": (
            ["case", "sqrt(s) [GeV]", "Q2 [GeV^2]", "xB", "t [GeV^2]",
             "W2 [GeV^2]", "y"],
            derived_rows,
        ),
        "solved": (
            ["case", "pIn [GeV]", "pOut [GeV]", "qOut [GeV]", "theta_in [rad]",
             "phi_in [rad]", "phiOut [rad]", "energy residual [GeV]"],
            solved_rows,
        ),
        "momenta": (
            ["case", "vec", "E [GeV]", "px [GeV]", "py [GeV]", "pz [GeV]",
             "phi_xy [rad]"],
            momentum_rows,
        ),
        "checks": (
            ["case", "check", "target", "from 4-mom", "diff", "unit"],
            check_rows,
        ),
        "spin": (["case", "S0", "Sx", "Sy", "Sz", "S.p", "S^2"], spin_rows),
        "benchmark": (
            ["kin", "ff", "F1", "F2", "unpol |M|^2", "analytic AB", "diff",
             "rel diff"],
            benchmark_rows,
        ),
        "polarized": (
            ["kin", "ff", "F1", "F2", "fixed h,S |M|^2", "analytic full",
             "diff", "rel diff"],
            polarized_rows,
        ),
    }
    tables = {
        name: ascii_table(headers, rows, group_by=0)
        for name, (headers, rows) in table_specs.items()
    }
    tables["helicity"] = ascii_table(
        ["hIn", "hOut", "sIn", "sOut", "lam", "Re M", "Im M", "|M|^2"],
        helicity_rows,
    )

    intro = dedent(f"""\
        BH helicity-amplitude benchmark

        Kinematics and variables
          Each K row is one independent user-frame COM input point.
          s: total incoming e+p invariant mass squared in GeV^2.
          theta_in, phi_in: incoming proton direction in the user frame.
          qOut: outgoing real-photon energy/momentum magnitude in GeV.
          phiOut: outgoing real-photon azimuth in the user frame.
          pOut is solved from energy conservation for each row.
          Q2, xB, t, W2, and y are derived diagnostics.
          m: proton mass in GeV.
          Independent inputs are converted to pIn and pOut, then built with
          momenta_user.
          Four-momenta are reported in the user-kinematics COM frame as
          [E, px, py, pz] in GeV, with pp=(E,0,pOut,0) and
          qout=qOut(1,cos(phiOut),sin(phiOut),0).
          phi_xy: atan2(py, px) in radians, shown for k, p, and qout.
          k, kp: incoming and outgoing electron momenta.
          p, pp: incoming and outgoing proton momenta.
          qout: outgoing real photon momentum.
          Pbar = (pp + p) / 2, Delta = pp - p, and t = Delta^2.
          S: initial-proton longitudinal spin four-vector, with S.p=0 and S^2=-1.
          F1, F2: proton electromagnetic form factors used in Gamma_nu.
          ff: form-factor row number for the chosen (F1, F2) pair.
          hIn, hOut: incoming and outgoing electron spinor helicity labels.
          sIn, sOut: incoming and outgoing proton helicity labels.
          lam: final photon helicity.
          unpol |M|^2: (1/4) sum over hIn,sIn,hOut,sOut,lambda.
          The factor 1/4 averages the incoming electron and proton spins.
          fixed h,S |M|^2: spinor hIn=+1, sIn=+1 initial state, summed
          over final electron/proton/photon helicities.
          analytic AB: analytic result using only A_BH and B_BH.
          analytic full: analytic result including A_BH, B_BH, At_BH, and Bt_BH.
          diff: numerical spinor result minus the corresponding analytic result.
          rel diff: diff divided by the corresponding analytic result.
          Re M and Im M in the final example table are one fixed-helicity
          complex amplitude, not a spin sum.
        """).strip()
    analytic_note = dedent(f"""\
        Analytic benchmark note
          Pbar = (pp + p) / 2
          Delta = pp - p
          t = Delta^2
          Unpolarized comparison uses only A_BH and B_BH.
          Polarized comparison adds At_BH and Bt_BH for spinor hIn=+1,
          sIn=+1, and the longitudinal initial-proton spin vector S shown below.
          Code electron spinor labels are hIn=+/-1; the tilde formula uses
          physical helicity h=hIn/2, so this benchmark uses h={pol_h_analytic:.1f}.
        """).strip()

    lines = [
        intro,
        "",
        "Independent user-frame inputs",
        tables["independent"],
        "",
        "Derived invariant diagnostics",
        tables["derived"],
        "",
        "Solved user-kinematics variables",
        tables["solved"],
        "",
        "Four-momenta from momenta_user",
        tables["momenta"],
        "",
        "Four-momentum checks",
        tables["checks"],
        "",
        analytic_note,
        "",
        "Initial-proton polarization vector used in polarized benchmark",
        tables["spin"],
        "",
        "Unpolarized analytic benchmark sweep over kinematics and form factors",
        tables["benchmark"],
        "",
        "Polarized analytic benchmark sweep over kinematics and form factors",
        tables["polarized"],
        "",
        (
            f"Example fixed-helicity amplitudes for {ref_case['id']} "
            f"at F1={ref_F1:.6g}, F2={ref_F2:.6g}"
        ),
        tables["helicity"],
    ]
    log_text = "\n".join(lines) + "\n"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")
    print(log_text, end="")
    print(f"\nSaved log to {log_path}")


if __name__ == "__main__":
    main()
