from itertools import product

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
    kinematics_cm_from_beam_energy,
    momenta_cm_from_beam_energy,
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


def bh_amplitude_user(
    pIn, pOut, qOut, th, ph, phOut,
    hIn, hOut,
    sIn, sOut,
    lam,
    m, F1, F2,
):
    mom = momenta_user(pIn, pOut, qOut, th, ph, phOut, m)
    return bh_amplitude_core(
        mom["k"], mom["kp"], mom["qout"],
        mom["p"], mom["pp"],
        photon_pol(mom["qout"], lam),
        hIn, hOut,
        sIn, sOut,
        m, F1, F2,
    )


def bh_unpolarized_squared_amplitude_user(
    pIn, pOut, qOut, th, ph, phOut,
    m, F1, F2,
    average_initial=True,
):
    mom = momenta_user(pIn, pOut, qOut, th, ph, phOut, m)
    return bh_unpolarized_squared_amplitude_core(
        mom["k"], mom["kp"], mom["qout"],
        mom["p"], mom["pp"],
        m, F1, F2,
        average_initial=average_initial,
    )


def bh_amplitude_cm_from_beam_energy(
    Eb, Q2, xB, t, phi,
    hIn, hOut,
    sIn, sOut,
    lam,
    m, F1, F2,
):
    mom = momenta_cm_from_beam_energy(Eb, Q2, xB, t, phi, m)
    return bh_amplitude_core(
        mom["k"], mom["kp"], mom["qout"],
        mom["p"], mom["pp"],
        photon_pol(mom["qout"], lam),
        hIn, hOut,
        sIn, sOut,
        m, F1, F2,
    )


def bh_unpolarized_squared_amplitude_cm_from_beam_energy(
    Eb, Q2, xB, t, phi,
    m, F1, F2,
    average_initial=True,
):
    mom = momenta_cm_from_beam_energy(Eb, Q2, xB, t, phi, m)
    return bh_unpolarized_squared_amplitude_core(
        mom["k"], mom["kp"], mom["qout"],
        mom["p"], mom["pp"],
        m, F1, F2,
        average_initial=average_initial,
    )


def bh_amplitude_same_electron_helicity(
    pIn, pOut, qOut, th, ph, phOut,
    h, sIn, sOut, lam,
    m, F1, F2,
):
    return bh_amplitude_user(
        pIn, pOut, qOut, th, ph, phOut,
        h, h, sIn, sOut, lam,
        m, F1, F2,
    )


def main():
    from pathlib import Path

    def ascii_table(headers, rows):
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
        body_lines = [
            "| " + " | ".join(item.rjust(width) for item, width in zip(row, widths)) + " |"
            for row in table_rows
        ]
        return "\n".join([separator, header_line, separator, *body_lines, separator])

    def fmt(value, digits=8):
        return f"{value:.{digits}e}"

    def rel_diff(value, reference):
        return value / reference if reference != 0.0 else float("nan")

    def vector_row(name, vector):
        return [name, *(fmt(component) for component in vector)]

    def analytic_ab_terms(mom):
        pbar = 0.5 * (mom["pp"] + mom["p"])
        delta = mom["pp"] - mom["p"]
        t = mdot(delta, delta)
        k_dot_qout = mdot(mom["k"], mom["qout"])
        kp_dot_qout = mdot(mom["kp"], mom["qout"])
        _check_not_singular(t, "analytic benchmark t")
        _check_not_singular(k_dot_qout, "analytic benchmark k.qout")
        _check_not_singular(kp_dot_qout, "analytic benchmark kp.qout")

        delta_sum = mdot(mom["k"], delta) ** 2 + mdot(mom["kp"], delta) ** 2
        pbar_sum = mdot(mom["k"], pbar) ** 2 + mdot(mom["kp"], pbar) ** 2
        a_bh = (
            -8.0
            / (t * k_dot_qout * kp_dot_qout)
            * (mdot(pbar, pbar) * delta_sum + t * pbar_sum)
        )
        b_bh = -4.0 / (k_dot_qout * kp_dot_qout) * delta_sum

        return {
            "t": _real_scalar(t, "t"),
            "A_BH": _real_scalar(a_bh, "A_BH"),
            "B_BH": _real_scalar(b_bh, "B_BH"),
        }

    def analytic_ab_m2(F1, F2, terms, m):
        t = terms["t"]
        _check_not_singular(t, "analytic benchmark t")
        return (
            terms["A_BH"] * (F1**2 - t * F2**2 / (4.0 * m**2))
            + terms["B_BH"] * (F1 + F2) ** 2
        ) / t

    m = 0.938
    Eb, Q2, xB, t_input, phi = 5.0, 2.0, 0.36, -0.4, 0.7
    form_factors = [
        (1.0, 0.0),
        (0.8, 0.0),
        (1.2, 0.0),
        (1.0, 0.2),
        (1.0, -0.2),
        (0.7, 0.5),
        (0.0, 1.0),
    ]
    log_path = Path("Output") / "BHHelicityAmp.log"

    ref_F1, ref_F2 = 1.0, 0.0
    kin = kinematics_cm_from_beam_energy(Eb, Q2, xB, t_input, phi, m)
    mom = kin["momenta"]
    analytic_terms = analytic_ab_terms(mom)
    energy_residual = (
        mom["k"][0] + mom["p"][0]
        - mom["kp"][0] - mom["pp"][0] - mom["qout"][0]
    )
    momentum_residual = (
        mom["k"][1:4] + mom["p"][1:4]
        - mom["kp"][1:4] - mom["pp"][1:4] - mom["qout"][1:4]
    )
    onshell_values = [
        mdot(mom["k"], mom["k"]),
        mdot(mom["kp"], mom["kp"]),
        mdot(mom["qout"], mom["qout"]),
        mdot(mom["p"], mom["p"]),
        mdot(mom["pp"], mom["pp"]),
    ]

    helicity_rows = []
    ref_photon_pols = {lam: photon_pol(mom["qout"], lam) for lam in HELICITIES}
    for hIn, hOut, sIn, sOut, lam in product(HELICITIES, repeat=5):
        amp = bh_amplitude_core(
            mom["k"], mom["kp"], mom["qout"],
            mom["p"], mom["pp"],
            ref_photon_pols[lam],
            hIn, hOut, sIn, sOut,
            m, ref_F1, ref_F2,
        )
        helicity_rows.append([
            hIn, hOut, sIn, sOut, lam,
            fmt(amp.real), fmt(amp.imag), fmt(abs(amp) ** 2),
        ])

    benchmark_rows = []
    for index, (F1, F2) in enumerate(form_factors, start=1):
        unpolarized_amp2 = bh_unpolarized_squared_amplitude_core(
            mom["k"], mom["kp"], mom["qout"],
            mom["p"], mom["pp"],
            m, F1, F2,
        )
        analytic_amp2 = analytic_ab_m2(F1, F2, analytic_terms, m)
        diff = unpolarized_amp2 - analytic_amp2
        benchmark_rows.append([
            index,
            f"{F1:.6g}",
            f"{F2:.6g}",
            fmt(unpolarized_amp2),
            fmt(analytic_amp2),
            f"{diff:.3e}",
            f"{rel_diff(diff, analytic_amp2):.3e}",
        ])

    helicity_table = ascii_table(
        ["hIn", "hOut", "sIn", "sOut", "lam", "Re M", "Im M", "|M|^2"],
        helicity_rows,
    )
    benchmark_table = ascii_table(
        ["case", "F1", "F2", "unpol |M|^2", "analytic AB", "diff", "rel diff"],
        benchmark_rows,
    )
    momenta_table = ascii_table(
        ["vec", "E", "px", "py", "pz"],
        [
            vector_row("k", mom["k"]),
            vector_row("p", mom["p"]),
            vector_row("kp", mom["kp"]),
            vector_row("pp", mom["pp"]),
            vector_row("qout", mom["qout"]),
            vector_row("q", mom["q"]),
        ],
    )

    lines = [
        "BH helicity-amplitude benchmark",
        "",
        "Kinematics",
        "  Frame: initial electron-proton center of momentum",
        "  Scalar inputs: Eb, Q2, xB, t, phi",
        f"  Eb scalar = {kin['Eb']:.16g}",
        f"  s = {kin['s']:.16g}",
        f"  sqrt(s) = {kin['sqrt_s']:.16g}",
        f"  Q2 = {kin['Q2']:.16g}",
        f"  xB = {kin['xB']:.16g}",
        f"  t = {kin['t']:.16g}",
        f"  phi = {kin['phi']:.16g}",
        f"  m = {m:.16g}",
        "  Derived variables",
        f"  nu = {kin['nu']:.16g}",
        f"  y = Q2/(2 m xB Eb) = {kin['y']:.16g}",
        "",
        "Four-momenta in initial e+p COM frame [E, px, py, pz]",
        momenta_table,
        f"  energy_balance = {energy_residual:.16e}",
        f"  momentum_conservation = {momentum_residual}",
        f"  onshell_check = {onshell_values}",
        "",
        "Analytic benchmark note",
        "  Pbar = (pp + p) / 2",
        "  Delta = pp - p",
        "  t = Delta^2",
        f"  t = {analytic_terms['t']:.16e}",
        f"  A_BH = {analytic_terms['A_BH']:.16e}",
        f"  B_BH = {analytic_terms['B_BH']:.16e}",
        "  Comparison uses the AB-only analytic expression; tilde terms are omitted.",
        "",
        f"Fixed-helicity amplitudes at F1={ref_F1:.6g}, F2={ref_F2:.6g}",
        helicity_table,
        "",
        "Unpolarized analytic benchmark sweep",
        benchmark_table,
        "",
        "Column meanings",
        "  hIn, hOut: incoming and outgoing electron helicities.",
        "  sIn, sOut: incoming and outgoing proton helicities.",
        "  lam: final photon helicity.",
        (
            "  Re M, Im M: one fixed-helicity amplitude for the listed "
            "helicity row. It is a complex amplitude, not a spin sum."
        ),
        (
            "  |M|^2: squared magnitude of one fixed-helicity amplitude; "
            "it is not summed or averaged over spins."
        ),
        "  case: row number for the chosen (F1, F2) pair.",
        "  F1, F2: proton electromagnetic form factors used in Gamma_nu.",
        (
            "  unpol |M|^2: unpolarized squared amplitude, "
            "(1/4) sum_{hIn,sIn,hOut,sOut,lambda} |M|^2. The factor 1/4 "
            "averages over the two incoming electron and proton helicities; "
            "the final electron, proton, and photon helicities are summed."
        ),
        "  analytic AB: AB-only analytic result from the supplied A_BH and B_BH.",
        "  diff: unpol |M|^2 minus analytic AB.",
        "  rel diff: diff divided by analytic AB.",
    ]
    log_text = "\n".join(lines) + "\n"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")
    print(log_text, end="")
    print(f"\nSaved log to {log_path}")


if __name__ == "__main__":
    main()
