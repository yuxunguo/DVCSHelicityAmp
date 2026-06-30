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
    kinematics_user_from_scalar_inputs,
    momenta_cm_from_beam_energy,
    momenta_user,
)

BENCHMARK_AZIMUTH_INPUT = "phi_hadron"  # Use "phi_hadron" or "phi_out".


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

    azimuth_input = BENCHMARK_AZIMUTH_INPUT
    if azimuth_input not in ("phi_hadron", "phi_out"):
        raise ValueError("BENCHMARK_AZIMUTH_INPUT must be 'phi_hadron' or 'phi_out'.")

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

    def angular_diff(value, reference):
        return float((value - reference + np.pi) % (2.0 * np.pi) - np.pi)

    def vector_azimuth(vector):
        vector = _as_four_vector(vector, "vector", dtype=float)
        return normalize_angle(np.arctan2(vector[2], vector[1]))

    def boost_to_rest_frame(v, beta):
        v = _as_four_vector(v, "v", dtype=float)
        beta = np.asarray(beta, dtype=float)
        beta2 = float(np.dot(beta, beta))
        if beta2 <= 1e-24:
            return np.array(v, dtype=float, copy=True)
        if beta2 >= 1.0:
            raise ValueError("Boost velocity must be subluminal.")

        gamma = 1.0 / np.sqrt(1.0 - beta2)
        beta_dot_v = float(np.dot(beta, v[1:4]))
        spatial = v[1:4] + (
            (gamma - 1.0) * beta_dot_v / beta2 - gamma * v[0]
        ) * beta
        return np.concatenate([[gamma * (v[0] - beta_dot_v)], spatial])

    def target_rest_momenta(mom):
        beta = mom["p"][1:4] / mom["p"][0]
        return {name: boost_to_rest_frame(vector, beta) for name, vector in mom.items()}

    def reconstruct_phi(mom):
        q_vec = mom["q"][1:4]
        qhat = q_vec / np.linalg.norm(q_vec)
        lepton_normal = np.cross(mom["k"][1:4], mom["kp"][1:4])
        lepton_normal /= np.linalg.norm(lepton_normal)
        ex = np.cross(lepton_normal, qhat)
        ex /= np.linalg.norm(ex)
        pp_transverse = mom["pp"][1:4] - np.dot(mom["pp"][1:4], qhat) * qhat
        return normalize_angle(
            np.arctan2(np.dot(pp_transverse, lepton_normal), np.dot(pp_transverse, ex))
        )

    def scalar_checks(case_id, kin):
        mom = kin["momenta"]
        m = kin["m"]
        q = mom["k"] - mom["kp"]
        rest_mom = target_rest_momenta(mom)
        s_from_p4 = _real_scalar(mdot(mom["k"] + mom["p"], mom["k"] + mom["p"]), "s")
        eb_from_p4 = (s_from_p4 - m**2) / (2.0 * m)
        q2_from_p4 = -_real_scalar(mdot(q, q), "Q2")
        xB_from_p4 = q2_from_p4 / (2.0 * _real_scalar(mdot(mom["p"], q), "p.q"))
        t_from_p4 = _real_scalar(mdot(mom["pp"] - mom["p"], mom["pp"] - mom["p"]), "t")
        phi_hadron_from_p4 = reconstruct_phi(rest_mom)
        phi_out_from_p4 = vector_azimuth(mom["qout"])
        conservation = mom["k"] + mom["p"] - mom["kp"] - mom["pp"] - mom["qout"]

        def mass_shell(name, target):
            value = _real_scalar(mdot(mom[name], mom[name]), f"{name}^2")
            return (f"{name}^2", target, value, "GeV^2", False)

        checks = [
            ("Eb", kin["Eb"], eb_from_p4, "GeV", False),
            ("Q2", kin["Q2"], q2_from_p4, "GeV^2", False),
            ("xB", kin["xB"], xB_from_p4, "1", False),
            ("t", kin["t"], t_from_p4, "GeV^2", False),
            ("phi_hadron", kin["phi_hadron"], phi_hadron_from_p4, "rad", True),
            ("phi_out", kin["phi_out"], phi_out_from_p4, "rad", True),
            mass_shell("k", 0.0),
            mass_shell("kp", 0.0),
            mass_shell("qout", 0.0),
            mass_shell("p", m**2),
            mass_shell("pp", m**2),
            ("energy balance", 0.0, conservation[0], "GeV", False),
            ("|3-mom balance|", 0.0, np.linalg.norm(conservation[1:4]), "GeV", False),
            ("user backend rebuild", 0.0, kin["user_rebuild_residual"], "GeV", False),
        ]
        rows = []
        for name, target, value, unit, is_angle in checks:
            diff = angular_diff(value, target) if is_angle else value - target
            rows.append([case_id, name, fmt(target), fmt(value), f"{diff:.3e}", unit])
        return rows

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

    input_keys = ("case", "Eb", "Q2", "xB", "t", "phi", "m")
    kinematic_inputs = [
        dict(zip(input_keys, row))
        for row in (
            ("K1", 5.0, 2.0, 0.36, -0.4, 0.7, 0.938),
            ("K2", 6.0, 1.5, 0.20, -0.25, 1.2, 0.938),
            ("K3", 10.0, 4.0, 0.30, -0.8, 2.1, 0.938),
        )
    ]
    form_factors = [(0.5, 0.0), (0.8, 0.0), (1.0, 0.2)]
    form_factors += [(1.0, -0.2), (0.7, 0.5), (0.0, 1.0)]
    log_path = Path("Output") / "BHHelicityAmp.log"

    ref_F1, ref_F2 = 1.0, 0.0
    pol_h_in, pol_s = +1, +1
    pol_h_analytic = 0.5 * pol_h_in
    azimuth_input_header = (
        "phi_hadron input [rad]"
        if azimuth_input == "phi_hadron"
        else "phi_out input [rad]"
    )

    def build_case(inputs):
        kin = kinematics_user_from_scalar_inputs(
            inputs["Eb"], inputs["Q2"], inputs["xB"],
            inputs["t"], inputs["phi"], inputs["m"],
            azimuth_input=azimuth_input,
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

    def backend_row(case):
        params = case["kin"]["user_params"]
        return [
            case["id"],
            *values_from(params, ("pIn", "pOut", "qOut", "th", "ph")),
            fmt(case["kin"]["phi_hadron"]),
            fmt(params["phOut"]),
            fmt(case["kin"]["user_rebuild_residual"]),
        ]

    def spin_row(case):
        S = case["S"]
        return [
            case["id"], *(fmt(component) for component in S),
            fmt(_real_scalar(mdot(S, case["mom"]["p"]), "S.p")),
            fmt(_real_scalar(mdot(S, S), "S^2")),
        ]

    independent_rows = [
        row_from(case, "input", ("Eb", "Q2", "xB", "t", "phi", "m"))
        for case in cases
    ]
    derived_rows = [
        row_from(case, "kin", ("s", "sqrt_s", "nu", "y", "beta_target_rest_to_cm"))
        for case in cases
    ]
    backend_rows = [backend_row(case) for case in cases]
    momentum_rows = [
        [case["id"], *vector_row(name, case["mom"][name])]
        for case in cases
        for name in ("k", "p", "kp", "pp", "qout", "q")
    ]
    scalar_rows = [
        row
        for case in cases
        for row in scalar_checks(case["id"], case["kin"])
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
            ["case", "Eb [GeV]", "Q2 [GeV^2]", "xB", "t [GeV^2]",
             azimuth_input_header, "m [GeV]"],
            independent_rows,
        ),
        "derived": (
            ["case", "s [GeV^2]", "sqrt(s) [GeV]", "nu [GeV]", "y", "beta_cm"],
            derived_rows,
        ),
        "backend": (
            ["case", "pIn [GeV]", "pOut [GeV]", "qOut [GeV]", "th [rad]",
             "ph [rad]", "phi_hadron [rad]", "phOut [rad]", "rebuild diff [GeV]"],
            backend_rows,
        ),
        "momenta": (
            ["case", "vec", "E [GeV]", "px [GeV]", "py [GeV]", "pz [GeV]",
             "phi_xy [rad]"],
            momentum_rows,
        ),
        "scalar": (
            ["case", "scalar", "target", "from 4-mom", "diff", "unit"],
            scalar_rows,
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
          Each K row is one independent scalar input point; these inputs are unchanged.
          Eb: target-rest beam-energy scalar in GeV; it fixes s = m^2 + 2 m Eb.
          Q2: -q^2 in GeV^2, with q = k - kp.
          xB: Bjorken xB = Q2 / (2 p.q).
          t: Delta^2 = (pp - p)^2 in GeV^2.
          phi_hadron: hadron-plane azimuth around the virtual photon direction.
          phi_out: user-backend outgoing-photon coordinate azimuth phOut.
          Selected input azimuth convention: {azimuth_input}.
          m: proton mass in GeV.
          Backend: scalar inputs are converted to pIn, pOut, qOut, th, ph,
          phOut, then rebuilt with momenta_user. Set BENCHMARK_AZIMUTH_INPUT =
          'phi_out' to interpret the input phi column as phOut.
          For fixed Eb, Q2, xB, and t, not every phi_out value is physical;
          invalid phi_out choices raise a clear error before benchmarking.
          Four-momenta are reported in the user-kinematics COM frame as
          [E, px, py, pz] in GeV, with pp=(E,0,pOut,0) and
          qout=qOut(1,cos(phOut),sin(phOut),0).
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
        "Independent scalar inputs",
        (
            "  The numeric input rows are kept unchanged; the phi column label "
            "shows how those numbers are interpreted in this run."
        ),
        tables["independent"],
        "",
        "Derived scalar kinematics",
        tables["derived"],
        "",
        "Derived user-kinematics backend variables",
        tables["backend"],
        "",
        "Four-momenta from momenta_user backend",
        tables["momenta"],
        "",
        "Four-momentum scalar and backend reproduction checks",
        tables["scalar"],
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
