"""Derive fully contracted planar user-frame Bethe-Heitler amplitudes.

This script uses SymPy only as an offline algebra engine.  It fixes
``theta_in = pi/2`` and contracts every Dirac matrix, external spinor, and
photon-polarization component in the conventions of ``Algebra.py`` and
``BHHelicityAmp.py``.  The generated Markdown file contains scalar formulas
for all helicity amplitudes; no spinors or gamma matrices remain.
"""

from itertools import product
import json
from pathlib import Path

import sympy as sp


I = sp.I
SQRT2 = sp.sqrt(2)

# Explicit script settings.
MARKDOWN_OUTPUT_PATH = Path("Output") / "AnalyticBH_theta_pi_over_2.md"
TEX_OUTPUT_PATH = Path("Output") / "AnalyticBH_theta_pi_over_2.tex"
WRITE_VALIDATION_JSON = False
VALIDATION_OUTPUT_PATH = Path("Output") / "AnalyticBH_symbolic_validation.json"
VALIDATION_S = 12.0
VALIDATION_ALPHA = 0.4
VALIDATION_PHOTON_ENERGY = 0.8
VALIDATION_PHI = 1.1
VALIDATION_PROTON_MASS = 0.938
VALIDATION_F1 = 0.7
VALIDATION_F2 = 1.2


def gamma_matrices():
    """Return Dirac-representation gamma matrices with mostly-minus metric."""
    one = sp.eye(2)
    zero = sp.zeros(2)
    sigma = (
        sp.Matrix([[0, 1], [1, 0]]),
        sp.Matrix([[0, -I], [I, 0]]),
        sp.Matrix([[1, 0], [0, -1]]),
    )
    return (
        sp.diag(1, 1, -1, -1),
        *(
            sp.Matrix.vstack(
                sp.Matrix.hstack(zero, matrix),
                sp.Matrix.hstack(-matrix, zero),
            )
            for matrix in sigma
        ),
    )


GAMMA = gamma_matrices()
ETA = (1, -1, -1, -1)


def slash(vector):
    return sum(
        (ETA[mu] * vector[mu] * GAMMA[mu] for mu in range(4)),
        sp.zeros(4),
    )


def bar(spinor):
    return sp.conjugate(spinor.T) * GAMMA[0]


def electron_spinor_in(h, P, ca, sa):
    root = sp.sqrt(P) / SQRT2
    chi = sp.Matrix([1, -ca - I * sa]) if h == 1 else sp.Matrix([ca - I * sa, 1])
    return root * sp.Matrix.vstack(chi, h * chi)


def electron_spinor_out(h, K, X, Y):
    root = sp.sqrt(K) / SQRT2
    chi = sp.Matrix([1, (-X - I * Y) / K]) if h == 1 else sp.Matrix([(X - I * Y) / K, 1])
    return root * sp.Matrix.vstack(chi, h * chi)


def proton_spinor_in(s, P, A, ca, sa):
    chi = sp.Matrix([1, ca + I * sa]) if s == 1 else sp.Matrix([-ca + I * sa, 1])
    return sp.Matrix.vstack(A * chi, s * P * chi / A) / SQRT2


def proton_spinor_out(s, R, B):
    chi = sp.Matrix([1, I]) if s == 1 else sp.Matrix([I, 1])
    return sp.Matrix.vstack(B * chi, s * R * chi / B) / SQRT2


def derive_amplitudes():
    P, R, w, E, Epr, K, A, B, m = sp.symbols(
        "P R w E Epr K A B m", positive=True, real=True
    )
    F1, F2, ca, sa, cf, sf = sp.symbols(
        "F1 F2 ca sa cf sf", real=True
    )
    X = w * cf
    Y = R + w * sf

    k = sp.Matrix([P, -P * ca, -P * sa, 0])
    p = sp.Matrix([E, P * ca, P * sa, 0])
    kp = sp.Matrix([K, -X, -Y, 0])
    pp = sp.Matrix([Epr, 0, R, 0])
    qout = sp.Matrix([w, X, w * sf, 0])

    dot = lambda left, right: sum(ETA[mu] * left[mu] * right[mu] for mu in range(4))
    t = 2 * m**2 - 2 * E * Epr + 2 * P * R * sa
    den1 = 2 * dot(kp, qout)
    den2 = -2 * dot(k, qout)
    qslash = slash(qout)
    psum_cov = sp.Matrix([ETA[mu] * (p[mu] + pp[mu]) for mu in range(4)])

    electron_in = {h: electron_spinor_in(h, P, ca, sa) for h in (-1, 1)}
    electron_out = {h: electron_spinor_out(h, K, X, Y) for h in (-1, 1)}
    proton_in = {s: proton_spinor_in(s, P, A, ca, sa) for s in (-1, 1)}
    proton_out = {s: proton_spinor_out(s, R, B) for s in (-1, 1)}

    hadronic = {}
    for s, spout in product((-1, 1), repeat=2):
        up = proton_in[s]
        upp = proton_out[spout]
        components = []
        for nu in range(4):
            gamma_lower = ETA[nu] * GAMMA[nu]
            vertex = (F1 + F2) * gamma_lower - psum_cov[nu] * F2 * sp.eye(4) / (2 * m)
            components.append(sp.expand((bar(upp) * vertex * up)[0]))
        hadronic[(s, spout)] = components

    leptonic = {}
    leptonic_numerators = {}
    for h, lam in product((-1, 1), repeat=2):
        ue = electron_in[h]
        uep = electron_out[h]
        eps = sp.Matrix([0, -I * lam * sf / SQRT2, I * lam * cf / SQRT2, -1 / SQRT2])
        eps_cov_star = sp.Matrix([ETA[mu] * sp.conjugate(eps[mu]) for mu in range(4)])
        components = []
        numerator_components = []
        for nu in range(4):
            numerator_one = 0
            numerator_two = 0
            for mu in range(4):
                numerator_one += eps_cov_star[mu] * (
                    bar(uep)
                    * (2 * kp[mu] * GAMMA[nu] + GAMMA[mu] * qslash * GAMMA[nu])
                    * ue
                )[0]
                numerator_two += eps_cov_star[mu] * (
                    bar(uep)
                    * (2 * k[mu] * GAMMA[nu] - GAMMA[nu] * qslash * GAMMA[mu])
                    * ue
                )[0]
            numerator_one = sp.factor_terms(sp.expand(numerator_one))
            numerator_two = sp.factor_terms(sp.expand(numerator_two))
            numerator_components.append((numerator_one, numerator_two))
            components.append(numerator_one / den1 + numerator_two / den2)
        leptonic[(h, lam)] = components
        leptonic_numerators[(h, lam)] = numerator_components

    amplitudes = {}
    for h, hp, s, spout, lam in product((-1, 1), repeat=5):
        if hp != h:
            amplitudes[(h, hp, s, spout, lam)] = sp.S.Zero
            continue
        value = sum(
            leptonic[(h, lam)][nu] * hadronic[(s, spout)][nu]
            for nu in range(4)
        ) / t
        amplitudes[(h, hp, s, spout, lam)] = sp.factor_terms(sp.together(value))

    definitions = {
        "P": "(s-m^2)/(2 sqrt(s))",
        "R": "pOut, fixed by energy conservation",
        "w": "qOut",
        "E": "sqrt(P^2+m^2)",
        "Epr": "sqrt(R^2+m^2)",
        "K": "sqrt(R^2+w^2+2 R w sf)",
        "A": "sqrt(E+m)",
        "B": "sqrt(Epr+m)",
        "ca": "cos(phi_in)",
        "sa": "sin(phi_in)",
        "cf": "cos(phiOut)",
        "sf": "sin(phiOut)",
    }
    return amplitudes, definitions, leptonic, hadronic, leptonic_numerators


def write_markdown(amplitudes, definitions, output_path):
    """Write a short physical index to the detailed TeX formulas."""
    lines = [
        "# Fully contracted Bethe-Heitler helicity amplitudes",
        "",
        "Planar user frame with `theta_in = pi/2`; helicities are doubled helicities.",
        "",
        "The complete scalar formulas are in `AnalyticBH_theta_pi_over_2.tex`.",
        "They contain no anonymous `x_i` substitutions.",
        "",
        "## Kinematics",
        "",
        "```text",
        "P = (s-m^2)/(2 sqrt(s))",
        "E = sqrt(P^2+m^2)",
        "R = pOut, fixed by sqrt(R^2+m^2) + K + qOut = sqrt(s)",
        "K = sqrt(R^2+qOut^2+2 R qOut sin(phiOut))",
        "Dplus  = 2 kp.qout = 2 qOut (K+qOut+R sin(phiOut))",
        "Dminus = -2 k.qout = -2 P qOut [1+cos(phi_in-phiOut)]",
        "t = 2m^2 - 2 E sqrt(R^2+m^2) + 2 P R sin(phi_in)",
        "```",
        "",
        "## Amplitude organization",
        "",
        "The TeX file first gives all contracted scalar leptonic numerators `N` and",
        "proton-current components `H`, then spells out all 16 surviving amplitudes as",
        "",
        "```text",
        "L^nu_(h,lambda) = N^(nu,+)_(h,lambda)/Dplus + N^(nu,-)_(h,lambda)/Dminus",
        "M_(h,s,sprime,lambda) = (1/t) sum_nu L^nu_(h,lambda) H_nu^(sprime,s)",
        "```",
        "",
        "The 16 channels with `hprime != h` vanish exactly for the massless electron.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return 16, 16, 0


def write_tex(amplitudes, leptonic_numerators, hadronic, output_path):
    """Write a physics-organized TeX result without anonymous CSE variables."""
    all_expressions = [
        expression
        for components in hadronic.values()
        for expression in components
    ] + [
        expression
        for components in leptonic_numerators.values()
        for pair in components
        for expression in pair
    ]
    source_symbols = set().union(*(expression.free_symbols for expression in all_expressions))
    by_name = {symbol.name: symbol for symbol in source_symbols}
    za, zam, zf, zfm = sp.symbols("z_a z_am z_f z_fm")
    phase_substitution = {
        by_name["ca"]: (za + zam) / 2,
        by_name["sa"]: (za - zam) / (2 * I),
        by_name["cf"]: (zf + zfm) / 2,
        by_name["sf"]: (zf - zfm) / (2 * I),
    }

    def clean(expression):
        result = sp.expand(expression.subs(phase_substitution))
        result = result.subs(zf * zfm, 1).subs(za * zam, 1)
        return sp.factor_terms(result)

    tex_names_by_text = {
        "Epr": r"E'", "F1": r"F_1", "F2": r"F_2",
        "z_a": r"z_a", "z_am": r"\bar z_a",
        "z_f": r"z_f", "z_fm": r"\bar z_f",
    }
    latex_symbols = source_symbols | {za, zam, zf, zfm}
    symbol_names = {
        symbol: tex_names_by_text[symbol.name]
        for symbol in latex_symbols
        if symbol.name in tex_names_by_text
    }

    def latex(expression):
        return sp.latex(clean(expression), symbol_names=symbol_names)

    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.55in,landscape]{geometry}",
        r"\usepackage{amsmath,amssymb,graphicx}",
        r"\allowdisplaybreaks",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
        r"\title{Contracted Bethe--Heitler Helicity Amplitudes in User-Frame Kinematics}",
        r"\author{$\theta_{\mathrm{in}}=\pi/2$}",
        r"\date{}",
        r"\maketitle",
        r"Helicity labels are doubled helicities in $\{-1,+1\}$. The electron is massless, so $h'=h$ and every amplitude with $h'\ne h$ vanishes.",
        r"\section*{User-frame kinematics}",
        r"Define $\alpha=\phi_{\mathrm{in}}$, $\varphi=\phi_{\mathrm{Out}}$, $w=q_{\mathrm{Out}}$, and",
        r"\begin{align*}",
        r"P&=\frac{s-m^2}{2\sqrt{s}}, & E&=\sqrt{P^2+m^2},\\",
        r"R&=p_{\mathrm{Out}}, & E'&=\sqrt{R^2+m^2},\\",
        r"K&=\sqrt{R^2+w^2+2Rw\sin\varphi}, &",
        r"\sqrt{E'^2-m^2}&=R.",
        r"\end{align*}",
        r"Energy conservation fixes $R$ through $E'+K+w=\sqrt{s}$. The external momenta are",
        r"\begin{align*}",
        r"k^\mu&=P(1,-\cos\alpha,-\sin\alpha,0),\\",
        r"p^\mu&=(E,P\cos\alpha,P\sin\alpha,0),\\",
        r"k'^\mu&=(K,-w\cos\varphi,-R-w\sin\varphi,0),\\",
        r"p'^\mu&=(E',0,R,0),\\",
        r"q'^\mu&=w(1,\cos\varphi,\sin\varphi,0).",
        r"\end{align*}",
        r"The momentum products that contain every propagator denominator are",
        r"\begin{align*}",
        r"D_+&\equiv2k'\!\cdot q'=2w(K+w+R\sin\varphi),\\",
        r"D_-&\equiv-2k\!\cdot q'=-2Pw\,[1+\cos(\alpha-\varphi)],\\",
        r"t&=(p'-p)^2=2m^2-2EE'+2PR\sin\alpha.",
        r"\end{align*}",
        r"To shorten phases without hiding kinematics, define only",
        r"\[z_a=e^{i\alpha},\quad \bar z_a=e^{-i\alpha},\qquad z_f=e^{i\varphi},\quad \bar z_f=e^{-i\varphi}.\]",
        r"\section*{Fully contracted scalar building blocks}",
        r"After all Dirac and polarization contractions, write",
        r"\[\mathcal L^\nu_{h\lambda}=\frac{N^{\nu,+}_{h\lambda}}{D_+}+\frac{N^{\nu,-}_{h\lambda}}{D_-},\qquad \mathcal H^{s's}_\nu=\bar u(p',s')\Gamma_\nu u(p,s).\]",
        r"The following equations are explicit scalars; no gamma matrices or spinors remain.",
        r"\subsection*{Leptonic numerators}",
        r"\small",
    ]
    for (h, lam), components in leptonic_numerators.items():
        lines.append(rf"\paragraph{{$h={h},\ \lambda={lam}$}}")
        for nu, (plus, minus) in enumerate(components):
            lines += [
                r"\begin{equation*}",
                rf"\resizebox{{0.98\textwidth}}{{!}}{{${{\displaystyle N^{{{nu},+}}_{{{h},{lam}}}={latex(plus)}}}$}}",
                r"\end{equation*}",
                r"\begin{equation*}",
                rf"\resizebox{{0.98\textwidth}}{{!}}{{${{\displaystyle N^{{{nu},-}}_{{{h},{lam}}}={latex(minus)}}}$}}",
                r"\end{equation*}",
            ]
    lines += [r"\normalsize", r"\subsection*{Proton-current components}", r"\small"]
    for (s, spout), components in hadronic.items():
        lines.append(rf"\paragraph{{$s={s},\ s'={spout}$}}")
        for nu, expression in enumerate(components):
            lines += [
                r"\begin{equation*}",
                rf"\resizebox{{0.98\textwidth}}{{!}}{{${{\displaystyle \mathcal H^{{{spout},{s}}}_{nu}={latex(expression)}}}$}}",
                r"\end{equation*}",
            ]
    lines += [
        r"\normalsize",
        r"\section*{All helicity amplitudes}",
        r"Every nonzero amplitude is the following explicit four-component momentum-space scalar product:",
        r"\[\boxed{\mathcal M_{h,s,s',\lambda}=\frac{1}{t}\sum_{\nu=0}^{3}\left(\frac{N^{\nu,+}_{h\lambda}}{D_+}+\frac{N^{\nu,-}_{h\lambda}}{D_-}\right)\mathcal H^{s's}_\nu}.\]",
        r"Spelling out the four terms for every surviving helicity channel gives",
        r"\small",
        r"\begin{align*}",
    ]
    amplitude_rows = []
    for h, s, spout, lam in product((-1, 1), repeat=4):
        terms = "+".join(
            rf"\left(\frac{{N^{{{nu},+}}_{{{h},{lam}}}}}{{D_+}}+\frac{{N^{{{nu},-}}_{{{h},{lam}}}}}{{D_-}}\right)\mathcal H^{{{spout},{s}}}_{nu}"
            for nu in range(4)
        )
        amplitude_rows.append(
            rf"\mathcal M_{{h={h},s={s},s'={spout},\lambda={lam}}}&=\frac{{1}}{{t}}\left[{terms}\right]"
        )
    lines.append(r" \\[5pt]".join(amplitude_rows))
    lines += [
        r"\end{align*}",
        r"\normalsize",
        r"The remaining 16 amplitudes obey $\mathcal M_{h'\ne h,s,s',\lambda}=0$ exactly.",
        r"\end{document}",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return 16, 16, 0


def main():
    amplitudes, definitions, leptonic, hadronic, leptonic_numerators = derive_amplitudes()
    counts = write_markdown(amplitudes, definitions, MARKDOWN_OUTPUT_PATH)
    print(
        f"Wrote {MARKDOWN_OUTPUT_PATH}: {counts[0]} nonzero, "
        f"{counts[1]} zero, {counts[2]} common expressions"
    )
    tex_counts = write_tex(
        amplitudes, leptonic_numerators, hadronic, TEX_OUTPUT_PATH
    )
    print(
        f"Wrote {TEX_OUTPUT_PATH}: {tex_counts[0]} nonzero, "
        f"{tex_counts[1]} zero, {tex_counts[2]} anonymous x substitutions"
    )
    if WRITE_VALIDATION_JSON:
        s_value = VALIDATION_S
        alpha = VALIDATION_ALPHA
        w_value = VALIDATION_PHOTON_ENERGY
        phi = VALIDATION_PHI
        m_value = VALIDATION_PROTON_MASS
        f1_value = VALIDATION_F1
        f2_value = VALIDATION_F2
        P_value = (s_value - m_value**2) / (2.0 * s_value**0.5)
        target = s_value**0.5
        low, high = 0.0, target
        for _ in range(100):
            middle = 0.5 * (low + high)
            residual = (
                (middle**2 + m_value**2) ** 0.5
                + (middle**2 + w_value**2 + 2 * middle * w_value * sp.sin(phi)) ** 0.5
                + w_value
                - target
            )
            if residual > 0:
                high = middle
            else:
                low = middle
        R_value = 0.5 * (low + high)
        values = {
            "P": P_value,
            "R": R_value,
            "w": w_value,
            "E": (P_value**2 + m_value**2) ** 0.5,
            "Epr": (R_value**2 + m_value**2) ** 0.5,
            "K": (R_value**2 + w_value**2 + 2 * R_value * w_value * float(sp.sin(phi))) ** 0.5,
            "A": ((P_value**2 + m_value**2) ** 0.5 + m_value) ** 0.5,
            "B": ((R_value**2 + m_value**2) ** 0.5 + m_value) ** 0.5,
            "m": m_value,
            "F1": f1_value,
            "F2": f2_value,
            "ca": float(sp.cos(alpha)),
            "sa": float(sp.sin(alpha)),
            "cf": float(sp.cos(phi)),
            "sf": float(sp.sin(phi)),
        }
        rows = {}
        for key, expression in amplitudes.items():
            substitution = {symbol: values[symbol.name] for symbol in expression.free_symbols}
            number = complex(sp.N(expression.subs(substitution), 16))
            rows[",".join(map(str, key))] = [number.real, number.imag]
        validation = {
            "inputs": {"s": s_value, "alpha": alpha, "w": w_value, "phi": phi, "m": m_value, "F1": f1_value, "F2": f2_value},
            "amplitudes": rows,
        }
        VALIDATION_OUTPUT_PATH.write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        print(f"Wrote {VALIDATION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
