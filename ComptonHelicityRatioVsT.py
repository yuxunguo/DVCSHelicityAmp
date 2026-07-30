"""Analyze lepton-helicity flip versus conservation in the GHZ Compton block.

For every saved canonical dGHZ configuration, this script evaluates

    gamma*(q) + lepton(k) -> gamma(qOut) + lepton(k')

and forms the transverse-polarization power ratio

    R_T = sum |C|^2_(hOut != hIn) / sum |C|^2_(hOut == hIn).

The corresponding ratio including the longitudinal virtual-photon
polarization is saved as ``R_all``.  The calculation uses the physical
lepton mass stored in each configuration row.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from Algebra import mdot, photon_pol
from BHHelicityAmp import proton_current_helicity_decomposition
from FormFactors import yahl_dirac_pauli_from_t
from Kinematics import kinematics_cm_from_independent
from PlotUtils import print_console_text, require_matplotlib
from ProtonVirtualPhotonAmp import (
    proton_emission_matrix,
    virtual_photon_polarizations,
)
from QuasiRealComptonHelicity import amplitude_with_vectors
from config import PROTON_MASS_GEV


DEFAULT_INPUT_ROOT = (
    Path("Output")
    / "GradientPhaseSpaceScan"
    / "electron"
    / "Data"
    / "GHZ"
    / "dghz"
)
DEFAULT_DATA_PATH = (
    DEFAULT_INPUT_ROOT
    / "helicity_analysis"
    / "compton_helicity_flip_to_conserving_vs_t_electron.csv"
)
DEFAULT_SUMMARY_PATH = (
    DEFAULT_INPUT_ROOT
    / "helicity_analysis"
    / "compton_helicity_flip_to_conserving_summary_electron.csv"
)
DEFAULT_PLOT_PATH = (
    Path("Output")
    / "GradientPhaseSpaceScan"
    / "electron"
    / "Plots"
    / "GHZ"
    / "helicity_analysis"
    / "Compton_Helicity_Flip_to_Conserving_Ratio_vs_t_Electron.pdf"
)

HELICITIES = (+1, -1)
TRANSVERSE_POLARIZATIONS = ("T-", "T+")
PHYSICAL_POLARIZATIONS = ("T-", "T+", "L")
ENDPOINT_ANGLE_TOLERANCE = 1.0e-8


def read_csv(path):
    """Return all rows from one UTF-8 CSV."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    """Write dictionaries to a CSV, preserving the first row's field order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows were supplied for {path}.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_configuration_rows(input_root):
    """Join canonical configuration IDs to their detailed kinematic rows."""
    input_root = Path(input_root)
    canonical = {}
    pattern = "canonical_dghz_*_configurations_*.csv"
    for path in sorted(input_root.glob(f"**/{pattern}")):
        for row in read_csv(path):
            canonical[row["detail_id"]] = row
    if not canonical:
        raise FileNotFoundError(
            f"No canonical configuration CSVs found below {input_root}."
        )

    examples = {}
    for path in sorted(input_root.glob("**/*_configuration_examples.csv")):
        for row in read_csv(path):
            detail_id = row["detail_id"]
            if detail_id in canonical:
                examples[detail_id] = row

    missing = sorted(set(canonical) - set(examples))
    if missing:
        raise KeyError(
            f"{len(missing)} canonical configurations have no detail row; "
            f"first missing ID: {missing[0]}"
        )
    return [
        (canonical[detail_id], examples[detail_id])
        for detail_id in sorted(canonical)
    ]


def geometry_class(theta_gamma):
    """Classify the final photon relative to the fixed initial CM axis."""
    if abs(theta_gamma) <= ENDPOINT_ANGLE_TOLERANCE:
        return "forward_endpoint"
    if abs(theta_gamma - np.pi) <= ENDPOINT_ANGLE_TOLERANCE:
        return "backward_endpoint"
    return "noncollinear"


def compton_ratio_row(canonical_row, detail_row):
    """Calculate helicity-flip and helicity-conserving Compton norms."""
    lepton_mass = float(detail_row["lepton_mass"])
    kin = kinematics_cm_from_independent(
        float(detail_row["s"]),
        float(detail_row["qOut"]),
        float(detail_row["theta_p_out"]),
        float(detail_row["phi_p_out"]),
        float(detail_row["theta_gamma_out"]),
        float(detail_row["phi_gamma_out"]),
        PROTON_MASS_GEV,
        lepton_mass,
    )
    momenta = kin["momenta"]
    virtual_photon = momenta["p"] - momenta["pp"]
    polarizations = virtual_photon_polarizations(virtual_photon)
    compton_momenta = {
        "q_in": virtual_photon,
        "k_in": momenta["k"],
        "q_out": momenta["qout"],
        "k_out": momenta["kp"],
    }

    sums = {
        "transverse_flip": 0.0,
        "transverse_conserving": 0.0,
        "all_flip": 0.0,
        "all_conserving": 0.0,
    }
    maxima = {"flip": 0.0, "conserving": 0.0}
    compton_matrix = np.zeros((2, 2, 3, 2), dtype=complex)
    for polarization_index, polarization_name in enumerate(
        PHYSICAL_POLARIZATIONS
    ):
        for h_in_index, h_in in enumerate(HELICITIES):
            for photon_index, photon_helicity in enumerate(HELICITIES):
                epsilon_out = photon_pol(
                    momenta["qout"],
                    photon_helicity,
                )
                for h_out_index, h_out in enumerate(HELICITIES):
                    amplitude = amplitude_with_vectors(
                        compton_momenta,
                        polarizations[polarization_name],
                        epsilon_out,
                        h_in,
                        h_out,
                        electron_mass=lepton_mass,
                    )
                    compton_matrix[
                        photon_index,
                        h_out_index,
                        polarization_index,
                        h_in_index,
                    ] = amplitude
                    amplitude_abs = abs(amplitude)
                    amplitude_abs2 = amplitude_abs**2
                    channel = (
                        "conserving" if h_out == h_in else "flip"
                    )
                    sums[f"all_{channel}"] += amplitude_abs2
                    if polarization_name in TRANSVERSE_POLARIZATIONS:
                        sums[f"transverse_{channel}"] += amplitude_abs2
                    maxima[channel] = max(maxima[channel], amplitude_abs)

    transverse_ratio = (
        sums["transverse_flip"] / sums["transverse_conserving"]
    )
    all_ratio = sums["all_flip"] / sums["all_conserving"]

    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], PROTON_MASS_GEV)
    decomposition = proton_current_helicity_decomposition(
        momenta["p"],
        momenta["pp"],
        PROTON_MASS_GEV,
        F1,
        F2,
    )
    proton_matrix = proton_emission_matrix(
        decomposition,
        polarizations,
    ).reshape(2, 3, 2)
    proton_state = np.asarray((
        np.cos(float(detail_row["alpha_p"])),
        np.sin(float(detail_row["alpha_p"])),
    ))
    electron_state = np.asarray((
        np.cos(float(detail_row["alpha_e"])),
        np.sin(float(detail_row["alpha_e"])),
    ))
    prepared_proton = np.einsum(
        "api,i->ap",
        proton_matrix,
        proton_state,
    )
    polarization_propagator = np.asarray((
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ), dtype=complex) / kin["t"]
    full_flip = np.zeros(8, dtype=complex)
    full_conserving = np.zeros(8, dtype=complex)
    full_transverse = np.zeros(8, dtype=complex)
    full_longitudinal = np.zeros(8, dtype=complex)
    final_index = 0
    for proton_out_index in range(2):
        for photon_index in range(2):
            for h_out_index, h_out in enumerate(HELICITIES):
                for proton_polarization_index in range(3):
                    for compton_polarization_index in range(3):
                        proton_factor = (
                            prepared_proton[
                                proton_out_index,
                                proton_polarization_index,
                            ]
                            * polarization_propagator[
                                compton_polarization_index,
                                proton_polarization_index,
                            ]
                        )
                        for h_in_index, h_in in enumerate(HELICITIES):
                            contribution = (
                                proton_factor
                                * compton_matrix[
                                    photon_index,
                                    h_out_index,
                                    compton_polarization_index,
                                    h_in_index,
                                ]
                                * electron_state[h_in_index]
                            )
                            if h_out == h_in:
                                full_conserving[final_index] += contribution
                            else:
                                full_flip[final_index] += contribution
                            if proton_polarization_index < 2:
                                full_transverse[final_index] += contribution
                            else:
                                full_longitudinal[final_index] += contribution
                final_index += 1
    full_flip_norm2 = float(np.vdot(full_flip, full_flip).real)
    full_conserving_norm2 = float(
        np.vdot(full_conserving, full_conserving).real
    )
    full_ratio = full_flip_norm2 / full_conserving_norm2
    full_interference = float(
        2.0 * np.real(np.vdot(full_conserving, full_flip))
    )
    full_transverse_norm2 = float(
        np.vdot(full_transverse, full_transverse).real
    )
    full_longitudinal_norm2 = float(
        np.vdot(full_longitudinal, full_longitudinal).real
    )
    full_longitudinal_to_transverse_ratio = (
        full_longitudinal_norm2 / full_transverse_norm2
    )
    full_longitudinal_transverse_interference = float(
        2.0 * np.real(np.vdot(full_transverse, full_longitudinal))
    )
    k_dot_qout = abs(
        float(np.real(mdot(momenta["k"], momenta["qout"])))
    )
    kp_dot_qout = abs(
        float(np.real(mdot(momenta["kp"], momenta["qout"])))
    )
    if lepton_mass > 0.0:
        collinearity_rho = (
            2.0 * min(k_dot_qout, kp_dot_qout) / lepton_mass**2
        )
    else:
        collinearity_rho = np.inf

    return {
        "detail_id": detail_row["detail_id"],
        "polarization_cluster_number": canonical_row[
            "polarization_cluster_number"
        ],
        "abs_t_GeV2": abs(float(kin["t"])),
        "t_GeV2": float(kin["t"]),
        "sqrt_s_GeV": float(kin["sqrt_s"]),
        "theta_gamma_out_rad": float(detail_row["theta_gamma_out"]),
        "theta_p_out_rad": float(detail_row["theta_p_out"]),
        "geometry_class": geometry_class(
            float(detail_row["theta_gamma_out"])
        ),
        "lepton_mass_GeV": lepton_mass,
        "k_dot_qout_GeV2": k_dot_qout,
        "kp_dot_qout_GeV2": kp_dot_qout,
        "collinearity_rho": collinearity_rho,
        "transverse_flip_abs2_sum": sums["transverse_flip"],
        "transverse_conserving_abs2_sum": sums[
            "transverse_conserving"
        ],
        "transverse_flip_to_conserving_ratio": transverse_ratio,
        "transverse_flip_to_conserving_amplitude_ratio": np.sqrt(
            transverse_ratio
        ),
        "all_flip_abs2_sum": sums["all_flip"],
        "all_conserving_abs2_sum": sums["all_conserving"],
        "all_flip_to_conserving_ratio": all_ratio,
        "all_flip_to_conserving_amplitude_ratio": np.sqrt(all_ratio),
        "maximum_flip_to_conserving_amplitude_ratio": (
            maxima["flip"] / maxima["conserving"]
        ),
        "full_chain_flip_norm2": full_flip_norm2,
        "full_chain_conserving_norm2": full_conserving_norm2,
        "full_chain_flip_to_conserving_ratio": full_ratio,
        "full_chain_flip_to_conserving_amplitude_ratio": np.sqrt(
            full_ratio
        ),
        "full_chain_flip_conserving_interference": full_interference,
        "full_chain_transverse_norm2": full_transverse_norm2,
        "full_chain_longitudinal_norm2": full_longitudinal_norm2,
        "full_chain_longitudinal_to_transverse_ratio": (
            full_longitudinal_to_transverse_ratio
        ),
        "full_chain_longitudinal_transverse_interference": (
            full_longitudinal_transverse_interference
        ),
        "dGHZ": float(detail_row["selected_concurrence"]),
        "alpha_e": float(detail_row["alpha_e"]),
        "alpha_p": float(detail_row["alpha_p"]),
    }


def summary_rows(rows):
    """Return geometry-separated summary statistics."""
    result = []
    for name in (
        "noncollinear",
        "backward_endpoint",
        "forward_endpoint",
        "all",
    ):
        selected = (
            rows
            if name == "all"
            else [row for row in rows if row["geometry_class"] == name]
        )
        if not selected:
            continue
        ratios = np.asarray([
            float(row["transverse_flip_to_conserving_ratio"])
            for row in selected
        ])
        abs_t = np.asarray([float(row["abs_t_GeV2"]) for row in selected])
        rho = np.asarray([
            float(row["collinearity_rho"])
            for row in selected
            if np.isfinite(float(row["collinearity_rho"]))
        ])
        result.append({
            "geometry_class": name,
            "configuration_count": len(selected),
            "abs_t_min_GeV2": float(np.min(abs_t)),
            "abs_t_max_GeV2": float(np.max(abs_t)),
            "ratio_median": float(np.median(ratios)),
            "ratio_q10": float(np.quantile(ratios, 0.10)),
            "ratio_q90": float(np.quantile(ratios, 0.90)),
            "full_chain_ratio_median": float(np.median([
                float(row["full_chain_flip_to_conserving_ratio"])
                for row in selected
            ])),
            "full_chain_flip_dominated_count": sum(
                float(row["full_chain_flip_to_conserving_ratio"]) > 1.0
                for row in selected
            ),
            "full_chain_conserving_dominated_count": sum(
                float(row["full_chain_flip_to_conserving_ratio"]) <= 1.0
                for row in selected
            ),
            "full_chain_longitudinal_to_transverse_ratio_median": (
                float(np.median([
                    float(
                        row[
                            "full_chain_longitudinal_to_transverse_ratio"
                        ]
                    )
                    for row in selected
                ]))
            ),
            "collinearity_rho_median": (
                float(np.median(rho)) if rho.size else np.nan
            ),
        })
    return result


def load_numeric_rows(path):
    """Load a previously calculated ratio CSV for plot-only mode."""
    rows = read_csv(path)
    numeric_fields = {
        "abs_t_GeV2",
        "t_GeV2",
        "sqrt_s_GeV",
        "theta_gamma_out_rad",
        "theta_p_out_rad",
        "lepton_mass_GeV",
        "k_dot_qout_GeV2",
        "kp_dot_qout_GeV2",
        "collinearity_rho",
        "transverse_flip_abs2_sum",
        "transverse_conserving_abs2_sum",
        "transverse_flip_to_conserving_ratio",
        "transverse_flip_to_conserving_amplitude_ratio",
        "all_flip_abs2_sum",
        "all_conserving_abs2_sum",
        "all_flip_to_conserving_ratio",
        "all_flip_to_conserving_amplitude_ratio",
        "maximum_flip_to_conserving_amplitude_ratio",
        "full_chain_flip_norm2",
        "full_chain_conserving_norm2",
        "full_chain_flip_to_conserving_ratio",
        "full_chain_flip_to_conserving_amplitude_ratio",
        "full_chain_flip_conserving_interference",
        "full_chain_transverse_norm2",
        "full_chain_longitudinal_norm2",
        "full_chain_longitudinal_to_transverse_ratio",
        "full_chain_longitudinal_transverse_interference",
        "dGHZ",
        "alpha_e",
        "alpha_p",
    }
    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])
    return rows


def save_plot(rows, path):
    """Save the t dependence and the controlling collinearity diagnostic."""
    plt, _ = require_matplotlib()
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
    })
    styles = {
        "noncollinear": ("tab:blue", "o", "non-collinear"),
        "backward_endpoint": (
            "tab:red",
            "^",
            r"backward endpoint $\theta_\gamma=\pi$",
        ),
        "forward_endpoint": (
            "tab:orange",
            "v",
            r"forward endpoint $\theta_\gamma=0$",
        ),
    }

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 10.5))
    for category, (color, marker, label) in styles.items():
        selected = [
            row for row in rows if row["geometry_class"] == category
        ]
        if not selected:
            continue
        abs_t = np.asarray([float(row["abs_t_GeV2"]) for row in selected])
        ratio = np.asarray([
            float(row["transverse_flip_to_conserving_ratio"])
            for row in selected
        ])
        rho = np.asarray([
            float(row["collinearity_rho"]) for row in selected
        ])
        log_ratio = np.log10(ratio)
        axes[0].scatter(
            abs_t,
            log_ratio,
            s=34,
            marker=marker,
            color=color,
            alpha=0.78,
            edgecolors="none",
            label=label,
        )
        finite = np.isfinite(rho) & (rho > 0.0)
        axes[1].scatter(
            rho[finite],
            log_ratio[finite],
            s=34,
            marker=marker,
            color=color,
            alpha=0.78,
            edgecolors="none",
            label=label,
        )

    formula = (
        r"$R_C^T=\sum|C_T|^2_{h_{\rm out}\ne h_{\rm in}}"
        r"/\sum|C_T|^2_{h_{\rm out}=h_{\rm in}}$"
    )
    axes[0].set_title("Compton lepton-helicity flip versus conservation")
    axes[0].set_xlabel(r"$|t|$ [GeV$^2$]")
    axes[0].set_ylabel(r"$\log_{10} R_C^T$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[0].text(
        0.02,
        0.03,
        formula,
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    axes[1].set_xscale("log")
    axes[1].set_xlabel(
        r"$\rho_{\rm coll}=2\min(|k\!\cdot\!q_\gamma|,"
        r"|k'\!\cdot\!q_\gamma|)/m_e^2$"
    )
    axes[1].set_ylabel(r"$\log_{10} R_C^T$")
    axes[1].set_title(
        "The endpoint-collinearity regulator, rather than t alone, "
        "controls the flip enhancement"
    )
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    """Parse command-line paths and plot-only mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse --data and regenerate only the PDF.",
    )
    return parser.parse_args()


def main():
    """Calculate the canonical GHZ ratios and save CSV/PDF artifacts."""
    args = parse_args()
    if args.plot_only:
        rows = load_numeric_rows(args.data)
    else:
        rows = [
            compton_ratio_row(canonical, detail)
            for canonical, detail in canonical_configuration_rows(
                args.input_root
            )
        ]
        rows.sort(key=lambda row: float(row["abs_t_GeV2"]))
        write_csv(args.data, rows)
        write_csv(args.summary, summary_rows(rows))
    save_plot(rows, args.plot)
    print_console_text(
        f"Calculated configurations: {len(rows)}\n"
        f"Ratio CSV: {args.data}\n"
        f"Summary CSV: {args.summary}\n"
        f"Plot PDF: {args.plot}\n"
    )


if __name__ == "__main__":
    main()
