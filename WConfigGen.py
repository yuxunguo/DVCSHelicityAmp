"""Generate representative low-D_W configurations from WScan output."""

import csv
import math
from pathlib import Path

import numpy as np

from AlignmentScan import ALIGNMENT_SPIN_CASES
from config import ELECTRON_MASS_GEV, PROTON_MASS_GEV
from ConfigGen import kinematics_from_config_row, plot_momentum_panels
from FormFactors import yahl_dirac_pauli_from_t
from PlotUtils import bin_edges_from_values, require_matplotlib
from SpinDensityMat import (
    amplitude_table,
    final_state_ensemble,
    outgoing_spin_states,
    spin_density_observables_from_amplitudes,
)
from WScan import FULL_CSV as W_SCAN_CSV


OUTPUT_DIR = Path("Output") / "WConfigGen"
SUMMARY_CSV = OUTPUT_DIR / "w_configuration_summary.csv"
MOMENTA_CSV = OUTPUT_DIR / "w_momentum_configurations.csv"
AMPLITUDES_CSV = OUTPUT_DIR / "w_amplitude_decomposition.csv"
LOG_PATH = Path("Output") / "WConfigGen.log"

TOP_CANDIDATES_PER_ENERGY = 80
MAX_CLUSTERS_PER_ENERGY = 4
CLUSTER_RADIUS = 0.42
AMPLITUDE_MIN_FRACTION = 0.02
AMPLITUDE_MAX_COMPONENTS = 8


def read_w_rows(path=W_SCAN_CSV):
    """Read the full WScan phase-space table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run WScan.py first.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _circular_distance(a, b):
    delta = abs(float(a) - float(b)) % (2.0 * np.pi)
    return min(delta, 2.0 * np.pi - delta)


def _row_distance(first, second):
    return math.hypot(
        _circular_distance(first["phi_in_electron"], second["phi_in_electron"]),
        _circular_distance(first["phiOut"], second["phiOut"]),
    )


def cluster_low_distance_rows(rows, distance_column):
    """Cluster the strongest low-D_W candidates on the angular torus."""
    candidates = sorted(rows, key=lambda row: float(row[distance_column]))[
        :TOP_CANDIDATES_PER_ENERGY
    ]
    clusters = []
    for row in candidates:
        nearest = None
        nearest_distance = np.inf
        for index, cluster in enumerate(clusters):
            distance = _row_distance(row, cluster["best"])
            if distance < nearest_distance:
                nearest, nearest_distance = index, distance
        if nearest is not None and nearest_distance <= CLUSTER_RADIUS:
            clusters[nearest]["rows"].append(row)
            if float(row[distance_column]) < float(clusters[nearest]["best"][distance_column]):
                clusters[nearest]["best"] = row
        elif len(clusters) < MAX_CLUSTERS_PER_ENERGY:
            clusters.append({"best": row, "rows": [row]})
        else:
            nearest = min(
                range(len(clusters)),
                key=lambda index: _row_distance(row, clusters[index]["best"]),
            )
            clusters[nearest]["rows"].append(row)
    for cluster in clusters:
        cluster["rows"].sort(key=lambda row: float(row[distance_column]))
        cluster["best"] = cluster["rows"][0]
    return clusters


def select_configurations(rows):
    """Select clustered D_W minima for every polarization and photon energy."""
    selected = []
    for prefix, label, spin_case in ALIGNMENT_SPIN_CASES:
        distance_column = f"{prefix}_D_W"
        energy_groups = {}
        for row in rows:
            energy_groups.setdefault(row["qOut_regime"], []).append(row)
        for energy_name, energy_rows in energy_groups.items():
            clusters = cluster_low_distance_rows(energy_rows, distance_column)
            for cluster_id, cluster in enumerate(clusters):
                source = cluster["best"]
                selected.append({
                    **source,
                    "detail_id": f"{prefix}_{energy_name}_W_{cluster_id}",
                    "polarization": prefix,
                    "polarization_label": label,
                    "spin_case": spin_case,
                    "energy_group": energy_name,
                    "cluster_id": cluster_id,
                    "cluster_size": len(cluster["rows"]),
                    "D_W": source[distance_column],
                    "purity": source[f"{prefix}_purity"],
                    "C_e_p": source[f"{prefix}_C_e_p"],
                    "C_p_gamma": source[f"{prefix}_C_p_gamma"],
                    "C_e_gamma": source[f"{prefix}_C_e_gamma"],
                })
    return selected


def _w_state():
    states = outgoing_spin_states()
    state = np.zeros(len(states), dtype=complex)
    for labels in ((+1, -1, -1), (-1, +1, -1), (-1, -1, +1)):
        state[states.index(labels)] = 1.0 / np.sqrt(3.0)
    return state


def evaluate_configuration(row):
    """Reconstruct one selected point and its W fidelity/amplitude records."""
    kin = kinematics_from_config_row(row)
    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], PROTON_MASS_GEV)
    amplitudes = amplitude_table(
        kin["momenta"], PROTON_MASS_GEV, F1, F2,
        electron_mass=ELECTRON_MASS_GEV,
    )
    spin_data = spin_density_observables_from_amplitudes(
        amplitudes, spin_case=row["spin_case"], normalize_trace=True,
    )
    rho = spin_data["rho"]
    w_state = _w_state()
    w_fidelity = float(np.real(np.vdot(w_state, rho @ w_state)))
    w_indices = np.flatnonzero(abs(w_state) > 0.0)
    w_subspace_population = float(np.real(np.trace(rho[np.ix_(w_indices, w_indices)])))

    components = []
    total = float(spin_data["spin_signal"])
    for initial in final_state_ensemble(amplitudes, row["spin_case"]):
        for labels, amplitude in zip(outgoing_spin_states(), initial["state"]):
            weighted_abs2 = float(initial["weight"] * abs(amplitude) ** 2)
            components.append({
                "detail_id": row["detail_id"],
                "polarization": row["polarization"],
                "polarization_label": row["polarization_label"],
                "energy_group": row["energy_group"],
                "initial_component": initial["label"],
                "ensemble_weight": initial["weight"],
                "h_e": labels[0], "h_p": labels[1], "h_gamma": labels[2],
                "amplitude_real": amplitude.real,
                "amplitude_imag": amplitude.imag,
                "amplitude_abs": abs(amplitude),
                "amplitude_phase_rad": np.angle(amplitude),
                "weighted_abs2": weighted_abs2,
                "fraction": weighted_abs2 / total,
            })
    components.sort(key=lambda item: item["fraction"], reverse=True)
    for rank, item in enumerate(components, start=1):
        item["decomposition_rank"] = rank
    leading_components = [
        item for item in components if item["fraction"] >= AMPLITUDE_MIN_FRACTION
    ][:AMPLITUDE_MAX_COMPONENTS]

    return {
        "kinematics": kin,
        "F1": F1,
        "F2": F2,
        "w_fidelity": w_fidelity,
        "w_subspace_population": w_subspace_population,
        "all_components": components,
        "components": leading_components,
    }


def _write_dict_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        if headers:
            writer.writeheader()
            writer.writerows(rows)
    return path


def build_output_rows(selected):
    """Evaluate selected configurations and build the three output tables."""
    summaries = []
    momenta = []
    amplitudes = []
    evaluated = []
    for row in selected:
        result = evaluate_configuration(row)
        kin = result["kinematics"]
        summary = {
            "detail_id": row["detail_id"],
            "polarization": row["polarization"],
            "polarization_label": row["polarization_label"],
            "energy_group": row["energy_group"],
            "cluster_id": row["cluster_id"],
            "cluster_size": row["cluster_size"],
            "D_W": row["D_W"],
            "w_fidelity": result["w_fidelity"],
            "w_subspace_population": result["w_subspace_population"],
            "purity": row["purity"],
            "C_e_p": row["C_e_p"],
            "C_p_gamma": row["C_p_gamma"],
            "C_e_gamma": row["C_e_gamma"],
            **{name: row[name] for name in (
                "kinematic_point", "s", "theta_in", "phi_in",
                "phi_in_electron", "qOut", "phiOut", "Q2", "xB", "t",
                "theta_e_gamma_deg", "aligned",
            )},
            "F1": result["F1"],
            "F2": result["F2"],
        }
        summaries.append(summary)
        for name in ("k", "p", "kp", "pp", "qout"):
            vector = kin["momenta"][name]
            momenta.append({
                "detail_id": row["detail_id"],
                "polarization": row["polarization"],
                "polarization_label": row["polarization_label"],
                "momentum": name,
                "E": vector[0], "px": vector[1], "py": vector[2], "pz": vector[3],
                "mass_shell": vector[0] ** 2 - np.dot(vector[1:4], vector[1:4]),
            })
        amplitudes.extend(result["all_components"])
        evaluated.append((row, result))
    return summaries, momenta, amplitudes, evaluated


def _distance_grid(rows, column):
    x = np.unique([float(row["phi_in_electron"]) for row in rows])
    y = np.unique([float(row["phiOut"]) for row in rows])
    grid = np.full((len(y), len(x)), np.nan)
    xi = {value: index for index, value in enumerate(x)}
    yi = {value: index for index, value in enumerate(y)}
    for row in rows:
        grid[yi[float(row["phiOut"])], xi[float(row["phi_in_electron"])]] = float(row[column])
    return x, y, grid


def _plot_overview(plt, pdf, rows, selected, prefix, label):
    energy_groups = {}
    for row in rows:
        energy_groups.setdefault(row["qOut_regime"], []).append(row)
    fig, axes = plt.subplots(1, len(energy_groups), figsize=(5 * len(energy_groups), 4.5),
                             squeeze=False, constrained_layout=True)
    image = None
    for ax, (energy_name, energy_rows) in zip(axes[0], energy_groups.items()):
        x, y, grid = _distance_grid(energy_rows, f"{prefix}_D_W")
        image = ax.pcolormesh(bin_edges_from_values(x), bin_edges_from_values(y), grid,
                              shading="auto", cmap="viridis_r", vmin=0,
                              vmax=2 / np.sqrt(3))
        for row in selected:
            if row["polarization"] == prefix and row["energy_group"] == energy_name:
                ax.plot(float(row["phi_in_electron"]), float(row["phiOut"]),
                        marker="*", color="red", markersize=8)
        ax.set_title(energy_name)
        ax.set_xlabel(r"$\phi_{e,\mathrm{in}}$")
        ax.set_ylabel(r"$\phi_\gamma$")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=r"$D_W$")
    fig.suptitle(f"W-configuration regions: {label}")
    pdf.savefig(fig)
    plt.close(fig)


def _plot_detail(plt, pdf, row, result):
    kin = result["kinematics"]
    fig = plt.figure(figsize=(13.2, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.05, 1.25))
    plot_momentum_panels(fig, grid, kin)
    text_ax = fig.add_subplot(grid[0, 1:])
    text_ax.axis("off")
    text = [
        f"{row['detail_id']}",
        f"polarization: {row['polarization_label']}",
        f"D_W={float(row['D_W']):.8g}",
        f"C_ep={float(row['C_e_p']):.8g}, C_pgamma={float(row['C_p_gamma']):.8g}, "
        f"C_egamma={float(row['C_e_gamma']):.8g}",
        f"purity={float(row['purity']):.8g}",
        f"W fidelity={result['w_fidelity']:.8g}",
        f"W-subspace population={result['w_subspace_population']:.8g}",
        "",
        f"s={kin['s']:.7g}, theta_in={kin['theta_in']:.7g}, phi_in={kin['phi_in']:.7g}",
        f"E_gamma={kin['qOut']:.7g}, phi_gamma={kin['phiOut']:.7g}",
        f"Q2={kin['Q2']:.7g}, xB={kin['xB']:.7g}, t={kin['t']:.7g}",
    ]
    text_ax.text(0, 1, "\n".join(text), va="top", family="monospace", fontsize=10)

    amp_ax = fig.add_subplot(grid[1, 1:])
    components = result["components"]
    labels = [
        rf"$h_e={item['h_e']:+d},\ h_p={item['h_p']:+d},\ "
        rf"h_\gamma={item['h_gamma']:+d}$" + "\n"
        f"{item['initial_component']}" for item in components
    ]
    fractions = [item["fraction"] for item in components]
    positions = np.arange(len(components))
    bars = amp_ax.barh(positions, fractions, color="tab:blue", alpha=0.75)
    amp_ax.set_yticks(positions, labels)
    amp_ax.invert_yaxis()
    amp_ax.set_xlabel("ensemble-weighted |A|^2 fraction")
    amp_ax.set_title("Leading helicity components and complex phases")
    amp_ax.set_xlim(0.0, max(0.08, 1.38 * max(fractions)))
    for bar, item in zip(bars, components):
        amp_ax.text(
            bar.get_width(),
            bar.get_y() + 0.5 * bar.get_height(),
            (
                rf"  $\arg A={item['amplitude_phase_rad']:+.3f}$ rad, "
                rf"$\mathrm{{Re}}A={item['amplitude_real']:+.2e}$, "
                rf"$\mathrm{{Im}}A={item['amplitude_imag']:+.2e}$"
            ),
            va="center",
            fontsize=7.5,
        )
    amp_ax.grid(axis="x", alpha=0.25)
    fig.suptitle("WScan selected configuration", fontsize=15)
    pdf.savefig(fig)
    plt.close(fig)


def write_pdfs(rows, selected, evaluated):
    """Write one overview/detail PDF for each polarization."""
    plt, PdfPages = require_matplotlib()
    paths = []
    evaluated_by_id = {row["detail_id"]: result for row, result in evaluated}
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        polarization_rows = [row for row in selected if row["polarization"] == prefix]
        path = OUTPUT_DIR / prefix / "w_configurations.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(path) as pdf:
            _plot_overview(plt, pdf, rows, selected, prefix, label)
            for row in sorted(polarization_rows, key=lambda item: float(item["D_W"])):
                _plot_detail(plt, pdf, row, evaluated_by_id[row["detail_id"]])
        paths.append(path)
    return paths


def build_report(summaries, pdf_paths):
    """Return a concise WConfigGen report."""
    lines = [
        "WScan configuration generator",
        f"  input: {W_SCAN_CSV}",
        f"  selected configurations: {len(summaries)}",
        f"  polarization PDFs: {len(pdf_paths)}",
        "  best configuration per polarization:",
    ]
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        group = [row for row in summaries if row["polarization"] == prefix]
        best = min(group, key=lambda row: float(row["D_W"]))
        lines.append(
            f"    {label}: D_W={float(best['D_W']):.8g}, "
            f"W fidelity={float(best['w_fidelity']):.8g}, "
            f"qOut={float(best['qOut']):.6g}, "
            f"phi_e={float(best['phi_in_electron']):.6g}, "
            f"phi_gamma={float(best['phiOut']):.6g}"
        )
    lines.extend((
        f"  summary csv: {SUMMARY_CSV}",
        f"  momentum csv: {MOMENTA_CSV}",
        f"  amplitude csv: {AMPLITUDES_CSV}",
    ))
    return "\n".join(lines) + "\n"


def main():
    """Generate clustered WScan configurations and visualization packages."""
    rows = read_w_rows()
    selected = select_configurations(rows)
    summaries, momenta, amplitudes, evaluated = build_output_rows(selected)
    _write_dict_csv(SUMMARY_CSV, summaries)
    _write_dict_csv(MOMENTA_CSV, momenta)
    _write_dict_csv(AMPLITUDES_CSV, amplitudes)
    pdf_paths = write_pdfs(rows, selected, evaluated)
    report = build_report(summaries, pdf_paths)
    LOG_PATH.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
