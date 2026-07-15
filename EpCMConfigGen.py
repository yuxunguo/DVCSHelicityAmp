"""Generate configuration packages from :mod:`EpCMEntanglementScan` results.

For each selected observable and incoming polarization, the generator finds
separated optimum regions in the ``(z, theta_cm)`` scan.  It reconstructs the
exact ep-CM momenta and writes representative configurations, momentum tables,
ensemble-aware final-helicity amplitude decompositions, and region heatmaps.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import math
from pathlib import Path

import numpy as np

from AlignmentScan import (
    COARSE_CONCURRENCE_NAMES,
    explicit_polarization_name,
    species_observable_name,
)
from EpCMEntanglementScan import (
    FULL_CSV as SCAN_CSV,
    LEPTON_MASS_GEV,
    ep_cm_momenta,
)
from FormFactors import yahl_dirac_pauli_from_t
from PlotUtils import print_console_text, require_matplotlib
from SpinDensityMat import (
    SPIN_CASES,
    amplitude_table,
    contract_initial_state,
    final_state_ensemble,
    outgoing_spin_states,
    process_density_matrix_from_amplitudes,
    spin_case_display_label,
)
from config import PROTON_MASS_GEV, SCAN_WORKERS


# Editable configuration-selection controls.
CONFIG_TARGETS = (
    ("C_e_p", "c_lepton_proton", False),
    ("C_e_gamma", "c_lepton_gamma", False),
    ("C_p_gamma", "c_proton_gamma", False),
    ("C_e_rest", "c_lepton_rest", False),
    ("C_p_rest", "c_proton_rest", False),
    ("C_gamma_rest", "c_gamma_rest", False),
    ("M_e", "m_lepton", False),
    ("M_p", "m_proton", False),
    ("M_gamma", "m_gamma", False),
    ("F3", "f3", False),
    ("GHZ_purity", "ghz_purity", False),
    ("W_purity", "w_purity", False),
    ("D_W", "dw", True),
)
TOP_CANDIDATES_PER_SPIN = 80
MAX_REGIONS_PER_SPIN = 3
REGION_SEPARATION = 0.12
AMPLITUDE_MIN_FRACTION = 0.02
AMPLITUDE_MAX_COMPONENTS = 8
CONFIGGEN_KINEMATIC_WORKERS = SCAN_WORKERS
CONFIGGEN_PLOT_WORKERS = max(1, min(SCAN_WORKERS, len(CONFIG_TARGETS)))

OUTPUT_DIR = Path("Output") / "EpCMConfigGen"
DATA_DIR = OUTPUT_DIR / "Data"
PLOT_DIR = OUTPUT_DIR / "Plots"
LOG_PATH = OUTPUT_DIR / "EpCMConfigGen.log"


def polarization_prefix(spin_case):
    """Return the exact heavy-lepton polarization label used by AlignmentScan."""
    return explicit_polarization_name(spin_case, "heavy")


def observable_column(spin_case, observable):
    """Return the focused scan's AlignmentScan-compatible observable column."""
    return (
        f"{polarization_prefix(spin_case)}_"
        f"{species_observable_name(observable, 'heavy')}"
    )


def read_scan_rows(path=SCAN_CSV):
    """Read and numerically parse the focused ep-CM scan."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python3 EpCMEntanglementScan.py first."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Focused scan CSV is empty: {path}")
    required = {
        observable_column(spin_case, observable)
        for spin_case in SPIN_CASES
        for observable, _tag, _minimized in CONFIG_TARGETS
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            "The focused scan CSV uses an obsolete or incomplete schema; "
            f"missing {len(missing)} AlignmentScan-compatible columns "
            f"(for example: {preview}). Rerun python3 EpCMEntanglementScan.py "
            "before running EpCMConfigGen.py."
        )
    numeric_rows = []
    for source in rows:
        row = {}
        for key, value in source.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = value
        numeric_rows.append(row)
    return numeric_rows


def write_rows(path, rows):
    """Write a dictionary CSV, including an empty file for no records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if headers:
            writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return path


def normalized_distance(first, second, z_span, theta_span):
    """Return scan-coordinate distance normalized by the sampled ranges."""
    dz = (first["z"] - second["z"]) / max(z_span, 1.0e-15)
    dt = (
        first["theta_cm_rad"] - second["theta_cm_rad"]
    ) / max(theta_span, 1.0e-15)
    return float(np.hypot(dz, dt))


def select_regions(rows, key, minimized):
    """Select separated local absolute extrema, then fill from global ranks."""
    finite = [row for row in rows if np.isfinite(float(row.get(key, np.nan)))]
    score = lambda row: abs(float(row[key]))
    finite.sort(key=score, reverse=not minimized)
    point_map = {
        (int(row["z_index"]), int(row["theta_index"])): row
        for row in finite
    }
    local_extrema = []
    for (z_index, theta_index), row in point_map.items():
        neighbors = [
            point_map.get((z_index + dz, theta_index + dt))
            for dz in (-1, 0, 1)
            for dt in (-1, 0, 1)
            if dz != 0 or dt != 0
        ]
        neighbor_scores = [score(item) for item in neighbors if item is not None]
        if not neighbor_scores:
            continue
        value = score(row)
        is_extremum = (
            value <= min(neighbor_scores) if minimized
            else value >= max(neighbor_scores)
        )
        is_strict = (
            value < max(neighbor_scores) if minimized
            else value > min(neighbor_scores)
        )
        if is_extremum and is_strict:
            local_extrema.append(row)
    local_extrema.sort(key=score, reverse=not minimized)
    seen = {id(row) for row in local_extrema}
    candidates = (
        local_extrema
        + [row for row in finite if id(row) not in seen]
    )[:TOP_CANDIDATES_PER_SPIN]
    if not candidates:
        return []
    z_values = np.asarray([row["z"] for row in rows], dtype=float)
    theta_values = np.asarray([row["theta_cm_rad"] for row in rows], dtype=float)
    z_span = float(np.ptp(z_values))
    theta_span = float(np.ptp(theta_values))
    selected = []
    for row in candidates:
        if all(
            normalized_distance(row, other, z_span, theta_span) >= REGION_SEPARATION
            for other in selected
        ):
            selected.append(row)
        if len(selected) == MAX_REGIONS_PER_SPIN:
            break
    return selected


def configuration_record(row, spin_case, observable, region_index):
    """Return common metadata for one representative configuration."""
    key = observable_column(spin_case, observable)
    return {
        "detail_id": f"{observable}_{spin_case}_region_{region_index}",
        "selected_observable": observable,
        "selected_spin_case": spin_case,
        "selected_spin_label": spin_case_display_label(spin_case),
        "region": region_index,
        "selected_value": row[key],
        "selected_abs_value": abs(row[key]),
        "z": row["z"],
        "theta_cm_rad": row["theta_cm_rad"],
        "mu": row["mu"],
        "p_cm_GeV": row["p_cm_GeV"],
        "sqrt_s_GeV": row["sqrt_s_GeV"],
        "subsystem_mass_GeV": row["subsystem_mass_GeV"],
        "t_GeV2": row["t_GeV2"],
    }


def momentum_records(config):
    """Reconstruct and serialize all five external four-momenta."""
    kin = ep_cm_momenta(config["z"], config["theta_cm_rad"])
    records = []
    labels = {"k": "l", "p": "P", "kp": "l'", "pp": "P'", "qout": "q_gamma"}
    for name in ("k", "p", "kp", "pp", "qout"):
        vector = kin["momenta"][name]
        records.append({
            **config,
            "momentum": name,
            "particle_label": labels[name],
            "E": vector[0],
            "px": vector[1],
            "py": vector[2],
            "pz": vector[3],
            "p_abs": np.linalg.norm(vector[1:]),
            "mass2": vector[0] ** 2 - np.dot(vector[1:], vector[1:]),
        })
    virtual = kin["virtual_photon"]
    records.append({
        **config,
        "momentum": "q_virtual",
        "particle_label": "q_virtual",
        "E": virtual[0],
        "px": virtual[1],
        "py": virtual[2],
        "pz": virtual[3],
        "p_abs": np.linalg.norm(virtual[1:]),
        "mass2": kin["t"],
    })
    return records


def explicit_initial_component(label):
    """Return explicit incoming lepton/proton state labels for CSV output."""
    pieces = str(label).split(", ")
    lepton = pieces[0].removeprefix("electron ")
    proton = pieces[1].removeprefix("proton ") if len(pieces) > 1 else "unknown"
    lepton_value = lepton.removeprefix("h=")
    proton_value = proton.removeprefix("h=")
    return {
        "incoming_lepton_state": lepton_value,
        "incoming_proton_state": proton_value,
        "initial_component": f"h_l={lepton_value}, h_p={proton_value}",
    }


def amplitude_records(config):
    """Return the leading ensemble-weighted final-helicity components."""
    kin = ep_cm_momenta(config["z"], config["theta_cm_rad"])
    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], PROTON_MASS_GEV)
    amplitudes = amplitude_table(
        kin["momenta"],
        PROTON_MASS_GEV,
        F1,
        F2,
        electron_mass=LEPTON_MASS_GEV,
    )
    process_rho = process_density_matrix_from_amplitudes(amplitudes)
    contracted = contract_initial_state(process_rho, config["selected_spin_case"])
    total = float(np.real_if_close(np.trace(contracted), tol=1000).real)
    if total <= 0.0 or not np.isfinite(total):
        raise ZeroDivisionError(f"Invalid amplitude norm for {config['detail_id']}.")
    records = []
    ensemble = final_state_ensemble(amplitudes, config["selected_spin_case"])
    for component in ensemble:
        for out_index, (labels, amplitude) in enumerate(
            zip(outgoing_spin_states(), component["state"])
        ):
            weighted_abs2 = float(component["weight"] * abs(amplitude) ** 2)
            h_out, s_out, photon_helicity = labels
            records.append({
                **config,
                **explicit_initial_component(component["label"]),
                "ensemble_weight": component["weight"],
                "out_index": out_index,
                "h_l": h_out,
                "h_p": s_out,
                "h_gamma": photon_helicity,
                "amplitude_real": amplitude.real,
                "amplitude_imag": amplitude.imag,
                "amplitude_abs": abs(amplitude),
                "amplitude_phase": np.angle(amplitude),
                "weighted_abs2": weighted_abs2,
                "fraction": weighted_abs2 / total,
            })
    records.sort(key=lambda record: record["fraction"], reverse=True)
    retained = [
        record for record in records
        if record["fraction"] >= AMPLITUDE_MIN_FRACTION
    ][:AMPLITUDE_MAX_COMPONENTS]
    retained_fraction = sum(record["fraction"] for record in retained)
    for rank, record in enumerate(retained, start=1):
        record["rank"] = rank
        record["retained_fraction_total"] = retained_fraction
    return retained


def _amplitude_records_worker(config):
    """Process-pool boundary for one independent configuration."""
    return amplitude_records(config)


def parallel_amplitude_records(configurations):
    """Evaluate configuration amplitudes with AlignmentScan-style batching."""
    tasks = list(configurations)
    if not tasks:
        return []
    workers = min(max(1, int(CONFIGGEN_KINEMATIC_WORKERS)), len(tasks))
    if workers == 1:
        results = [_amplitude_records_worker(config) for config in tasks]
    else:
        chunksize = max(1, math.ceil(len(tasks) / (4 * workers)))
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(
                    _amplitude_records_worker,
                    tasks,
                    chunksize=chunksize,
                ))
        except (OSError, PermissionError):
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_amplitude_records_worker, tasks))
    return [record for result in results for record in result]


def build_target(target, rows):
    """Build combined and per-polarization CSV packages for one target."""
    observable, tag, minimized = target
    configurations = []
    momenta = []
    selected_by_spin = {}
    configs_by_spin = {}
    momenta_by_spin = {}
    for spin_case in SPIN_CASES:
        key = observable_column(spin_case, observable)
        if key not in rows[0]:
            raise KeyError(f"Missing required scan column {key!r}.")
        selected = select_regions(rows, key, minimized)
        selected_by_spin[spin_case] = selected
        spin_configs = [
            configuration_record(row, spin_case, observable, region_index)
            for region_index, row in enumerate(selected, start=1)
        ]
        spin_momenta = [
            record for config in spin_configs for record in momentum_records(config)
        ]
        configs_by_spin[spin_case] = spin_configs
        momenta_by_spin[spin_case] = spin_momenta
        configurations.extend(spin_configs)
        momenta.extend(spin_momenta)

    amplitudes = parallel_amplitude_records(configurations)
    for spin_case in SPIN_CASES:
        base = DATA_DIR / tag / polarization_prefix(spin_case)
        spin_amplitudes = [
            record for record in amplitudes
            if record["selected_spin_case"] == spin_case
        ]
        write_rows(base / "configuration_examples.csv", configs_by_spin[spin_case])
        write_rows(base / "momentum_configurations.csv", momenta_by_spin[spin_case])
        write_rows(base / "final_state_amplitude_decomposition.csv", spin_amplitudes)

    combined = DATA_DIR / tag / "combined"
    write_rows(combined / "configuration_examples.csv", configurations)
    write_rows(combined / "momentum_configurations.csv", momenta)
    write_rows(combined / "final_state_amplitude_decomposition.csv", amplitudes)
    return {
        "target": target,
        "configurations": configurations,
        "selected_by_spin": selected_by_spin,
        "configs_by_spin": configs_by_spin,
        "momenta": momenta,
        "amplitudes": amplitudes,
    }


def scan_grid(rows, key):
    """Return sorted scan axes and a rectangular observable grid."""
    theta_values = np.unique([row["theta_cm_rad"] for row in rows])
    z_values = np.unique([row["z"] for row in rows])
    mu_for_z = {
        z: next(row["mu"] for row in rows if row["z"] == z)
        for z in z_values
    }
    mu_values = np.asarray([mu_for_z[z] for z in z_values])
    theta_index = {value: index for index, value in enumerate(theta_values)}
    z_index = {value: index for index, value in enumerate(z_values)}
    grid = np.full((len(z_values), len(theta_values)), np.nan)
    for row in rows:
        grid[z_index[row["z"]], theta_index[row["theta_cm_rad"]]] = row[key]
    return theta_values, mu_values, grid


def target_plot_path(target, spin_case):
    """Return the AlignmentScan-style polarization/observable PDF path."""
    observable, tag, minimized = target
    direction = "min" if minimized else "max_abs"
    return PLOT_DIR / polarization_prefix(spin_case) / f"{direction}_{tag}_regions.pdf"


def _perpendicular(vector):
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        return np.array((0.0, 1.0))
    return np.array((-vector[1], vector[0])) / norm


def _draw_arrow(ax, start, end, color, linestyle="-", linewidth=1.7):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops={
            "arrowstyle": "->", "color": color, "linewidth": linewidth,
            "linestyle": linestyle, "shrinkA": 0.0, "shrinkB": 0.0,
        },
    )


def _draw_wavy_photon(ax, start, end, color, amplitude):
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-14:
        return
    parameter = np.linspace(0.0, 1.0, 160)
    normal = _perpendicular(delta)
    wave = (
        start
        + parameter[:, None] * delta
        + amplitude * np.sin(12.0 * np.pi * parameter)[:, None] * normal
    )
    ax.plot(wave[:, 0], wave[:, 1], color=color, linewidth=1.6)
    _draw_arrow(ax, wave[-10], end, color, linewidth=1.3)


def plot_momentum_configuration(ax, config):
    """Plot styled ep-CM particle trajectories in the x--z plane."""
    kin = ep_cm_momenta(config["z"], config["theta_cm_rad"])
    styles = {
        "k": (r"$\ell$", "tab:blue", "lepton", True),
        "p": (r"$P$", "tab:orange", "proton", True),
        "kp": (r"$\ell'$", "tab:cyan", "lepton", False),
        "pp": (r"$P'$", "tab:red", "proton", False),
        "qout": (r"$q_\gamma$", "tab:green", "photon", False),
    }
    momentum_scale = max(
        np.linalg.norm(kin["momenta"][name][[1, 3]]) for name in styles
    )
    for name, (label, color, kind, incoming) in styles.items():
        vector = kin["momenta"][name]
        spatial = np.asarray((vector[1], vector[3]), dtype=float)
        start, end = (-spatial, np.zeros(2)) if incoming else (np.zeros(2), spatial)
        if kind == "photon":
            _draw_wavy_photon(ax, start, end, color, amplitude=0.018 * momentum_scale)
        else:
            _draw_arrow(
                ax, start, end, color,
                linestyle="--" if kind == "lepton" else "-",
                linewidth=1.7,
            )
        label_point = start if incoming else end
        ax.text(
            label_point[0], label_point[1], f" {label}",
            color=color, fontsize=12, va="center",
        )
    limit = 1.12 * max(
        np.linalg.norm(kin["momenta"][name][[1, 3]])
        for name in styles
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axhline(0.0, color="0.65", linewidth=0.5)
    ax.axvline(0.0, color="0.65", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$p_x$ [GeV]")
    ax.set_ylabel(r"$p_z$ [GeV]")
    ax.set_title("Momentum configuration in the ep CM frame")


def plot_kinematic_text(ax, config):
    """Display exact kinematics and selected-extremum metadata."""
    kin = ep_cm_momenta(config["z"], config["theta_cm_rad"])
    momenta = kin["momenta"]
    lines = [
        f"detail: {config['detail_id']}",
        f"observable: {config['selected_observable']}",
        f"polarization: {config['selected_spin_label']}",
        f"raw value: {config['selected_value']:.10g}",
        f"absolute value: {config['selected_abs_value']:.10g}",
        f"z = {config['z']:.10g}",
        f"theta_cm = {config['theta_cm_rad']:.10g} rad",
        f"mu = {config['mu']:.10g}",
        f"sqrt(s) = {config['sqrt_s_GeV']:.10g} GeV",
        f"t = {config['t_GeV2']:.10g} GeV^2",
        f"W(q+l) = {config['subsystem_mass_GeV']:.10g} GeV",
        "",
    ]
    labels = {
        "k": r"l ", "p": "P ", "kp": r"l'", "pp": "P'", "qout": "q_gamma",
    }
    for name in ("k", "p", "kp", "pp", "qout"):
        vector = momenta[name]
        lines.append(
            f"{labels[name]:2s}: ({vector[0]:.6g}, {vector[1]:.6g}, "
            f"{vector[2]:.6g}, {vector[3]:.6g})"
        )
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("Kinematics and four-momenta")


def plot_amplitude_components(ax, config, amplitudes):
    """Plot leading ensemble-weighted outgoing helicity components."""
    records = [
        record for record in amplitudes
        if record["detail_id"] == config["detail_id"]
    ]
    records.sort(key=lambda record: record["fraction"], reverse=True)
    if not records:
        ax.text(0.5, 0.5, "No retained amplitude components", ha="center", va="center")
        ax.axis("off")
        return
    labels = [
        rf"$h_\ell={int(record['h_l']):+d},\ h_p={int(record['h_p']):+d},$" + "\n"
        rf"$h_\gamma={int(record['h_gamma']):+d}$" + "\n"
        + record["initial_component"]
        for record in records
    ]
    fractions = np.asarray([record["fraction"] for record in records])
    phases = np.asarray([record["amplitude_phase"] for record in records])
    colors = require_matplotlib()[0].cm.twilight((phases + np.pi) / (2.0 * np.pi))
    positions = np.arange(len(records))
    ax.bar(positions, fractions, color=colors)
    ax.set_xticks(positions, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ensemble-weighted fraction")
    ax.set_ylim(0.0, max(1.05 * fractions.max(), 0.05))
    ax.set_title("Leading final-helicity amplitudes; color encodes phase")
    ax.grid(axis="y", alpha=0.25)


def save_polarization_target_plot(package, rows, spin_case):
    """Write a scan page and detailed configuration pages for one case."""
    observable, tag, minimized = package["target"]
    output = target_plot_path(package["target"], spin_case)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt, PdfPages = require_matplotlib()
    with PdfPages(output) as pdf:
        key = observable_column(spin_case, observable)
        theta_values, mu_values, grid = scan_grid(rows, key)
        grid = np.abs(grid)
        fig, ax = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True)
        image = ax.pcolormesh(
            theta_values, mu_values, grid, shading="auto",
            cmap="viridis_r" if minimized else "viridis",
            vmin=0.0,
        )
        selected = package["selected_by_spin"][spin_case]
        for region, row in enumerate(selected, start=1):
            ax.scatter(
                row["theta_cm_rad"], row["mu"], marker="o", s=90,
                facecolors="none", edgecolors="red", linewidths=1.5,
            )
            ax.annotate(str(region), (row["theta_cm_rad"], row["mu"]), color="red")
        ax.axvspan(2.4, 2.5, color="white", alpha=0.1)
        ax.axhline(22.0, color="white", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$\theta_{\gamma\ell}^{(q\ell\,\mathrm{CM})}$ [rad]")
        ax.set_ylabel(r"$\mu=p_\ell^{\rm CM}/m_\ell$")
        ax.set_title(
            f"|{species_observable_name(observable, 'heavy')}|: "
            f"{spin_case_display_label(spin_case)}\n"
            f"red circles = local {'minima' if minimized else 'maxima'}"
        )
        fig.colorbar(image, ax=ax, label=f"|{observable}|")
        pdf.savefig(fig)
        plt.close(fig)

        for config in package["configs_by_spin"][spin_case]:
            fig = plt.figure(figsize=(13.0, 8.2), constrained_layout=True)
            grid_spec = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.9))
            momentum_ax = fig.add_subplot(grid_spec[0, 0])
            text_ax = fig.add_subplot(grid_spec[0, 1])
            amplitude_ax = fig.add_subplot(grid_spec[1, :])
            plot_momentum_configuration(momentum_ax, config)
            plot_kinematic_text(text_ax, config)
            plot_amplitude_components(amplitude_ax, config, package["amplitudes"])
            fig.suptitle(
                f"Region {config['region']}: {spin_case_display_label(spin_case)}, "
                f"|{species_observable_name(observable, 'heavy')}|"
            )
            pdf.savefig(fig)
            plt.close(fig)
    return output


_PLOT_WORKER_ROWS = None
_PLOT_WORKER_PACKAGES = None


def _initialize_plot_worker(rows, packages):
    """Load shared scan/package payloads once in each plotting process."""
    global _PLOT_WORKER_ROWS, _PLOT_WORKER_PACKAGES
    _PLOT_WORKER_ROWS = rows
    _PLOT_WORKER_PACKAGES = {
        package["target"][0]: package for package in packages
    }


def _save_target_plot_worker(task):
    observable, spin_case = task
    package = _PLOT_WORKER_PACKAGES[observable]
    key = (observable, spin_case)
    return key, save_polarization_target_plot(package, _PLOT_WORKER_ROWS, spin_case)


def save_target_plots(packages, rows):
    """Save every target/polarization detail PDF in bounded processes."""
    package_by_observable = {
        package["target"][0]: package for package in packages
    }
    tasks = [
        (package["target"][0], spin_case)
        for package in packages
        for spin_case in SPIN_CASES
    ]
    if (
        not CONFIGGEN_PLOT_WORKERS
        or CONFIGGEN_PLOT_WORKERS <= 1
        or len(tasks) == 1
    ):
        return {
            (observable, spin_case): save_polarization_target_plot(
                package_by_observable[observable], rows, spin_case
            )
            for observable, spin_case in tasks
        }
    workers = min(int(CONFIGGEN_PLOT_WORKERS), len(tasks))
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_plot_worker,
            initargs=(rows, packages),
        ) as executor:
            return dict(executor.map(
                _save_target_plot_worker,
                tasks,
                chunksize=1,
            ))
    except (OSError, PermissionError):
        return {
            (observable, spin_case): save_polarization_target_plot(
                package_by_observable[observable], rows, spin_case
            )
            for observable, spin_case in tasks
        }


def build_report(packages, input_path):
    lines = [
        "Focused ep-CM configuration generator",
        f"  input: {input_path}",
        f"  output data: {DATA_DIR}",
        f"  output plots: {PLOT_DIR}",
        f"  regions per target/polarization: up to {MAX_REGIONS_PER_SPIN}",
        f"  minimum normalized region separation: {REGION_SEPARATION}",
        f"  parallel kinematic/amplitude workers: {CONFIGGEN_KINEMATIC_WORKERS}",
        f"  parallel PDF workers: {CONFIGGEN_PLOT_WORKERS}",
        "",
    ]
    for package in packages:
        observable, tag, minimized = package["target"]
        lines.append(
            f"Target |{observable}| "
            f"({'local minimum' if minimized else 'local maximum'}):"
        )
        for spin_case in SPIN_CASES:
            selected = package["selected_by_spin"][spin_case]
            values = ", ".join(
                f"{row[observable_column(spin_case, observable)]:.6g} at "
                f"(z={row['z']:.5g}, theta={row['theta_cm_rad']:.5g}, mu={row['mu']:.5g})"
                for row in selected
            )
            lines.append(f"  {spin_case_display_label(spin_case)}: {values}")
        lines.append("  per-polarization PDFs:")
        for spin_case in SPIN_CASES:
            lines.append(f"    {spin_case}: {target_plot_path(package['target'], spin_case)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def validate_settings():
    configured_observables = {observable for observable, _tag, _min in CONFIG_TARGETS}
    if configured_observables != set(COARSE_CONCURRENCE_NAMES):
        missing = set(COARSE_CONCURRENCE_NAMES) - configured_observables
        extra = configured_observables - set(COARSE_CONCURRENCE_NAMES)
        raise ValueError(
            "CONFIG_TARGETS must match AlignmentScan's complete observable set; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )
    if TOP_CANDIDATES_PER_SPIN < 1 or MAX_REGIONS_PER_SPIN < 1:
        raise ValueError("Candidate and region counts must be positive.")
    if not 0.0 < REGION_SEPARATION <= np.sqrt(2.0):
        raise ValueError("REGION_SEPARATION must lie in (0, sqrt(2)].")
    if CONFIGGEN_KINEMATIC_WORKERS < 1 or CONFIGGEN_PLOT_WORKERS < 1:
        raise ValueError("Kinematic and plot worker counts must be positive.")


def main():
    validate_settings()
    rows = read_scan_rows(SCAN_CSV)
    packages = [build_target(target, rows) for target in CONFIG_TARGETS]
    save_target_plots(packages, rows)
    report = build_report(packages, SCAN_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(report, encoding="utf-8")
    print_console_text(report)


if __name__ == "__main__":
    main()
