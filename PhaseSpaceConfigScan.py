"""Generate ConfigGen-style packages from PhaseSpaceScan results.

The continuous PhaseSpaceScan photon-energy coordinate is divided into three
balanced low/mid/high ``E_gamma`` bands.  Fixed-polarization scans reuse
ConfigGen's established machinery.  Coherent mixing-angle scans instead
cluster in all seven scan coordinates and preserve the selected ``theta_e``
and ``theta_p`` when reconstructing amplitudes.
"""

import csv
import math
from pathlib import Path

import numpy as np

import ConfigGen as config
import config as scan_settings
from AlignmentScan import LEPTON_SPECS
from config import SCAN_WORKERS
from FormFactors import yahl_dirac_pauli_from_t
from PlotUtils import print_console_text
from SpinDensityMat import (
    amplitude_table,
    mixed_angle_final_state,
    outgoing_spin_states,
)


# Script controls. Edit these values before running PhaseSpaceConfigScan.py.
PHASE_SPACE_CONFIG_LEPTONS = ("electron", "muon", "heavy", "massless")
PHASE_SPACE_CONFIG_WORKERS = SCAN_WORKERS
PHASE_SPACE_CONFIG_PLOT_WORKERS = max(1, min(SCAN_WORKERS, 24))
ENERGY_BAND_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)
ENERGY_BAND_LABELS = ("low_Egamma", "mid_Egamma", "high_Egamma")
PHASE_SPACE_CONFIG_TARGETS = config.CONFIG_TARGETS

PHASE_SPACE_OUTPUT_ROOT = Path("Output") / "PhaseSpaceScan"
OUTPUT_ROOT = Path("Output") / "PhaseSpaceConfigScan"
LOG_PATH = Path("Output") / "PhaseSpaceConfigScan.log"


def phase_space_input_path(lepton_name):
    """Return the PhaseSpaceScan CSV matching the central polarization mode."""
    try:
        stem = LEPTON_SPECS[lepton_name]["file_stem"]
    except KeyError as exc:
        raise ValueError(
            f"Unknown lepton {lepton_name!r}; choose from {tuple(LEPTON_SPECS)}."
        ) from exc
    suffix = (
        "mixing_angle_phase_space.csv"
        if scan_settings.SCAN_INITIAL_MIXING_ANGLES
        else "entanglement_phase_space.csv"
    )
    path = PHASE_SPACE_OUTPUT_ROOT / lepton_name / f"{stem}_{suffix}"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PhaseSpaceScan input {path}. Run PhaseSpaceScan.py with "
            f"SCAN_INITIAL_MIXING_ANGLES="
            f"{scan_settings.SCAN_INITIAL_MIXING_ANGLES} first."
        )
    return path


def assign_energy_bands(rows):
    """Assign balanced low/mid/high E_gamma bands to phase-space rows."""
    qout_values = np.asarray(
        [config.parse_float(row.get("qOut")) for row in rows],
        dtype=float,
    )
    finite = np.isfinite(qout_values)
    if not np.any(finite):
        raise ValueError("PhaseSpaceScan rows contain no finite qOut values.")

    finite_qout = qout_values[finite]
    boundaries = np.quantile(finite_qout, ENERGY_BAND_QUANTILES)
    if boundaries[0] >= boundaries[1]:
        raise ValueError(
            "Cannot form distinct photon-energy bands from the saved qOut values."
        )

    counts = {label: 0 for label in ENERGY_BAND_LABELS}
    for row, qout in zip(rows, qout_values):
        row["phase_space_stage"] = row.get("qOut_regime", "")
        if not np.isfinite(qout):
            row["qOut_regime"] = "invalid_Egamma"
            continue
        if qout <= boundaries[0]:
            label = ENERGY_BAND_LABELS[0]
        elif qout <= boundaries[1]:
            label = ENERGY_BAND_LABELS[1]
        else:
            label = ENERGY_BAND_LABELS[2]
        row["qOut_regime"] = label
        counts[label] += 1

    return {
        "minimum": float(np.min(finite_qout)),
        "lower_boundary": float(boundaries[0]),
        "upper_boundary": float(boundaries[1]),
        "maximum": float(np.max(finite_qout)),
        "counts": counts,
    }


def prepared_input_path(lepton_name):
    """Return the worker-readable CSV with PhaseSpaceScan energy bands."""
    return OUTPUT_ROOT / lepton_name / "phase_space_config_input.csv"


def energy_band_report(bands, prepared_path):
    """Return report lines describing the PhaseSpaceScan energy partition."""
    return [
        "PhaseSpaceConfigScan photon-energy preparation",
        f"  prepared worker csv: {prepared_path}",
        "  energy bands use qOut terciles over valid PhaseSpaceScan rows",
        (
            f"  {ENERGY_BAND_LABELS[0]}: "
            f"{bands['minimum']:.8g} <= E_gamma <= "
            f"{bands['lower_boundary']:.8g} GeV "
            f"({bands['counts'][ENERGY_BAND_LABELS[0]]} rows)"
        ),
        (
            f"  {ENERGY_BAND_LABELS[1]}: "
            f"{bands['lower_boundary']:.8g} < E_gamma <= "
            f"{bands['upper_boundary']:.8g} GeV "
            f"({bands['counts'][ENERGY_BAND_LABELS[1]]} rows)"
        ),
        (
            f"  {ENERGY_BAND_LABELS[2]}: "
            f"{bands['upper_boundary']:.8g} < E_gamma <= "
            f"{bands['maximum']:.8g} GeV "
            f"({bands['counts'][ENERGY_BAND_LABELS[2]]} rows)"
        ),
    ]


def mixing_prefix(lepton_name):
    """Return the coherent-polarization column prefix used by PhaseSpaceScan."""
    return f"lepton_{lepton_name}_theta_mix_proton_theta_p_mix"


def validate_mixing_columns(rows, lepton_name):
    """Validate the coherent scan's two angles and configured observables."""
    if not rows:
        raise ValueError("The mixing-angle PhaseSpaceScan CSV contains no rows.")
    names = set(rows[0])
    required = {"theta_e", "theta_p"}
    prefix = mixing_prefix(lepton_name)
    required.update(
        f"{prefix}_{observable}"
        for observable, _file_tag in PHASE_SPACE_CONFIG_TARGETS
    )
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            "The mixing-angle scan CSV is missing required columns: "
            + ", ".join(missing)
            + ". Rerun PhaseSpaceScan.py in mixing-angle mode."
        )


def _threshold_mask(values, observable):
    """Select finite values within STEP of the requested minimum/maximum."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return finite
    optimum = (
        float(np.min(values[finite]))
        if config.observable_is_minimized(observable)
        else float(np.max(values[finite]))
    )
    tolerance = scan_settings.PHASE_SPACE_CONFIG_STEP
    return finite & (np.abs(values - optimum) <= tolerance + 1.0e-12)


def _threshold_fixed_rows(rows):
    """Mask fixed-polarization values farther than STEP from each optimum."""
    filtered = [dict(row) for row in rows]
    for observable, _file_tag in PHASE_SPACE_CONFIG_TARGETS:
        for key in config.target_columns(rows, observable):
            values = np.asarray(
                [config.parse_float(row.get(key)) for row in rows],
                dtype=float,
            )
            keep = _threshold_mask(values, observable)
            for row, retained in zip(filtered, keep):
                if not retained:
                    row[key] = ""
    return filtered


def _plain_write_csv(path, rows):
    """Write mixed-angle rows without fixed-spin name conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        if headers:
            writer.writeheader()
            writer.writerows(rows)
    return path


def _mixing_row_distance(a, b):
    """Distance in the five kinematic and two polarization scan dimensions."""
    s_scale = max(
        abs(config.parse_float(a.get("s"))),
        abs(config.parse_float(b.get("s"))),
        1.0,
    )
    q_scale = max(
        abs(config.parse_float(a.get("qOut"))),
        abs(config.parse_float(b.get("qOut"))),
        1.0,
    )
    pieces = (
        (config.parse_float(a.get("s")) - config.parse_float(b.get("s"))) / s_scale,
        (
            config.parse_float(a.get("theta_in"))
            - config.parse_float(b.get("theta_in"))
        ) / math.pi,
        config.circular_distance(
            config.parse_float(a.get("phi_in")),
            config.parse_float(b.get("phi_in")),
        ) / math.pi,
        (
            config.parse_float(a.get("qOut"))
            - config.parse_float(b.get("qOut"))
        ) / q_scale,
        config.circular_distance(
            config.parse_float(a.get("phiOut")),
            config.parse_float(b.get("phiOut")),
        ) / math.pi,
        (
            config.parse_float(a.get("theta_e"))
            - config.parse_float(b.get("theta_e"))
        ) / math.pi,
        (
            config.parse_float(a.get("theta_p"))
            - config.parse_float(b.get("theta_p"))
        ) / math.pi,
    )
    return float(np.linalg.norm(np.asarray(pieces, dtype=float)))


def _annotated_mixing_candidates(rows, lepton_name, observable):
    """Return ranked coherent-angle candidates with ConfigGen metadata."""
    prefix = mixing_prefix(lepton_name)
    key = f"{prefix}_{observable}"
    values = np.asarray(
        [config.parse_float(row.get(key)) for row in rows],
        dtype=float,
    )
    keep = _threshold_mask(values, observable)
    candidates = []
    for row, value, retained in zip(rows, values, keep):
        theta_e = config.parse_float(row.get("theta_e"))
        theta_p = config.parse_float(row.get("theta_p"))
        if not retained or not all(
            np.isfinite(item) for item in (value, theta_e, theta_p)
        ):
            continue
        item = dict(row)
        item.update({
            "selected_observable": observable,
            "selected_observable_label": config.observable_label(observable),
            "selected_spin_case": "mixing_angles",
            "selected_spin_label": (
                f"theta_e={theta_e:.8g}, theta_p={theta_p:.8g}"
            ),
            "selected_concurrence_key": key,
            "selected_concurrence": value,
            "selected_purity": config.parse_float(row.get(f"{prefix}_purity")),
            "pair_delta_xy": config.target_pair_delta(item, observable),
            "scan_phi_lepton_in": config.parse_float(row.get("phi_in_lepton")),
            "scan_phi_p_in": config.parse_float(row.get("phi_in")),
            "scan_phi_gamma": config.parse_float(row.get("phiOut")),
        })
        candidates.append(item)
    candidates.sort(
        key=lambda item: item["selected_concurrence"],
        reverse=not config.observable_is_minimized(observable),
    )
    return candidates


def _select_mixing_regions(rows, lepton_name, observable, file_tag):
    """Greedily select separated optima using all seven scan variables."""
    candidates = _annotated_mixing_candidates(rows, lepton_name, observable)
    selected = []
    for band in ENERGY_BAND_LABELS:
        band_selected = []
        band_candidates = [
            candidate for candidate in candidates
            if candidate.get("qOut_regime") == band
        ][: config.TOP_ROWS_PER_TARGET_SPIN]
        for candidate in band_candidates:
            if band_selected and min(
                _mixing_row_distance(candidate, prior)
                for prior in band_selected
            ) <= config.CLUSTER_RADIUS:
                continue
            band_selected.append(candidate)
            if len(band_selected) >= config.MAX_CLUSTERS_PER_TARGET_SPIN:
                break
        for band_index, row in enumerate(band_selected):
            index = len(selected)
            row["cluster_id"] = index
            row["energy_band_cluster_id"] = band_index
            row["selected_region"] = (
                f"{band}_{config.selected_region_name(observable, band_index)}"
            )
            row["detail_id"] = (
                f"{file_tag}_{band}_mixing_angles_region_{band_index}"
            )
            row["detail_source"] = "phase_space_mixing_angle_scan"
            selected.append(row)
    return selected


def _mixing_cluster_rows(detail_rows):
    """Return one cluster-summary record per selected seven-dimensional region."""
    return [{
        "detail_id": row["detail_id"],
        "selected_observable": row["selected_observable"],
        "selected_observable_label": row["selected_observable_label"],
        "selected_spin_case": row["selected_spin_case"],
        "selected_region": row["selected_region"],
        "qOut_regime": row.get("qOut_regime", ""),
        "cluster_id": row["cluster_id"],
        "energy_band_cluster_id": row["energy_band_cluster_id"],
        "selected_concurrence": f"{row['selected_concurrence']:.16e}",
        "selected_purity": f"{row['selected_purity']:.16e}",
        "theta_e": f"{config.parse_float(row['theta_e']):.16e}",
        "theta_p": f"{config.parse_float(row['theta_p']):.16e}",
        "sqrt_s": row.get("sqrt_s", ""),
        "theta_in": row.get("theta_in", ""),
        "phi_in": row.get("phi_in", ""),
        "qOut": row.get("qOut", ""),
        "phiOut": row.get("phiOut", ""),
    } for row in detail_rows]


def _mixing_momentum_rows(detail_rows):
    """Return reconstructed four-momenta while retaining both mixing angles."""
    records = []
    for row in detail_rows:
        kin = config.kinematics_from_config_row(row)
        for name in config.DISPLAY_MOMENTA:
            vector = kin["momenta"][name]
            records.append({
                "detail_id": row["detail_id"],
                "selected_observable": row["selected_observable"],
                "selected_region": row["selected_region"],
                "theta_e": f"{config.parse_float(row['theta_e']):.16e}",
                "theta_p": f"{config.parse_float(row['theta_p']):.16e}",
                "momentum": name,
                "E": f"{vector[0]:.16e}",
                "px": f"{vector[1]:.16e}",
                "py": f"{vector[2]:.16e}",
                "pz": f"{vector[3]:.16e}",
                "p_abs": f"{np.linalg.norm(vector[1:4]):.16e}",
                "phi_xy": f"{config.vector_phi_xy(vector):.16e}",
                "s": f"{kin['s']:.16e}",
                "sqrt_s": f"{kin['sqrt_s']:.16e}",
                "pIn": f"{kin['pIn']:.16e}",
                "pOut": f"{kin['pOut']:.16e}",
                "theta_in": f"{kin['theta_in']:.16e}",
                "phi_in": f"{kin['phi_in']:.16e}",
                "qOut": f"{kin['qOut']:.16e}",
                "phiOut": f"{kin['phiOut']:.16e}",
            })
    return records


def _mixing_amplitude_rows(detail_rows):
    """Reconstruct pure-state amplitudes at each selected pair of angles."""
    records = []
    for row in detail_rows:
        kin = config.kinematics_from_config_row(row)
        F1, F2 = yahl_dirac_pauli_from_t(kin["t"], config.M)
        amplitudes = amplitude_table(
            kin["momenta"],
            config.M,
            F1,
            F2,
            electron_mass=kin["electron_mass"],
        )
        theta_e = config.parse_float(row["theta_e"])
        theta_p = config.parse_float(row["theta_p"])
        state = mixed_angle_final_state(amplitudes, theta_e, theta_p)
        norms = np.abs(state) ** 2
        total = float(np.sum(norms))
        if not np.isfinite(total) or total <= 0.0:
            raise ZeroDivisionError(
                f"Non-positive mixed-state norm for {row['detail_id']}."
            )
        components = []
        for index, ((h_out, s_out, lam), amplitude, norm) in enumerate(
            zip(outgoing_spin_states(), state, norms)
        ):
            fraction = float(norm / total)
            components.append({
                "detail_id": row["detail_id"],
                "selected_observable": row["selected_observable"],
                "selected_region": row["selected_region"],
                "incoming_state": (
                    "cos(theta_e)|+>+sin(theta_e)|-> tensor "
                    "cos(theta_p)|+>+sin(theta_p)|->"
                ),
                "theta_e": f"{theta_e:.16e}",
                "theta_p": f"{theta_p:.16e}",
                "final_state_order": "p_gamma_lepton",
                "final_ket": f"|{s_out:+d}{lam:+d}{h_out:+d}>",
                "out_index": index,
                "hOut": h_out,
                "sOut": s_out,
                "lambda": lam,
                "amplitude_real": f"{amplitude.real:.16e}",
                "amplitude_imag": f"{amplitude.imag:.16e}",
                "amplitude_abs": f"{abs(amplitude):.16e}",
                "amplitude_phase": f"{np.angle(amplitude):.16e}",
                "amplitude_abs2": f"{norm:.16e}",
                "fraction": f"{fraction:.16e}",
            })
        components.sort(key=lambda item: float(item["fraction"]), reverse=True)
        kept = [
            item
            for item in components
            if float(item["fraction"]) >= config.AMPLITUDE_MIN_FRACTION
        ][: config.AMPLITUDE_MAX_COMPONENTS]
        retained = sum(float(item["fraction"]) for item in kept)
        for rank, item in enumerate(kept, start=1):
            item["decomposition_rank"] = rank
            item["retained_fraction_total"] = f"{retained:.16e}"
        records.extend(kept)
    return records


def _plot_mixing_configuration_text(ax, row, kin):
    """Draw the old ConfigGen kinematic summary with coherent-angle metadata."""
    ax.axis("off")
    pair_delta = config.parse_float(row.get("pair_delta_xy"))
    region_line = f"region: {row['selected_region']}"
    if np.isfinite(pair_delta):
        region_line += f"  final-pair delta_xy={pair_delta:.6g} rad"
    theta_e = config.parse_float(row["theta_e"])
    theta_p = config.parse_float(row["theta_p"])
    lines = [
        (
            f"{row['detail_id']}  {row['selected_observable_label']}="
            f"{row['selected_concurrence']:.6g}"
        ),
        region_line,
        f"outgoing-state purity: {row['selected_purity']:.6g}",
        f"kinematic point: {row.get('kinematic_point', '')}",
        (
            "incoming lepton: "
            f"cos({theta_e:.8g})|+> + sin({theta_e:.8g})|->"
        ),
        (
            "incoming proton: "
            f"cos({theta_p:.8g})|+> + sin({theta_p:.8g})|->"
        ),
        "",
        (
            rf"$s$={kin['s']:.6g}, $\sqrt{{s}}$={kin['sqrt_s']:.6g}, "
            rf"$|\vec{{P}}|$={kin['pIn']:.6g}, "
            rf"$|\vec{{P}}^{{\,\prime}}|$={kin['pOut']:.6g}"
        ),
        (
            rf"$\theta_{{\rm in}}$={kin['theta_in']:.6g}, "
            rf"$\phi_\ell$="
            f"{config.parse_float(row.get('phi_in_lepton')):.6g}, "
            rf"$\phi_P$={kin['phi_in']:.6g}"
        ),
        rf"$E_\gamma$={kin['qOut']:.6g}, $\phi_\gamma$={kin['phiOut']:.6g}",
        (
            rf"$Q^2$={kin['Q2']:.6g}, $x_B$={kin['xB']:.6g}, "
            rf"$t$={kin['t']:.6g}, $W^2$={kin['W2']:.6g}, "
            rf"$y$={kin['y']:.6g}"
        ),
        "",
        r"four-momenta [$E$, $p_x$, $p_y$, $p_z$] GeV:",
    ]
    lines.extend(
        config.format_vector_line(name, kin["momenta"][name])
        for name in config.DISPLAY_MOMENTA
    )
    ax.text(
        0.0, 1.0, "\n".join(lines),
        va="top", ha="left", family="monospace", fontsize=8.7,
    )


def _plot_mixing_amplitude_decomposition(ax, row):
    """Draw leading coherent final-state components in |p gamma lepton> order."""
    records = _mixing_amplitude_rows([row])
    if not records:
        ax.text(0.5, 0.5, "No retained amplitude components", ha="center")
        ax.axis("off")
        return
    labels = [
        (
            rf"$|p\,\gamma\,\ell\rangle={item['final_ket']}$"
            "\n"
            rf"$(h_p,h_\gamma,h_\ell)="
            rf"({item['sOut']:+d},{item['lambda']:+d},{item['hOut']:+d})$"
        )
        for item in records
    ]
    fractions = np.asarray(
        [config.parse_float(item["fraction"]) for item in records],
        dtype=float,
    )
    phases = np.asarray(
        [config.parse_float(item["amplitude_phase"]) for item in records],
        dtype=float,
    )
    y_pos = np.arange(len(records))
    bars = ax.barh(y_pos, fractions, color="tab:blue", alpha=0.72)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"coherent final-state $|A|^2$ fraction")
    retained = config.parse_float(records[0]["retained_fraction_total"])
    ax.set_title(
        f"Leading final-state amplitudes "
        f"(N={len(records)}, retained={retained:.1%})"
    )
    ax.set_xlim(0.0, max(0.08, float(np.nanmax(fractions)) * 1.25))
    for bar, item, phase in zip(bars, records, phases):
        ax.text(
            bar.get_width(),
            bar.get_y() + 0.5 * bar.get_height(),
            (
                f"  phase={phase:.2f}, "
                f"Re={config.parse_float(item['amplitude_real']):.2e}, "
                f"Im={config.parse_float(item['amplitude_imag']):.2e}"
            ),
            va="center",
            fontsize=8,
        )
    ax.tick_params(axis="y", labelsize=8)


def _save_mixing_detail_pages(pdf, plt, detail_rows):
    """Append the old momentum/text/amplitude page for every selected optimum."""
    for row in detail_rows:
        kin = config.kinematics_from_config_row(row)
        fig = plt.figure(figsize=(13.2, 8.2), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.05, 1.25))
        config.plot_momentum_panels(fig, grid, kin)
        text_ax = fig.add_subplot(grid[0, 1:])
        _plot_mixing_configuration_text(text_ax, row, kin)
        amplitude_ax = fig.add_subplot(grid[1, 1:])
        _plot_mixing_amplitude_decomposition(amplitude_ax, row)
        fig.suptitle(
            (
                f"coherent mixing angles: "
                f"{config.observable_optimum_word(row['selected_observable'])} "
                f"{config.observable_math_label(row['selected_observable'])} "
                f"configuration ({row.get('qOut_regime', '')})"
            ),
            fontsize=16,
        )
        pdf.savefig(fig)
        plt.close(fig)


def _write_mixing_target_plot(rows, detail_rows, lepton_name, observable, path):
    """Write scan overview followed by detailed selected-configuration pages."""
    plt, PdfPages = config._require_matplotlib()
    prefix = mixing_prefix(lepton_name)
    key = f"{prefix}_{observable}"
    panels = (
        ("theta_in", "qOut", r"$\theta_{in}$", r"$E_\gamma$ [GeV]"),
        ("sqrt_s", "qOut", r"$\sqrt{s}$ [GeV]", r"$E_\gamma$ [GeV]"),
        ("phi_in", "phiOut", r"$\phi_{P,in}$", r"$\phi_\gamma$"),
        ("theta_e", "theta_p", r"$\theta_e$", r"$\theta_p$"),
        ("sqrt_s", "theta_e", r"$\sqrt{s}$ [GeV]", r"$\theta_e$"),
        ("sqrt_s", "theta_p", r"$\sqrt{s}$ [GeV]", r"$\theta_p$"),
        ("qOut", "theta_e", r"$E_\gamma$ [GeV]", r"$\theta_e$"),
        ("qOut", "theta_p", r"$E_\gamma$ [GeV]", r"$\theta_p$"),
    )
    values = np.asarray([config.parse_float(row.get(key)) for row in rows])
    finite = _threshold_mask(values, observable)
    cmap, vmin, vmax = config.observable_plot_style(observable)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(
            3, 3, figsize=(14.0, 11.5), constrained_layout=True
        )
        image = None
        for ax, (x_name, y_name, x_label, y_label) in zip(
            axes.ravel()[:8], panels
        ):
            x = np.asarray([config.parse_float(row.get(x_name)) for row in rows])
            y = np.asarray([config.parse_float(row.get(y_name)) for row in rows])
            mask = finite & np.isfinite(x) & np.isfinite(y)
            image = ax.scatter(
                x[mask], y[mask], c=values[mask], s=6, cmap=cmap,
                vmin=vmin, vmax=vmax, rasterized=True,
            )
            if detail_rows:
                ax.scatter(
                    [config.parse_float(row.get(x_name)) for row in detail_rows],
                    [config.parse_float(row.get(y_name)) for row in detail_rows],
                    marker="*", s=120, c="red", edgecolors="black",
                )
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
        axes[2, 2].hist(values[finite], bins=60, color="tab:blue", alpha=0.8)
        axes[2, 2].set_xlabel(config.observable_label(observable))
        axes[2, 2].set_ylabel("samples")
        fig.suptitle(
            f"{lepton_name}: coherent mixing-angle configuration regions; "
            f"within STEP={scan_settings.PHASE_SPACE_CONFIG_STEP:g} "
            f"of the {config.observable_optimum_word(observable)}"
        )
        if image is not None:
            fig.colorbar(
                image,
                ax=axes.ravel()[:8].tolist(),
                label=config.observable_label(observable),
            )
        pdf.savefig(fig)
        plt.close(fig)
        _save_mixing_detail_pages(pdf, plt, detail_rows)
    return path


def run_mixing_species(lepton_name, rows, input_path, bands, prepared_path):
    """Build coherent-angle packages without fixed-spin assumptions."""
    outputs = []
    for target in PHASE_SPACE_CONFIG_TARGETS:
        observable, _default_tag = target
        file_tag = config.target_file_tag(target)
        detail_rows = _select_mixing_regions(
            rows, lepton_name, observable, file_tag
        )
        paths = config.target_paths(file_tag)
        _plain_write_csv(paths["examples"], detail_rows)
        _plain_write_csv(paths["clusters"], _mixing_cluster_rows(detail_rows))
        _plain_write_csv(paths["momenta"], _mixing_momentum_rows(detail_rows))
        _plain_write_csv(
            paths["amplitudes"], _mixing_amplitude_rows(detail_rows)
        )
        plot_path = (
            config.OUTPUT_DIR
            / mixing_prefix(lepton_name)
            / f"{file_tag}_regions.pdf"
        )
        _write_mixing_target_plot(
            rows, detail_rows, lepton_name, observable, plot_path
        )
        outputs.append((observable, len(detail_rows), paths, plot_path))

    lines = energy_band_report(bands, prepared_path)
    lines.extend([
        "PhaseSpaceConfigScan coherent mixing-angle configuration generation",
        f"  lepton: {lepton_name}",
        f"  input: {input_path}",
        f"  rows: {len(rows)}",
        (
            f"  displayed/eligible points: within STEP="
            f"{scan_settings.PHASE_SPACE_CONFIG_STEP:g} of each target optimum"
        ),
        "  scan dimensions: s, theta_in, phi_in, qOut, phiOut, theta_e, theta_p",
        f"  polarization prefix: {mixing_prefix(lepton_name)}",
    ])
    for observable, count, paths, plot_path in outputs:
        lines.append(
            f"  {config.observable_label(observable)}: {count} regions; "
            f"examples={paths['examples']}; amplitudes={paths['amplitudes']}; "
            f"plot={plot_path}"
        )
    return "\n".join(lines)


def run_species(lepton_name):
    """Build one species' configuration packages from PhaseSpaceScan rows."""
    input_path = phase_space_input_path(lepton_name)
    config.configure_lepton(
        lepton_name,
        input_path=input_path,
        output_root=OUTPUT_ROOT,
    )
    config.CONFIGGEN_KINEMATIC_WORKERS = PHASE_SPACE_CONFIG_WORKERS
    config.CONFIGGEN_PLOT_WORKERS = PHASE_SPACE_CONFIG_PLOT_WORKERS

    rows = config.read_csv_rows(input_path)
    bands = assign_energy_bands(rows)
    prepared_path = prepared_input_path(lepton_name)
    if scan_settings.SCAN_INITIAL_MIXING_ANGLES:
        validate_mixing_columns(rows, lepton_name)
        _plain_write_csv(prepared_path, rows)
        config.clean_egamma_config_outputs()
        config.clean_data_outputs()
        return run_mixing_species(
            lepton_name, rows, input_path, bands, prepared_path
        )

    config.validate_config_target_columns(rows)
    rows = _threshold_fixed_rows(rows)
    config.write_dict_csv(prepared_path, rows)
    config.clean_egamma_config_outputs()
    config.clean_data_outputs()
    egamma_outputs, egamma_detail_rows = (
        config.save_all_egamma_target_region_pdfs(
            rows,
            input_path=prepared_path,
        )
    )
    packages = [
        config.build_target_package(target, rows, egamma_detail_rows)
        for target in PHASE_SPACE_CONFIG_TARGETS
    ]
    report = config.build_report(
        input_path,
        len(rows),
        packages,
        egamma_outputs,
        source_label="PhaseSpaceScan",
    )
    return "\n".join(energy_band_report(bands, prepared_path)) + "\n" + report


def validate_settings():
    """Validate explicit script controls before changing output trees."""
    unknown = set(PHASE_SPACE_CONFIG_LEPTONS) - set(LEPTON_SPECS)
    if unknown:
        raise ValueError(f"Unknown PhaseSpaceConfigScan leptons: {sorted(unknown)}")
    if not PHASE_SPACE_CONFIG_LEPTONS:
        raise ValueError("PHASE_SPACE_CONFIG_LEPTONS must not be empty.")
    if PHASE_SPACE_CONFIG_WORKERS < 1:
        raise ValueError("PHASE_SPACE_CONFIG_WORKERS must be positive.")
    if PHASE_SPACE_CONFIG_PLOT_WORKERS < 1:
        raise ValueError("PHASE_SPACE_CONFIG_PLOT_WORKERS must be positive.")
    if (
        not np.isfinite(scan_settings.PHASE_SPACE_CONFIG_STEP)
        or scan_settings.PHASE_SPACE_CONFIG_STEP < 0.0
    ):
        raise ValueError("PHASE_SPACE_CONFIG_STEP must be finite and non-negative.")
    if len(ENERGY_BAND_QUANTILES) != 2:
        raise ValueError("ENERGY_BAND_QUANTILES must contain two boundaries.")
    if not 0.0 < ENERGY_BAND_QUANTILES[0] < ENERGY_BAND_QUANTILES[1] < 1.0:
        raise ValueError("ENERGY_BAND_QUANTILES must be ordered inside (0, 1).")
    if "D_W" not in {observable for observable, _tag in PHASE_SPACE_CONFIG_TARGETS}:
        raise ValueError("PhaseSpaceConfigScan requires the D_W ConfigGen target.")
    if "M2_magic" not in {
        observable for observable, _tag in PHASE_SPACE_CONFIG_TARGETS
    }:
        raise ValueError(
            "PhaseSpaceConfigScan requires the M2_magic ConfigGen target."
        )


def main():
    """Generate PhaseSpaceScan-driven configurations for selected species."""
    validate_settings()
    reports = [run_species(name) for name in PHASE_SPACE_CONFIG_LEPTONS]
    report_text = "\n\n".join(report.rstrip() for report in reports) + "\n"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(report_text, encoding="utf-8")
    print_console_text(report_text)


if __name__ == "__main__":
    main()
