"""Electron-photon concurrence kinematic scan at fixed theta_in with forward/backward phi_e regions.

The scan fixes ``theta_in = pi/2`` and scans ``s`` and ``qOut``.  For each
kinematic point it samples incoming-electron azimuths near two regions:

* forward:  ``phi_e_in ~= 3*pi/2``
* backward: ``phi_e_in ~= pi/2``

The real-photon azimuth ``phi_gamma`` is scanned over the full ``[0, 2*pi)``
range in both regions and is kept as one of the scanned kinematic coordinates.
Outputs include the full phase-space CSV and a reduced CSV selecting the best
``C_e_gamma`` point over the local ``phi_e_in`` samples for each
``(s, qOut, phi_gamma, region)`` kinematic point.
"""

import csv
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from AlignmentScan import (
    ALIGNMENT_SPIN_CASES,
    COARSE_E_GAMMA_TOP_N,
    _concurrence_csv_headers,
    _concurrence_csv_row,
    normalize_azimuth,
    observable_latex_label,
    observable_text_label,
    scan_final_electron_photon_alignment,
)
from SpinDensityMat import USER_S_CENTER


OUTPUT_DIR = Path("Output") / "EGammaKinematicScan"
LOG_PATH = Path("Output") / "EGammaKinematicScan.log"
FULL_CSV_PATH = OUTPUT_DIR / "theta_pi_over_2_phi_e_regions_e_gamma_phase_space.csv"
FORWARD_CSV_PATH = OUTPUT_DIR / "theta_pi_over_2_forward_e_gamma_phase_space.csv"
BACKWARD_CSV_PATH = OUTPUT_DIR / "theta_pi_over_2_backward_e_gamma_phase_space.csv"
BEST_CSV_PATH = OUTPUT_DIR / "theta_pi_over_2_e_gamma_best_by_kinematics.csv"
TOP_CSV_PATH = OUTPUT_DIR / "theta_pi_over_2_e_gamma_top.csv"
PLOT_PREFIX = "theta_pi_over_2_e_gamma_kinematic_scan"

THETA_IN = 0.5 * math.pi
S_VALUES = np.linspace(0.65 * USER_S_CENTER, 1.30 * USER_S_CENTER, 7)
QOUT_VALUES = np.linspace(0.50, 2.00, 7)
PHI_E_HALF_WIDTH = 0.25
PHI_E_STEPS = 7
PHI_GAMMA_VALUES = np.linspace(0.0, 2.0 * math.pi, 36, endpoint=False)
SCAN_MAX_WORKERS = 1
REGIONS = (
    ("forward", 1.5 * math.pi),
    ("backward", 0.5 * math.pi),
)


def _require_matplotlib():
    """Import matplotlib in headless mode with a writable cache directory."""
    cache_dir = Path(tempfile.gettempdir()) / "dvcs_helicity_amp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    return plt, PdfPages


def region_phi_e_values(center):
    """Return a local phi_e scan centered on one region."""
    offsets = np.linspace(-PHI_E_HALF_WIDTH, PHI_E_HALF_WIDTH, PHI_E_STEPS)
    return np.asarray([normalize_azimuth(center + offset) for offset in offsets], dtype=float)


def kinematic_points():
    """Return fixed-theta kinematic anchors for the scan."""
    points = []
    for s_index, s in enumerate(S_VALUES):
        for qout_index, qout in enumerate(QOUT_VALUES):
            points.append({
                "kinematic_point": f"s{s_index:02d}_qout{qout_index:02d}",
                "s_regime": f"s_index_{s_index:02d}",
                "theta_in_regime": "theta_pi_over_2",
                "qOut_regime": f"qout_index_{qout_index:02d}",
                "s": float(s),
                "theta_in": float(THETA_IN),
                "qOut": float(qout),
            })
    return points


def annotate_region_rows(rows, region_name, region_center):
    """Attach region metadata to rows returned by AlignmentScan helpers."""
    for row in rows:
        phi_e = float(row["phi_in_electron"])
        offset = abs((phi_e - region_center + math.pi) % (2.0 * math.pi) - math.pi)
        row["phi_e_region"] = region_name
        row["phi_e_region_center"] = float(region_center)
        row["phi_e_region_offset"] = float(offset)
    return rows


def scan_region(region_name, region_center, points):
    """Run the electron-photon concurrence scan for one forward/backward phi_e region."""
    scan = scan_final_electron_photon_alignment(
        kinematic_points=points,
        phi_in_electron_values=region_phi_e_values(region_center),
        phiOut_values=PHI_GAMMA_VALUES,
        max_workers=SCAN_MAX_WORKERS,
    )
    annotate_region_rows(scan["rows"], region_name, region_center)
    return scan


def scan_all_regions():
    """Run the fixed-theta kinematic electron-photon concurrence scan for all regions."""
    points = kinematic_points()
    rows = []
    failures = []
    scans = {}
    for region_name, region_center in REGIONS:
        scan = scan_region(region_name, region_center, points)
        scans[region_name] = scan
        rows.extend(scan["rows"])
        failures.extend(scan["failures"])
    return {
        "rows": rows,
        "failures": failures,
        "region_scans": scans,
        "kinematic_points": points,
    }


def full_csv_headers():
    """Return CSV headers for the full regional electron-photon concurrence scan."""
    return [
        "phi_e_region",
        "phi_e_region_center",
        "phi_e_region_offset",
        *_concurrence_csv_headers(),
    ]


def full_csv_row(row):
    """Return one formatted full-scan CSV row."""
    return [
        row["phi_e_region"],
        f"{row['phi_e_region_center']:.16e}",
        f"{row['phi_e_region_offset']:.16e}",
        *_concurrence_csv_row(row),
    ]


def write_full_csv(path, rows):
    """Write a full regional electron-photon concurrence phase-space CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(full_csv_headers())
        for row in rows:
            writer.writerow(full_csv_row(row))
    return path


def best_rows_by_kinematics(rows):
    """Return one best phi_e point per region, kinematic point, phi_gamma, and spin."""
    best_rows = []
    for region_name, _region_center in REGIONS:
        region_rows = [row for row in rows if row["phi_e_region"] == region_name]
        point_ids = sorted({row["kinematic_point"] for row in region_rows})
        for point_id in point_ids:
            point_rows = [row for row in region_rows if row["kinematic_point"] == point_id]
            phi_gamma_values = sorted({row["phiOut"] for row in point_rows})
            for phi_gamma in phi_gamma_values:
                gamma_rows = [row for row in point_rows if row["phiOut"] == phi_gamma]
                for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
                    key = f"{prefix}_C_e_gamma"
                    finite_rows = [row for row in gamma_rows if np.isfinite(row.get(key, np.nan))]
                    if not finite_rows:
                        continue
                    best = max(finite_rows, key=lambda row: row[key])
                    best_rows.append({
                        "phi_e_region": region_name,
                        "spin_case": prefix,
                        "spin_label": label,
                        "best_C_e_gamma_key": key,
                        "best_C_e_gamma": best[key],
                        "kinematic_point": best["kinematic_point"],
                        "s": best["s"],
                        "sqrt_s": best["sqrt_s"],
                        "qOut": best["qOut"],
                        "theta_in": best["theta_in"],
                        "phi_in_electron": best["phi_in_electron"],
                        "phi_in": best["phi_in"],
                        "phiOut": best["phiOut"],
                        "phi_e_region_offset": best["phi_e_region_offset"],
                        "Q2": best["Q2"],
                        "xB": best["xB"],
                        "t": best["t"],
                        "W2": best["W2"],
                        "y": best["y"],
                        "theta_e_gamma_deg": best["theta_e_gamma_deg"],
                        "squared_amplitude_M2": best["squared_amplitude_M2"],
                    })
    return best_rows


def best_csv_headers():
    """Return CSV headers for best-by-kinematics rows."""
    return [
        "phi_e_region",
        "spin_case",
        "spin_label",
        "best_C_e_gamma_key",
        "best_C_e_gamma",
        "kinematic_point",
        "s",
        "sqrt_s",
        "qOut",
        "theta_in",
        "phi_in_electron",
        "phi_in",
        "phiOut",
        "phi_e_region_offset",
        "Q2",
        "xB",
        "t",
        "W2",
        "y",
        "theta_e_gamma_deg",
        "squared_amplitude_M2",
    ]


def write_best_csv(path, rows):
    """Write the best-by-kinematics summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = best_csv_headers()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: (
                    f"{row[name]:.16e}"
                    if isinstance(row.get(name), float)
                    else row.get(name, "")
                )
                for name in headers
            })
    return path


def write_top_csv(path, best_rows, top_n=COARSE_E_GAMMA_TOP_N):
    """Write globally ranked best-by-kinematics rows for each region and spin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["rank", *best_csv_headers()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for region_name, _region_center in REGIONS:
            for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
                region_spin_rows = [
                    row for row in best_rows
                    if row["phi_e_region"] == region_name and row["spin_case"] == prefix
                ]
                ordered = sorted(region_spin_rows, key=lambda row: row["best_C_e_gamma"], reverse=True)
                for rank, row in enumerate(ordered[:top_n], start=1):
                    item = {"rank": rank}
                    item.update({
                        name: (
                            f"{row[name]:.16e}"
                            if isinstance(row.get(name), float)
                            else row.get(name, "")
                        )
                        for name in best_csv_headers()
                    })
                    writer.writerow(item)
    return path


def spin_plot_path(prefix):
    """Return the per-polarization plot path."""
    return OUTPUT_DIR / f"{PLOT_PREFIX}_{prefix}.pdf"


def save_spin_plot(path, best_rows, prefix, label):
    """Save kinematic electron-photon concurrence plots for one polarization/spin case."""
    plt, PdfPages = _require_matplotlib()
    observable_label = observable_latex_label("C_e_gamma")
    path.parent.mkdir(parents=True, exist_ok=True)
    spin_rows = [row for row in best_rows if row["spin_case"] == prefix]
    s_values_unique = sorted({row["s"] for row in spin_rows})
    e_gamma_all = np.asarray([row["best_C_e_gamma"] for row in spin_rows], dtype=float)
    e_gamma_max = float(np.nanmax(e_gamma_all)) if e_gamma_all.size else 1.0
    if e_gamma_max <= 0.0:
        e_gamma_max = 1.0
    with PdfPages(path) as pdf:
        for s_value in s_values_unique:
            fig, axes = plt.subplots(
                1,
                len(REGIONS),
                figsize=(12.8, 5.4),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            if len(REGIONS) == 1:
                axes = [axes]
            scatter = None
            for ax, (region_name, _region_center) in zip(axes, REGIONS):
                rows = [
                    row for row in spin_rows
                    if row["s"] == s_value and row["phi_e_region"] == region_name
                ]
                if not rows:
                    ax.set_axis_off()
                    continue
                qout_values = np.asarray([row["qOut"] for row in rows], dtype=float)
                e_gamma_values = np.asarray([row["best_C_e_gamma"] for row in rows], dtype=float)
                phi_gamma_values = np.asarray([row["phiOut"] for row in rows], dtype=float)
                scatter = ax.scatter(
                    phi_gamma_values,
                    qout_values,
                    c=e_gamma_values,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=e_gamma_max,
                    s=58,
                )
                ax.set_title(region_name, fontsize=13)
                ax.set_xlabel("phi_gamma [rad]", fontsize=12)
                ax.set_xlim(0.0, 2.0 * math.pi)
                ax.set_ylim(float(np.min(QOUT_VALUES)), float(np.max(QOUT_VALUES)))
                ax.tick_params(labelsize=10)
            axes[0].set_ylabel("qOut [GeV]", fontsize=12)
            fig.suptitle(f"{label}: {observable_label} at s={s_value:.6g}", fontsize=16)
            if scatter is not None:
                colorbar = fig.colorbar(scatter, ax=axes, label=observable_label)
                colorbar.set_label(observable_label, fontsize=12)
                pdf.savefig(fig)
            plt.close(fig)
    return path


def save_plots(best_rows):
    """Save one kinematic scan PDF for each polarization/spin case."""
    outputs = {}
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        outputs[prefix] = save_spin_plot(spin_plot_path(prefix), best_rows, prefix, label)
    return outputs


def build_report(scan, paths, best_rows):
    """Return a text report for the fixed-theta electron-photon concurrence scan."""
    rows = scan["rows"]
    observable_label = observable_text_label("C_e_gamma")
    lines = [
        f"Fixed-theta {observable_label} kinematic scan",
        f"  theta_in: {THETA_IN:.16e} rad",
        f"  s scan: {len(S_VALUES)} values from {S_VALUES[0]:.6g} to {S_VALUES[-1]:.6g}",
        f"  qOut scan: {len(QOUT_VALUES)} values from {QOUT_VALUES[0]:.6g} to {QOUT_VALUES[-1]:.6g}",
        f"  phi_e local scan: {PHI_E_STEPS} values within +/- {PHI_E_HALF_WIDTH:.6g} rad",
        f"  phi_gamma kinematic scan: {len(PHI_GAMMA_VALUES)} values over [0, 2*pi)",
        f"  valid points: {len(rows)}",
        f"  invalid points: {len(scan['failures'])}",
        f"  saved full csv: {paths['full_csv']}",
        f"  saved forward csv: {paths['forward_csv']}",
        f"  saved backward csv: {paths['backward_csv']}",
        f"  saved best-by-kinematics csv: {paths['best_csv']}",
        f"  saved top csv: {paths['top_csv']}",
        "  saved plot pdfs:",
    ]
    for prefix, path in paths["plots"].items():
        lines.append(f"    {prefix}: {path}")
    lines.extend([
        "",
        f"Top best-by-kinematics {observable_label} rows:",
    ])
    for region_name, _region_center in REGIONS:
        lines.append(f"  {region_name}:")
        for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
            spin_rows = [
                row for row in best_rows
                if row["phi_e_region"] == region_name and row["spin_case"] == prefix
            ]
            if not spin_rows:
                continue
            best = max(spin_rows, key=lambda row: row["best_C_e_gamma"])
            lines.append(
                "    "
                f"{label}: {observable_label}={best['best_C_e_gamma']:.8g}, "
                f"s={best['s']:.8g}, qOut={best['qOut']:.8g}, "
                f"phi_e_in={best['phi_in_electron']:.8g}, "
                f"phi_gamma={best['phiOut']:.8g}, "
                f"Q2={best['Q2']:.8g}, xB={best['xB']:.8g}, t={best['t']:.8g}"
            )
    if scan["failures"]:
        lines.append("")
        lines.append("First invalid points:")
        for failure in scan["failures"][:10]:
            point_id, s, theta_in, phi_in, phi_e, qout, phiout, message = failure
            lines.append(
                "  "
                f"point={point_id}, s={s:.8g}, theta_in={theta_in:.8g}, "
                f"phi_e_in={phi_e:.8g}, phi_p_in={phi_in:.8g}, "
                f"qOut={qout:.8g}, phi_gamma={phiout:.8g}: {message}"
            )
    lines.append("")
    lines.append(f"Saved log: {LOG_PATH}")
    return "\n".join(lines) + "\n"


def main():
    """Generate fixed-theta forward/backward electron-photon concurrence scan outputs."""
    scan = scan_all_regions()
    rows = scan["rows"]
    forward_rows = [row for row in rows if row["phi_e_region"] == "forward"]
    backward_rows = [row for row in rows if row["phi_e_region"] == "backward"]
    best_rows = best_rows_by_kinematics(rows)
    paths = {
        "full_csv": write_full_csv(FULL_CSV_PATH, rows),
        "forward_csv": write_full_csv(FORWARD_CSV_PATH, forward_rows),
        "backward_csv": write_full_csv(BACKWARD_CSV_PATH, backward_rows),
        "best_csv": write_best_csv(BEST_CSV_PATH, best_rows),
        "top_csv": write_top_csv(TOP_CSV_PATH, best_rows),
        "plots": save_plots(best_rows),
    }
    log_text = build_report(scan, paths, best_rows)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text, end="")


if __name__ == "__main__":
    main()
