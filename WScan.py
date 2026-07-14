"""Search AlignmentScan phase space for the three-qubit W concurrence signature.

Run ``python3 AlignmentScan.py`` first, then ``python3 WScan.py``. This script
uses the identical phase-space points and polarization categories without
recalculating Bethe-Heitler amplitudes.
"""

import csv
from pathlib import Path

import numpy as np

from AlignmentScan import ALIGNMENT_SPIN_CASES, CONCURRENCE_PHASE_SPACE_CSV
from PlotUtils import bin_edges_from_values, require_matplotlib


W_CONCURRENCE = 2.0 / 3.0
TOP_POINTS_PER_POLARIZATION = 100
OUTPUT_DIR = Path("Output") / "WScan"
FULL_CSV = OUTPUT_DIR / "w_distance_phase_space.csv"
ALIGNED_CSV = OUTPUT_DIR / "w_distance_aligned.csv"
TOP_CSV = OUTPUT_DIR / "w_distance_top.csv"
LOG_PATH = Path("Output") / "WScan.log"

KINEMATIC_COLUMNS = (
    "kinematic_point",
    "s_regime",
    "theta_in_regime",
    "qOut_regime",
    "electron_mass",
    "s",
    "sqrt_s",
    "pIn",
    "pOut",
    "qOut",
    "theta_in",
    "phi_in",
    "phi_in_electron",
    "phiOut",
    "Q2",
    "xB",
    "t",
    "F1",
    "F2",
    "W2",
    "y",
    "theta_e_gamma_deg",
    "aligned",
)


def w_distance(c_e_p, c_p_gamma, c_e_gamma):
    """Return Euclidean distance from the W-state concurrence triple."""
    values = np.asarray((c_e_p, c_p_gamma, c_e_gamma), dtype=float)
    return float(np.linalg.norm(values - W_CONCURRENCE))


def _required_columns():
    required = set(KINEMATIC_COLUMNS)
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        required.update({
            f"{prefix}_purity",
            f"{prefix}_C_e_p",
            f"{prefix}_C_p_gamma",
            f"{prefix}_C_e_gamma",
        })
    return required


def load_w_scan_rows(input_path=CONCURRENCE_PHASE_SPACE_CSV):
    """Load AlignmentScan rows and append W-distance columns."""
    input_path = Path(input_path)
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(_required_columns() - fields)
        if missing:
            raise ValueError(
                f"{input_path} is missing required columns: {', '.join(missing)}. "
                "Re-run AlignmentScan.py."
            )
        source_rows = list(reader)

    rows = []
    for source in source_rows:
        row = {name: source[name] for name in KINEMATIC_COLUMNS}
        for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
            c_e_p = float(source[f"{prefix}_C_e_p"])
            c_p_gamma = float(source[f"{prefix}_C_p_gamma"])
            c_e_gamma = float(source[f"{prefix}_C_e_gamma"])
            row.update({
                f"{prefix}_purity": source[f"{prefix}_purity"],
                f"{prefix}_C_e_p": source[f"{prefix}_C_e_p"],
                f"{prefix}_C_p_gamma": source[f"{prefix}_C_p_gamma"],
                f"{prefix}_C_e_gamma": source[f"{prefix}_C_e_gamma"],
                f"{prefix}_D_W": f"{w_distance(c_e_p, c_p_gamma, c_e_gamma):.16e}",
            })
        rows.append(row)
    return rows


def output_headers():
    """Return columns for full and aligned W-distance tables."""
    headers = list(KINEMATIC_COLUMNS)
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        headers.extend((
            f"{prefix}_purity",
            f"{prefix}_C_e_p",
            f"{prefix}_C_p_gamma",
            f"{prefix}_C_e_gamma",
            f"{prefix}_D_W",
        ))
    return headers


def _is_true(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_w_csvs(rows):
    """Write full, aligned-only, and ranked W-distance CSVs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = output_headers()
    for path, selected in (
        (FULL_CSV, rows),
        (ALIGNED_CSV, [row for row in rows if _is_true(row["aligned"])]),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
            writer.writeheader()
            writer.writerows(selected)

    top_headers = [
        "polarization", "polarization_label", "rank", "D_W", "purity",
        "C_e_p", "C_p_gamma", "C_e_gamma",
        *KINEMATIC_COLUMNS,
    ]
    with TOP_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=top_headers, lineterminator="\n")
        writer.writeheader()
        for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
            ranked = sorted(rows, key=lambda row: float(row[f"{prefix}_D_W"]))
            for rank, row in enumerate(ranked[:TOP_POINTS_PER_POLARIZATION], start=1):
                writer.writerow({
                    "polarization": prefix,
                    "polarization_label": label,
                    "rank": rank,
                    "D_W": row[f"{prefix}_D_W"],
                    "purity": row[f"{prefix}_purity"],
                    "C_e_p": row[f"{prefix}_C_e_p"],
                    "C_p_gamma": row[f"{prefix}_C_p_gamma"],
                    "C_e_gamma": row[f"{prefix}_C_e_gamma"],
                    **{name: row[name] for name in KINEMATIC_COLUMNS},
                })
    return FULL_CSV, ALIGNED_CSV, TOP_CSV


def _point_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["kinematic_point"], []).append(row)
    return groups


def _distance_grid(point_rows, column):
    x = np.asarray([float(row["phi_in_electron"]) for row in point_rows])
    y = np.asarray([float(row["phiOut"]) for row in point_rows])
    z = np.asarray([float(row[column]) for row in point_rows])
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    grid = np.full((len(y_unique), len(x_unique)), np.nan)
    x_index = {value: index for index, value in enumerate(x_unique)}
    y_index = {value: index for index, value in enumerate(y_unique)}
    for xv, yv, zv in zip(x, y, z):
        grid[y_index[yv], x_index[xv]] = zv
    return x_unique, y_unique, grid


def write_w_plots(rows):
    """Write one multipanel phase-space PDF for every polarization."""
    plt, PdfPages = require_matplotlib()
    groups = _point_groups(rows)
    paths = []
    maximum_distance = 2.0 / np.sqrt(3.0)
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        path = OUTPUT_DIR / f"w_distance_{prefix}.pdf"
        with PdfPages(path) as pdf:
            fig, axes = plt.subplots(
                1, len(groups), figsize=(5.0 * len(groups), 4.4), squeeze=False,
                constrained_layout=True,
            )
            image = None
            for ax, (point_name, point_rows) in zip(axes[0], groups.items()):
                x, y, grid = _distance_grid(point_rows, f"{prefix}_D_W")
                image = ax.pcolormesh(
                    bin_edges_from_values(x),
                    bin_edges_from_values(y),
                    grid,
                    shading="auto",
                    cmap="viridis_r",
                    vmin=0.0,
                    vmax=maximum_distance,
                )
                best = min(point_rows, key=lambda row: float(row[f"{prefix}_D_W"]))
                ax.plot(float(best["phi_in_electron"]), float(best["phiOut"]),
                        marker="*", color="red", markersize=9)
                ax.set_title(
                    f"{point_name}\nmin $D_W$={float(best[f'{prefix}_D_W']):.4g}",
                    fontsize=9,
                )
                ax.set_xlabel(r"$\phi_{e,\mathrm{in}}$")
                ax.set_ylabel(r"$\phi_\gamma$")
            if image is not None:
                fig.colorbar(image, ax=axes.ravel().tolist(), label=r"$D_W$")
            fig.suptitle(f"W-distance scan: {label}")
            pdf.savefig(fig)
            plt.close(fig)
        paths.append(path)
    return paths


def build_report(rows, plot_paths):
    """Return a concise report of the best point for each polarization."""
    lines = [
        "W-state concurrence-distance scan",
        f"  input: {CONCURRENCE_PHASE_SPACE_CSV}",
        f"  target concurrences: C_e_p=C_p_gamma=C_e_gamma={W_CONCURRENCE:.12g}",
        f"  phase-space rows: {len(rows)}",
        f"  aligned rows: {sum(_is_true(row['aligned']) for row in rows)}",
        "  best points:",
    ]
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        best = min(rows, key=lambda row: float(row[f"{prefix}_D_W"]))
        lines.append(
            f"    {label}: D_W={float(best[f'{prefix}_D_W']):.8g}, "
            f"purity={float(best[f'{prefix}_purity']):.8g}, "
            f"C=({float(best[f'{prefix}_C_e_p']):.8g}, "
            f"{float(best[f'{prefix}_C_p_gamma']):.8g}, "
            f"{float(best[f'{prefix}_C_e_gamma']):.8g}), "
            f"qOut={float(best['qOut']):.6g}, "
            f"phi_e={float(best['phi_in_electron']):.6g}, "
            f"phi_gamma={float(best['phiOut']):.6g}"
        )
    lines.extend((
        f"  full csv: {FULL_CSV}",
        f"  aligned csv: {ALIGNED_CSV}",
        f"  ranked csv: {TOP_CSV}",
        f"  plots: {len(plot_paths)} PDFs under {OUTPUT_DIR}",
    ))
    return "\n".join(lines) + "\n"


def main():
    """Generate W-distance tables, plots, and report from AlignmentScan data."""
    rows = load_w_scan_rows()
    write_w_csvs(rows)
    plot_paths = write_w_plots(rows)
    report = build_report(rows, plot_paths)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
