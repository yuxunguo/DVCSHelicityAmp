"""Two-stage W-distance search over an expanded five-dimensional phase space."""

import csv
from pathlib import Path

import numpy as np

from AlignmentScan import (
    ALIGNMENT_ANGLE_MAX_RAD,
    ALIGNMENT_SPIN_CASES,
    _scan_alignment_point_task,
)
from config import ELECTRON_MASS_GEV, NORMALIZE_TRACE, PROTON_MASS_GEV
from PlotUtils import require_matplotlib
from SpinDensityMat import USER_S_CENTER
from WScan import FULL_CSV as GRID_W_CSV, w_distance


RANDOM_SEED = 271828
BROAD_SAMPLES = 4096
REFINEMENT_SAMPLES = 2048
REFINEMENT_CENTERS = 16
GRID_SEED_CENTERS = 16
TOP_POINTS_PER_POLARIZATION = 100

S_RANGE = (0.72 * USER_S_CENTER, 1.20 * USER_S_CENTER)
THETA_IN_RANGE = (0.35, 2.80)
QOUT_RANGE = (0.30, 1.80)
ANGLE_RANGE = (0.0, 2.0 * np.pi)

PURE_SPIN_PREFIXES = ("LL", "LTx", "LTy", "TxTx", "TxTy")
OUTPUT_DIR = Path("Output") / "BroadWScan"
FULL_CSV = OUTPUT_DIR / "broad_w_phase_space.csv"
TOP_CSV = OUTPUT_DIR / "broad_w_top.csv"
PLOT_PDF = OUTPUT_DIR / "broad_w_search.pdf"
LOG_PATH = Path("Output") / "BroadWScan.log"


def _uniform_samples(rng, count):
    return np.column_stack((
        rng.uniform(*S_RANGE, count),
        rng.uniform(*THETA_IN_RANGE, count),
        rng.uniform(*QOUT_RANGE, count),
        rng.uniform(*ANGLE_RANGE, count),
        rng.uniform(*ANGLE_RANGE, count),
    ))


def _circular_delta(first, second):
    return (first - second + np.pi) % (2.0 * np.pi) - np.pi


def _normalized_distance(first, second):
    scales = np.array((S_RANGE[1] - S_RANGE[0], THETA_IN_RANGE[1] - THETA_IN_RANGE[0],
                       QOUT_RANGE[1] - QOUT_RANGE[0], np.pi, np.pi))
    delta = first - second
    delta[3:] = [_circular_delta(first[i], second[i]) for i in (3, 4)]
    return float(np.linalg.norm(delta / scales))


def _select_refinement_centers(rows):
    ranked = sorted(
        rows,
        key=lambda row: min(float(row[f"{prefix}_D_W"]) for prefix in PURE_SPIN_PREFIXES),
    )
    centers = []
    for row in ranked:
        point = np.array([
            float(row["s"]), float(row["theta_in"]), float(row["qOut"]),
            float(row["phi_in_electron"]), float(row["phiOut"]),
        ])
        if all(_normalized_distance(point.copy(), other.copy()) > 0.08 for other in centers):
            centers.append(point)
        if len(centers) == REFINEMENT_CENTERS:
            break
    return centers


def _grid_seed_points(path=GRID_W_CSV):
    """Return separated low-D_W points from the existing regular grid."""
    path = Path(path)
    if not path.exists():
        return np.empty((0, 5))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ranked = sorted(
        rows,
        key=lambda row: min(float(row[f"{prefix}_D_W"]) for prefix in PURE_SPIN_PREFIXES),
    )
    centers = []
    for row in ranked:
        point = np.array([
            float(row["s"]), float(row["theta_in"]), float(row["qOut"]),
            float(row["phi_in_electron"]), float(row["phiOut"]),
        ])
        if all(_normalized_distance(point.copy(), other.copy()) > 0.08 for other in centers):
            centers.append(point)
        if len(centers) == GRID_SEED_CENTERS:
            break
    return np.asarray(centers)


def _refinement_samples(rng, centers, count):
    scales = np.array((0.04 * USER_S_CENTER, 0.16, 0.14, 0.24, 0.24))
    samples = []
    for index in range(count):
        point = centers[index % len(centers)] + rng.normal(size=5) * scales
        point[0] = np.clip(point[0], *S_RANGE)
        point[1] = np.clip(point[1], *THETA_IN_RANGE)
        point[2] = np.clip(point[2], *QOUT_RANGE)
        point[3:] %= 2.0 * np.pi
        samples.append(point)
    return np.asarray(samples)


def _evaluate_sample(point, sample_id, stage):
    s, theta_in, qout, phi_e, phi_gamma = map(float, point)
    anchor = {
        "kinematic_point": f"{stage}_{sample_id:05d}",
        "s_regime": stage,
        "theta_in_regime": stage,
        "qOut_regime": stage,
        "s": s,
        "theta_in": theta_in,
        "qOut": qout,
    }
    settings = {
        "m": PROTON_MASS_GEV,
        "electron_mass": ELECTRON_MASS_GEV,
        "normalize_trace": NORMALIZE_TRACE,
        "angle_max_rad": ALIGNMENT_ANGLE_MAX_RAD,
    }
    try:
        result = _scan_alignment_point_task((anchor, phi_e, phi_gamma, settings))
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return None
    if not result["ok"]:
        return None
    row = result["row"]
    row["search_stage"] = stage
    row["sample_id"] = sample_id
    for prefix, _label, _spin_case in ALIGNMENT_SPIN_CASES:
        row[f"{prefix}_D_W"] = w_distance(
            row[f"{prefix}_C_e_p"],
            row[f"{prefix}_C_p_gamma"],
            row[f"{prefix}_C_e_gamma"],
        )
    return row


def _evaluate_samples(samples, stage, start_id=0):
    rows = []
    for offset, point in enumerate(samples):
        row = _evaluate_sample(point, start_id + offset, stage)
        if row is not None:
            rows.append(row)
    return rows


def run_broad_search():
    rng = np.random.default_rng(RANDOM_SEED)
    broad_rows = _evaluate_samples(_uniform_samples(rng, BROAD_SAMPLES), "broad")
    seed_points = _grid_seed_points()
    seed_rows = _evaluate_samples(seed_points, "grid_seed", start_id=len(broad_rows))
    centers = _select_refinement_centers(broad_rows + seed_rows)
    refined = _refinement_samples(rng, centers, REFINEMENT_SAMPLES)
    refinement_rows = _evaluate_samples(
        refined, "refined", start_id=len(broad_rows) + len(seed_rows)
    )
    return (
        broad_rows + seed_rows + refinement_rows,
        len(broad_rows),
        len(seed_rows),
        len(refinement_rows),
    )


def _write_dict_csv(path, rows, headers=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = headers or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        if headers:
            writer.writeheader()
            writer.writerows(rows)


def write_outputs(rows):
    _write_dict_csv(FULL_CSV, rows)
    top_rows = []
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        for rank, row in enumerate(
            sorted(rows, key=lambda item: float(item[f"{prefix}_D_W"]))[
                :TOP_POINTS_PER_POLARIZATION
            ],
            start=1,
        ):
            top_rows.append({
                "polarization": prefix,
                "polarization_label": label,
                "rank": rank,
                "D_W": row[f"{prefix}_D_W"],
                "purity": row[f"{prefix}_purity"],
                "C_e_p": row[f"{prefix}_C_e_p"],
                "C_p_gamma": row[f"{prefix}_C_p_gamma"],
                "C_e_gamma": row[f"{prefix}_C_e_gamma"],
                **{name: row[name] for name in (
                    "search_stage", "sample_id", "s", "theta_in", "qOut",
                    "phi_in", "phi_in_electron", "phiOut", "Q2", "xB", "t",
                    "theta_e_gamma_deg", "aligned",
                )},
            })
    _write_dict_csv(TOP_CSV, top_rows)
    return top_rows


def write_plot(rows):
    plt, PdfPages = require_matplotlib()
    labels = {prefix: label for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES}
    with PdfPages(PLOT_PDF) as pdf:
        for prefix in PURE_SPIN_PREFIXES:
            distance = np.asarray([float(row[f"{prefix}_D_W"]) for row in rows])
            cutoff = np.quantile(distance, 0.10)
            color_max = max(distance.min() + 1.0e-9, cutoff)
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
            panels = (
                ("theta_in", "qOut", r"$\theta_{in}$", r"$E_\gamma$ [GeV]"),
                ("s", "qOut", r"$s$ [GeV$^2$]", r"$E_\gamma$ [GeV]"),
                ("phi_in_electron", "phiOut", r"$\phi_{e,in}$", r"$\phi_\gamma$"),
            )
            image = None
            for ax, (x_name, y_name, x_label, y_label) in zip(axes.ravel()[:3], panels):
                image = ax.scatter(
                    [float(row[x_name]) for row in rows],
                    [float(row[y_name]) for row in rows],
                    c=distance, s=6, cmap="viridis_r", vmin=distance.min(),
                    vmax=color_max, rasterized=True,
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
            axes[1, 1].hist(distance, bins=60, color="tab:blue", alpha=0.8)
            axes[1, 1].set_xlabel(r"$D_W$")
            axes[1, 1].set_ylabel("samples")
            fig.colorbar(image, ax=axes.ravel()[:3].tolist(), label=r"$D_W$")
            fig.suptitle(
                f"Expanded W search: {labels[prefix]} [{prefix}], "
                f"min $D_W$={distance.min():.5g}"
            )
            pdf.savefig(fig)
            plt.close(fig)
    return PLOT_PDF


def build_report(rows, broad_valid, seed_valid, refinement_valid):
    lines = [
        "Expanded W-state phase-space search",
        f"  random seed: {RANDOM_SEED}",
        f"  ranges: s={S_RANGE}, theta_in={THETA_IN_RANGE}, qOut={QOUT_RANGE}",
        f"  broad valid samples: {broad_valid}/{BROAD_SAMPLES}",
        f"  regular-grid seed samples: {seed_valid}/{GRID_SEED_CENTERS}",
        f"  refinement valid samples: {refinement_valid}/{REFINEMENT_SAMPLES}",
        f"  total valid samples: {len(rows)}",
        "  best points:",
    ]
    for prefix, label, _spin_case in ALIGNMENT_SPIN_CASES:
        best = min(rows, key=lambda row: float(row[f"{prefix}_D_W"]))
        lines.append(
            f"    {label}: D_W={best[f'{prefix}_D_W']:.8g}, "
            f"C=({best[f'{prefix}_C_e_p']:.7g}, {best[f'{prefix}_C_p_gamma']:.7g}, "
            f"{best[f'{prefix}_C_e_gamma']:.7g}), s={best['s']:.7g}, "
            f"theta={best['theta_in']:.7g}, qOut={best['qOut']:.7g}, "
            f"phi_e={best['phi_in_electron']:.7g}, phi_gamma={best['phiOut']:.7g}"
        )
    lines.extend((f"  full csv: {FULL_CSV}", f"  ranked csv: {TOP_CSV}",
                  f"  diagnostic plots: {PLOT_PDF}"))
    return "\n".join(lines) + "\n"


def main():
    rows, broad_valid, seed_valid, refinement_valid = run_broad_search()
    write_outputs(rows)
    write_plot(rows)
    report = build_report(rows, broad_valid, seed_valid, refinement_valid)
    LOG_PATH.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
