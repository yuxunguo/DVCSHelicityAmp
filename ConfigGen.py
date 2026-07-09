"""Generate representative high-C_e_gamma configurations from AlignmentScan outputs.

This script reads the electron-photon concurrence CSVs written by ``AlignmentScan.py`` and builds a
small set of representative user-frame configurations directly from those
coarse alignment results. It prefers the ranked ``electron_photon_e_gamma_top.csv``
table and falls back to the full phase-space concurrence CSV when needed.
"""

import csv
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from AlignmentScan import observable_latex_label, observable_text_label
from FormFactors import yahl_dirac_pauli_from_t
from Kinematics import kinematics_user_from_independent
from SpinDensityMat import (
    ENTANGLEMENT_INITIAL_STATE,
    M,
    SPIN_CASE_TRANSVERSE_TX,
    SPIN_CASE_TRANSVERSE_TY,
    amplitude_table,
    initial_spin_states,
    outgoing_spin_states,
    transverse_electron_coefficients,
)


ALIGNMENT_CONCURRENCE_DIR = Path("Output") / "AlignmentScan" / "ConcurrenceScan"
RANKED_E_GAMMA_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_e_gamma_top.csv"
LEGACY_RANKED_C13_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_c13_top.csv"
FULL_CONCURRENCE_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_concurrence_phase_space.csv"

OUTPUT_DIR = Path("Output") / "ConfigGen"
LOG_PATH = Path("Output") / "ConfigGen.log"
EXAMPLES_CSV_PATH = OUTPUT_DIR / "high_e_gamma_configuration_examples.csv"
CLUSTERS_CSV_PATH = OUTPUT_DIR / "high_e_gamma_cluster_summary.csv"
MOMENTA_CSV_PATH = OUTPUT_DIR / "high_e_gamma_momentum_configurations.csv"
AMPLITUDE_CSV_PATH = OUTPUT_DIR / "high_e_gamma_final_state_amplitude_decomposition.csv"
PLOT_PATH = OUTPUT_DIR / "high_e_gamma_user_frame_configurations.pdf"

TOP_ROWS_PER_SPIN = 60
MAX_CLUSTERS_PER_SPIN = 8
EXAMPLES_PER_CLUSTER = 2
CLUSTER_RADIUS = 0.42
CONFIG_SPIN_CASES = ("Tx", "Ty")
CONFIG_RELATIVE_REGIONS = (
    ("near_azimuth", 0.0),
    ("back_to_back_azimuth", math.pi),
)
REGION_HALF_WIDTH = 0.25
DISPLAY_MOMENTA = ("k", "p", "kp", "pp", "qout")
INCOMING_MOMENTA = ("k", "p")
HIGH_E_GAMMA_PREFIX = "high_e_gamma_"

KINEMATIC_COLUMNS = (
    "kinematic_point",
    "s_regime",
    "theta_in_regime",
    "qOut_regime",
    "s",
    "sqrt_s",
    "pIn",
    "pOut",
    "theta_in",
    "phi_in_electron",
    "phi_in",
    "qOut",
    "phiOut",
    "Q2",
    "xB",
    "t",
    "W2",
    "y",
    "theta_e_gamma_deg",
    "k_dot_qout",
    "kp_dot_qout",
    "abs_k_dot_qout",
    "abs_kp_dot_qout",
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


def parse_float(value, default=np.nan):
    """Parse a CSV numeric field, preserving missing values as NaN."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_csv_rows(path):
    """Read a CSV file as dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def alignment_input_path():
    """Return the best available AlignmentScan electron-photon concurrence CSV path."""
    if RANKED_E_GAMMA_CSV.exists():
        return RANKED_E_GAMMA_CSV
    if LEGACY_RANKED_C13_CSV.exists():
        return LEGACY_RANKED_C13_CSV
    if FULL_CONCURRENCE_CSV.exists():
        return FULL_CONCURRENCE_CSV
    raise FileNotFoundError(
        "No AlignmentScan electron-photon concurrence CSV found. Run AlignmentScan.py "
        f"first to create {RANKED_E_GAMMA_CSV} or {FULL_CONCURRENCE_CSV}."
    )


def e_gamma_columns(rows):
    """Return electron-photon concurrence columns present in the alignment CSV rows."""
    if not rows:
        return []
    columns = [
        name for name in rows[0]
        if name.endswith("_C_e_gamma") or name.endswith("_C13")
    ]
    preferred = [f"{spin_case}_C_e_gamma" for spin_case in CONFIG_SPIN_CASES]
    legacy_preferred = [f"{spin_case}_C13" for spin_case in CONFIG_SPIN_CASES]
    return [name for name in preferred if name in columns] + [
        name for name in legacy_preferred if name in columns
    ] + [
        name for name in columns
        if (
            name not in preferred
            and name not in legacy_preferred
            and spin_label_from_key(name) in CONFIG_SPIN_CASES
        )
    ]


def spin_label_from_key(key):
    """Return the spin-case prefix from an electron-photon concurrence column name."""
    if key.endswith("_C_e_gamma"):
        return key[: -len("_C_e_gamma")]
    if key.endswith("_C13"):
        return key[:-4]
    return key


def electron_photon_delta(row):
    """Return the shortest azimuthal separation between incoming electron and photon."""
    return circular_distance(parse_float(row.get("phi_in_electron")), parse_float(row.get("phiOut")))


def region_offset(delta, center):
    """Return distance of an electron/photon separation from a region center."""
    if not np.isfinite(delta):
        return np.nan
    return abs(delta - center)


def candidate_rows(rows, key, region_name, region_center):
    """Return ranked candidates for one spin electron-photon concurrence column."""
    if rows and "rank_group" in rows[0]:
        group_rows = [row for row in rows if row.get("rank_group") == key]
        source_rows = group_rows if group_rows else rows
    else:
        source_rows = rows

    candidates = []
    seen = set()
    for row in source_rows:
        delta = electron_photon_delta(row)
        offset = region_offset(delta, region_center)
        if not np.isfinite(offset) or offset > REGION_HALF_WIDTH:
            continue
        value = parse_float(row.get("rank_value") if row.get("rank_group") == key else row.get(key))
        if not np.isfinite(value):
            continue
        identity = tuple(row.get(name, "") for name in KINEMATIC_COLUMNS)
        if identity in seen:
            continue
        seen.add(identity)
        item = dict(row)
        item["selected_spin_case"] = spin_label_from_key(key)
        item["selected_e_gamma_key"] = key
        item["selected_C_e_gamma"] = value
        item["selected_region"] = region_name
        item["electron_photon_delta"] = delta
        item["electron_photon_region_center"] = region_center
        item["electron_photon_region_offset"] = offset
        candidates.append(item)
    candidates.sort(key=lambda item: item["selected_C_e_gamma"], reverse=True)
    return candidates[:TOP_ROWS_PER_SPIN]


def circular_distance(a, b):
    """Return angular distance on [0, 2*pi)."""
    diff = abs(float(a) - float(b)) % (2.0 * math.pi)
    return min(diff, 2.0 * math.pi - diff)


def row_distance(a, b):
    """Return a normalized distance between two user-frame configurations."""
    s_scale = max(abs(parse_float(a.get("s"))), abs(parse_float(b.get("s"))), 1.0)
    q_scale = max(abs(parse_float(a.get("qOut"))), abs(parse_float(b.get("qOut"))), 1.0)
    pieces = [
        (parse_float(a.get("s")) - parse_float(b.get("s"))) / s_scale,
        (parse_float(a.get("theta_in")) - parse_float(b.get("theta_in"))) / math.pi,
        circular_distance(parse_float(a.get("phi_in_electron")), parse_float(b.get("phi_in_electron"))) / math.pi,
        (parse_float(a.get("qOut")) - parse_float(b.get("qOut"))) / q_scale,
        circular_distance(parse_float(a.get("phiOut")), parse_float(b.get("phiOut"))) / math.pi,
    ]
    return float(np.sqrt(np.sum(np.asarray(pieces, dtype=float) ** 2)))


def cluster_candidates(candidates):
    """Greedily cluster high-C_e_gamma candidate rows by user-frame coordinates."""
    clusters = []
    for row in candidates:
        best_index = None
        best_distance = np.inf
        for index, cluster in enumerate(clusters):
            distance = row_distance(row, cluster["best"])
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is not None and best_distance <= CLUSTER_RADIUS:
            clusters[best_index]["rows"].append(row)
            if row["selected_C_e_gamma"] > clusters[best_index]["best"]["selected_C_e_gamma"]:
                clusters[best_index]["best"] = row
        elif len(clusters) < MAX_CLUSTERS_PER_SPIN:
            clusters.append({"best": row, "rows": [row]})
        else:
            nearest = min(range(len(clusters)), key=lambda index: row_distance(row, clusters[index]["best"]))
            clusters[nearest]["rows"].append(row)

    for index, cluster in enumerate(clusters):
        cluster["cluster_id"] = index
        cluster["rows"].sort(key=lambda item: item["selected_C_e_gamma"], reverse=True)
        cluster["best"] = cluster["rows"][0]
    return clusters


def numeric_range(rows, name):
    """Return min/max for a numeric CSV column."""
    values = np.asarray([parse_float(row.get(name)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(values.min()), float(values.max())


def format_range(rows, name):
    """Return a compact numeric range for reports."""
    low, high = numeric_range(rows, name)
    if not np.isfinite(low) or not np.isfinite(high):
        return "nan..nan"
    return f"{low:.6g}..{high:.6g}"


def cluster_summary_rows(grouped_clusters):
    """Return dictionaries for the cluster summary CSV."""
    summaries = []
    for spin_case, region_name, clusters in grouped_clusters:
        for cluster in clusters:
            rows = cluster["rows"]
            best = cluster["best"]
            summary = {
                "selected_spin_case": spin_case,
                "selected_region": region_name,
                "cluster_id": cluster["cluster_id"],
                "size": len(rows),
                "max_C_e_gamma": f"{best['selected_C_e_gamma']:.16e}",
                "best_kinematic_point": best.get("kinematic_point", ""),
                "best_electron_photon_delta": f"{best['electron_photon_delta']:.16e}",
                "best_electron_photon_region_offset": (
                    f"{best['electron_photon_region_offset']:.16e}"
                ),
            }
            for name in (
                "s",
                "theta_in",
                "phi_in_electron",
                "qOut",
                "phiOut",
                "Q2",
                "xB",
                "t",
                "W2",
                "y",
                "theta_e_gamma_deg",
                "abs_k_dot_qout",
                "abs_kp_dot_qout",
            ):
                low, high = numeric_range(rows, name)
                summary[f"{name}_min"] = f"{low:.16e}" if np.isfinite(low) else ""
                summary[f"{name}_max"] = f"{high:.16e}" if np.isfinite(high) else ""
            for name in KINEMATIC_COLUMNS:
                summary[f"best_{name}"] = best.get(name, "")
            summaries.append(summary)
    return summaries


def example_rows(grouped_clusters):
    """Return representative example configuration rows."""
    examples = []
    for spin_case, region_name, clusters in grouped_clusters:
        for cluster in clusters:
            for rank, row in enumerate(cluster["rows"][:EXAMPLES_PER_CLUSTER], start=1):
                item = {
                    "selected_spin_case": spin_case,
                    "selected_region": region_name,
                    "cluster_id": cluster["cluster_id"],
                    "example_rank": rank,
                    "selected_C_e_gamma": f"{row['selected_C_e_gamma']:.16e}",
                    "electron_photon_delta": f"{row['electron_photon_delta']:.16e}",
                    "electron_photon_region_offset": (
                        f"{row['electron_photon_region_offset']:.16e}"
                    ),
                }
                for name in KINEMATIC_COLUMNS:
                    item[name] = row.get(name, "")
                for key, value in row.items():
                    if key.endswith("_C_e_gamma") or key.endswith("_C13"):
                        item[key] = value
                examples.append(item)
    return examples


def write_dict_csv(path, rows):
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        headers = list(rows[0])
    else:
        headers = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def polarization_file_tag(spin_case):
    """Return a filesystem-friendly tag for one selected polarization."""
    tag = "".join(char.lower() if char.isalnum() else "_" for char in str(spin_case))
    return tag.strip("_") or "unknown"


def polarization_output_path(base_path, spin_case):
    """Return the per-polarization output path next to an aggregate path."""
    tag = polarization_file_tag(spin_case)
    name = base_path.name
    if name.startswith(HIGH_E_GAMMA_PREFIX):
        name = f"{HIGH_E_GAMMA_PREFIX}{tag}_{name[len(HIGH_E_GAMMA_PREFIX):]}"
    else:
        name = f"{base_path.stem}_{tag}{base_path.suffix}"
    return base_path.with_name(name)


def polarization_csv_path(base_path, spin_case):
    """Return the per-polarization CSV path next to an aggregate CSV path."""
    return polarization_output_path(base_path, spin_case)


def write_polarization_csvs(base_path, rows, spin_cases):
    """Write one filtered CSV per selected polarization."""
    outputs = []
    for spin_case in spin_cases:
        spin_rows = [row for row in rows if row.get("selected_spin_case") == spin_case]
        outputs.append((
            spin_case,
            write_dict_csv(polarization_csv_path(base_path, spin_case), spin_rows),
        ))
    return outputs


def representative_rows(grouped_clusters):
    """Return one characteristic row per selected spin/cluster."""
    rows = []
    for spin_case, region_name, clusters in grouped_clusters:
        for cluster in clusters:
            row = dict(cluster["best"])
            row["selected_spin_case"] = spin_case
            row["selected_region"] = region_name
            row["cluster_id"] = cluster["cluster_id"]
            row["detail_id"] = f"{spin_case}_{region_name}_cluster_{cluster['cluster_id']}"
            rows.append(row)
    return rows


def kinematics_from_config_row(row):
    """Rebuild full user-frame kinematics for a ConfigGen row."""
    return kinematics_user_from_independent(
        parse_float(row.get("s")),
        parse_float(row.get("theta_in")),
        parse_float(row.get("phi_in")),
        parse_float(row.get("qOut")),
        parse_float(row.get("phiOut")),
        M,
        label=row.get("detail_id") or row.get("kinematic_point"),
    )


def vector_phi_xy(vector):
    """Return azimuth of a four-vector's transverse momentum."""
    vector = np.asarray(vector, dtype=float)
    return float(math.atan2(vector[2], vector[1]))


def momentum_configuration_rows(detail_rows):
    """Return one CSV row per four-momentum in each detailed configuration."""
    records = []
    for row in detail_rows:
        kin = kinematics_from_config_row(row)
        for name in DISPLAY_MOMENTA:
            vector = kin["momenta"][name]
            records.append({
                "detail_id": row["detail_id"],
                "selected_spin_case": row["selected_spin_case"],
                "selected_region": row["selected_region"],
                "cluster_id": row["cluster_id"],
                "selected_C_e_gamma": f"{row['selected_C_e_gamma']:.16e}",
                "electron_photon_delta": f"{row['electron_photon_delta']:.16e}",
                "electron_photon_region_offset": f"{row['electron_photon_region_offset']:.16e}",
                "kinematic_point": row.get("kinematic_point", ""),
                "momentum": name,
                "E": f"{vector[0]:.16e}",
                "px": f"{vector[1]:.16e}",
                "py": f"{vector[2]:.16e}",
                "pz": f"{vector[3]:.16e}",
                "p_abs": f"{np.linalg.norm(vector[1:4]):.16e}",
                "phi_xy": f"{vector_phi_xy(vector):.16e}",
                "s": f"{kin['s']:.16e}",
                "sqrt_s": f"{kin['sqrt_s']:.16e}",
                "pIn": f"{kin['pIn']:.16e}",
                "pOut": f"{kin['pOut']:.16e}",
                "theta_in": f"{kin['theta_in']:.16e}",
                "phi_in": f"{kin['phi_in']:.16e}",
                "qOut": f"{kin['qOut']:.16e}",
                "phiOut": f"{kin['phiOut']:.16e}",
                "Q2": f"{kin['Q2']:.16e}",
                "xB": f"{kin['xB']:.16e}",
                "t": f"{kin['t']:.16e}",
                "W2": f"{kin['W2']:.16e}",
                "y": f"{kin['y']:.16e}",
            })
    return records


def selected_final_state_amplitudes(amplitudes, spin_case):
    """Return the final-state amplitude vector used for one selected spin case."""
    in_states = initial_spin_states()
    proton_spin = ENTANGLEMENT_INITIAL_STATE[1]
    if spin_case == "unpolarized":
        return np.sum(amplitudes, axis=0) / np.sqrt(len(in_states))
    if spin_case == "longitudinal_polarized":
        return (
            amplitudes[in_states.index((+1, proton_spin))]
            - amplitudes[in_states.index((-1, proton_spin))]
        ) / np.sqrt(2.0)
    if spin_case == "Tx":
        coefficients = transverse_electron_coefficients(SPIN_CASE_TRANSVERSE_TX)
    elif spin_case == "Ty":
        coefficients = transverse_electron_coefficients(SPIN_CASE_TRANSVERSE_TY)
    else:
        coefficients = None
    if coefficients is not None:
        return sum(
            coefficients[h_in] * amplitudes[in_states.index((h_in, proton_spin))]
            for h_in in (-1, +1)
        )
    raise ValueError(f"Unknown ConfigGen spin case: {spin_case}")


def selected_initial_state_label(spin_case):
    """Return a compact label for the incoming spin state behind a plot."""
    proton_spin = ENTANGLEMENT_INITIAL_STATE[1]
    if spin_case == "unpolarized":
        return "coherent equal sum over all (hIn,sIn) rows, divided by sqrt(4)"
    if spin_case == "longitudinal_polarized":
        return f"(A[hIn=+1,sIn={proton_spin}] - A[hIn=-1,sIn={proton_spin}]) / sqrt(2)"
    if spin_case == "Tx":
        return f"(A[hIn=+1,sIn={proton_spin}] + A[hIn=-1,sIn={proton_spin}]) / sqrt(2)"
    if spin_case == "Ty":
        return f"(A[hIn=+1,sIn={proton_spin}] + i A[hIn=-1,sIn={proton_spin}]) / sqrt(2)"
    return spin_case


def amplitude_decomposition(row):
    """Return final-state amplitude decomposition records for one detail row."""
    kin = kinematics_from_config_row(row)
    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], M)
    amplitudes = amplitude_table(kin["momenta"], M, F1, F2)
    state = selected_final_state_amplitudes(amplitudes, row["selected_spin_case"])
    norms = np.abs(state) ** 2
    total = float(np.sum(norms))
    if total <= 1.0e-14:
        raise ZeroDivisionError(f"Zero selected amplitude norm for {row['detail_id']}.")
    records = []
    for index, ((h_out, s_out, lam), amplitude, norm) in enumerate(
        zip(outgoing_spin_states(), state, norms)
    ):
        records.append({
            "detail_id": row["detail_id"],
            "selected_spin_case": row["selected_spin_case"],
            "selected_region": row["selected_region"],
            "cluster_id": row["cluster_id"],
            "selected_C_e_gamma": f"{row['selected_C_e_gamma']:.16e}",
            "electron_photon_delta": f"{row['electron_photon_delta']:.16e}",
            "electron_photon_region_offset": f"{row['electron_photon_region_offset']:.16e}",
            "kinematic_point": row.get("kinematic_point", ""),
            "incoming_state": selected_initial_state_label(row["selected_spin_case"]),
            "out_index": index,
            "hOut": h_out,
            "sOut": s_out,
            "lambda": lam,
            "amplitude_real": f"{amplitude.real:.16e}",
            "amplitude_imag": f"{amplitude.imag:.16e}",
            "amplitude_abs": f"{abs(amplitude):.16e}",
            "amplitude_phase": f"{np.angle(amplitude):.16e}",
            "amplitude_abs2": f"{norm:.16e}",
            "fraction": f"{norm / total:.16e}",
        })
    records.sort(key=lambda item: parse_float(item["fraction"]), reverse=True)
    return records


def amplitude_decomposition_rows(detail_rows):
    """Return final-state amplitude decomposition rows for all details."""
    rows = []
    for row in detail_rows:
        rows.extend(amplitude_decomposition(row))
    return rows


def _plot_vector_3d(ax, vector, label, color, incoming=False):
    """Draw one 3D momentum vector."""
    spatial = np.asarray(vector, dtype=float)[1:4]
    start = np.zeros(3)
    delta = spatial
    text_position = spatial
    ax.quiver(
        start[0],
        start[1],
        start[2],
        delta[0],
        delta[1],
        delta[2],
        color=color,
        arrow_length_ratio=0.08,
    )
    ax.text(
        text_position[0],
        text_position[1],
        text_position[2],
        f" {label}",
        color=color,
        fontsize=8,
    )


def _plot_vector_2d(ax, vector, label, color, incoming=False):
    """Draw one transverse momentum vector."""
    spatial = np.asarray(vector, dtype=float)[1:3]
    start = np.zeros(2)
    delta = spatial
    text_position = spatial
    ax.arrow(
        start[0],
        start[1],
        delta[0],
        delta[1],
        color=color,
        width=0.0,
        head_width=0.045,
        length_includes_head=True,
    )
    ax.text(text_position[0], text_position[1], f" {label}", color=color, fontsize=10, va="center")


def _set_symmetric_limits_3d(ax, momenta):
    """Use symmetric limits so 3D momentum directions are not distorted."""
    vectors = np.asarray([momenta[name][1:4] for name in DISPLAY_MOMENTA])
    limit = max(1.0, float(np.nanmax(np.abs(vectors))) * 1.15)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)


def plot_momentum_panels(fig, grid_spec, kin):
    """Draw 3D and transverse-plane momentum panels."""
    colors = {
        "k": "tab:blue",
        "p": "tab:orange",
        "kp": "tab:cyan",
        "pp": "tab:red",
        "qout": "tab:green",
    }
    momenta = kin["momenta"]
    ax3d = fig.add_subplot(grid_spec[0, 0], projection="3d")
    ax2d = fig.add_subplot(grid_spec[1, 0])
    for name in DISPLAY_MOMENTA:
        incoming = name in INCOMING_MOMENTA
        _plot_vector_3d(ax3d, momenta[name], name, colors[name], incoming=incoming)
        _plot_vector_2d(ax2d, momenta[name], name, colors[name], incoming=incoming)
    _set_symmetric_limits_3d(ax3d, momenta)
    ax3d.set_title("3D momenta")
    ax3d.set_xlabel("px [GeV]")
    ax3d.set_ylabel("py [GeV]")
    ax3d.set_zlabel("pz [GeV]")
    transverse = np.asarray([momenta[name][1:3] for name in DISPLAY_MOMENTA])
    limit = max(1.0, float(np.nanmax(np.abs(transverse))) * 1.20)
    ax2d.set_xlim(-limit, limit)
    ax2d.set_ylim(-limit, limit)
    ax2d.axhline(0.0, color="0.82", linewidth=0.8)
    ax2d.axvline(0.0, color="0.82", linewidth=0.8)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_title("Transverse plane")
    ax2d.set_xlabel("px [GeV]")
    ax2d.set_ylabel("py [GeV]")


def format_vector_line(name, vector):
    """Return one compact four-vector line."""
    return (
        f"{name:5s} E={vector[0]:8.4f} "
        f"px={vector[1]:8.4f} py={vector[2]:8.4f} pz={vector[3]:8.4f}"
    )


def plot_configuration_text(ax, row, kin):
    """Draw kinematic and momentum text for a detailed configuration page."""
    ax.axis("off")
    momenta = kin["momenta"]
    lines = [
        f"{row['detail_id']}  {observable_text_label('C_e_gamma')}={row['selected_C_e_gamma']:.6g}",
        (
            f"region: {row['selected_region']}  "
            f"|phi_e_in - phi_gamma|={row['electron_photon_delta']:.6g}"
        ),
        f"kinematic point: {row.get('kinematic_point', '')}",
        f"incoming state: {selected_initial_state_label(row['selected_spin_case'])}",
        "",
        (
            f"s={kin['s']:.6g}, sqrt(s)={kin['sqrt_s']:.6g}, "
            f"pIn={kin['pIn']:.6g}, pOut={kin['pOut']:.6g}"
        ),
        (
            f"theta_in={kin['theta_in']:.6g}, "
            f"phi_e_in={parse_float(row.get('phi_in_electron')):.6g}, "
            f"phi_p_in={kin['phi_in']:.6g}"
        ),
        (
            f"qOut={kin['qOut']:.6g}, "
            f"phi_gamma={kin['phiOut']:.6g}"
        ),
        (
            f"Q2={kin['Q2']:.6g}, xB={kin['xB']:.6g}, "
            f"t={kin['t']:.6g}, W2={kin['W2']:.6g}, y={kin['y']:.6g}"
        ),
        f"theta(e',gamma)={parse_float(row.get('theta_e_gamma_deg')):.6g} deg",
        "",
        "four-momenta [E, px, py, pz] GeV:",
    ]
    lines.extend(format_vector_line(name, momenta[name]) for name in DISPLAY_MOMENTA)
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)


def plot_amplitude_decomposition(ax, row):
    """Draw final-state amplitude fractions and phases."""
    records = amplitude_decomposition(row)
    labels = [
        f"h'={item['hOut']}, s'={item['sOut']}, lam={item['lambda']}"
        for item in records
    ]
    fractions = np.asarray([parse_float(item["fraction"]) for item in records], dtype=float)
    phases = np.asarray([parse_float(item["amplitude_phase"]) for item in records], dtype=float)
    y_pos = np.arange(len(records))
    bars = ax.barh(y_pos, fractions, color="tab:blue", alpha=0.72)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("final-state |A|^2 fraction")
    ax.set_title("Final-state amplitude decomposition")
    ax.set_xlim(0.0, max(0.08, float(np.nanmax(fractions)) * 1.25))
    for bar, item, phase in zip(bars, records, phases):
        ax.text(
            bar.get_width(),
            bar.get_y() + 0.5 * bar.get_height(),
            (
                f"  phase={phase:.2f}, "
                f"Re={parse_float(item['amplitude_real']):.2e}, "
                f"Im={parse_float(item['amplitude_imag']):.2e}"
            ),
            va="center",
            fontsize=8,
        )
    ax.tick_params(axis="y", labelsize=8)


def save_detail_pages(pdf, plt, detail_rows):
    """Append one momentum/amplitude page for every representative row."""
    for row in detail_rows:
        kin = kinematics_from_config_row(row)
        fig = plt.figure(figsize=(13.2, 8.2), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.05, 1.25))
        plot_momentum_panels(fig, grid, kin)
        text_ax = fig.add_subplot(grid[0, 1:])
        plot_configuration_text(text_ax, row, kin)
        amp_ax = fig.add_subplot(grid[1, 1:])
        plot_amplitude_decomposition(amp_ax, row)
        fig.suptitle(
            f"{row['selected_spin_case']} characteristic high-{observable_latex_label('C_e_gamma')} configuration",
            fontsize=16,
        )
        pdf.savefig(fig)
        plt.close(fig)


def save_configuration_plot(grouped_clusters, path=PLOT_PATH):
    """Save diagnostic angle maps and detailed momentum/amplitude pages."""
    plt, PdfPages = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    detail_rows = representative_rows(grouped_clusters)
    with PdfPages(path) as pdf:
        for spin_case, region_name, clusters in grouped_clusters:
            rows = [row for cluster in clusters for row in cluster["rows"]]
            if not rows:
                continue
            x = np.asarray([parse_float(row.get("phi_in_electron")) for row in rows], dtype=float)
            y = np.asarray([parse_float(row.get("phiOut")) for row in rows], dtype=float)
            c = np.asarray([row["selected_C_e_gamma"] for row in rows], dtype=float)
            fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=True)
            scatter = ax.scatter(x, y, c=c, cmap="viridis", s=28, alpha=0.75)
            for cluster in clusters:
                best = cluster["best"]
                ax.scatter(
                    [parse_float(best.get("phi_in_electron"))],
                    [parse_float(best.get("phiOut"))],
                    marker="x",
                    s=80,
                    color="black",
                    linewidths=1.4,
                )
                ax.text(
                    parse_float(best.get("phi_in_electron")),
                    parse_float(best.get("phiOut")),
                    f" {cluster['cluster_id']}",
                    fontsize=8,
                    va="center",
                )
            ax.set_title(
                f"{spin_case} {region_name}: representative high-{observable_latex_label('C_e_gamma')} configs",
                fontsize=14,
            )
            ax.set_xlabel(r"$\phi_{e,\rm in}$ [rad]", fontsize=12)
            ax.set_ylabel(r"$\phi_{\gamma}'$ [rad]", fontsize=12)
            ax.set_xlim(0.0, 2.0 * math.pi)
            ax.set_ylim(0.0, 2.0 * math.pi)
            ax.tick_params(labelsize=10)
            colorbar = fig.colorbar(scatter, ax=ax, label=observable_latex_label("C_e_gamma"))
            colorbar.set_label(observable_latex_label("C_e_gamma"), fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
        save_detail_pages(pdf, plt, detail_rows)
    return path


def save_polarization_plots(grouped_clusters, base_path=PLOT_PATH):
    """Write one configuration plot PDF per selected polarization."""
    outputs = []
    spin_cases = []
    for spin_case, _region_name, _clusters in grouped_clusters:
        if spin_case not in spin_cases:
            spin_cases.append(spin_case)
    for spin_case in spin_cases:
        spin_groups = [group for group in grouped_clusters if group[0] == spin_case]
        path = polarization_output_path(base_path, spin_case)
        outputs.append((spin_case, save_configuration_plot(spin_groups, path)))
    return outputs


def build_report(
    input_path,
    total_rows,
    grouped_clusters,
    examples,
    polarization_outputs,
    polarization_plot_outputs,
):
    """Build the text report for the generated configurations."""
    observable_label = observable_text_label("C_e_gamma")
    lines = [
        f"High-{observable_label} user-frame configuration generator from AlignmentScan results",
        f"  input csv: {input_path}",
        f"  top rows per spin case: {TOP_ROWS_PER_SPIN}",
        f"  cluster radius: {CLUSTER_RADIUS}",
        f"  total input rows: {total_rows}",
        f"  spin cases: {len({spin for spin, _region, _clusters in grouped_clusters})}",
        f"  angular regions: {', '.join(region for region, _center in CONFIG_RELATIVE_REGIONS)}",
        "  angular region criterion: shortest azimuthal separation "
        "|phi_e_in - phi_gamma|",
        f"  angular region half width: {REGION_HALF_WIDTH:.6g} rad",
        f"  clusters: {sum(len(clusters) for _spin, _region, clusters in grouped_clusters)}",
        f"  examples: {len(examples)}",
        f"  saved examples csv: {EXAMPLES_CSV_PATH}",
        f"  saved clusters csv: {CLUSTERS_CSV_PATH}",
        f"  saved momentum configuration csv: {MOMENTA_CSV_PATH}",
        f"  saved final-state amplitude decomposition csv: {AMPLITUDE_CSV_PATH}",
        f"  saved configuration plot: {PLOT_PATH}",
        "  saved per-polarization csvs:",
    ]
    for label, outputs in polarization_outputs:
        lines.append(f"    {label}:")
        for spin_case, path in outputs:
            lines.append(f"      {spin_case}: {path}")
    lines.append("  saved per-polarization plot pdfs:")
    for spin_case, path in polarization_plot_outputs:
        lines.append(f"    {spin_case}: {path}")
    lines.extend([
        "",
        "Cluster summaries:",
    ])
    for spin_case, region_name, clusters in grouped_clusters:
        lines.append(f"  {spin_case} {region_name}: {len(clusters)} clusters")
        for cluster in clusters:
            best = cluster["best"]
            rows = cluster["rows"]
            lines.append(
                "    "
                f"cluster {cluster['cluster_id']}: size={len(rows)}, "
                f"max_{observable_label}={best['selected_C_e_gamma']:.6g}, "
                f"|phi_e-phi_gamma|={format_range(rows, 'electron_photon_delta')}, "
                f"theta(e',gamma)={format_range(rows, 'theta_e_gamma_deg')}, "
                f"s={format_range(rows, 's')}, "
                f"theta_in={format_range(rows, 'theta_in')}, "
                f"qOut={format_range(rows, 'qOut')}, "
                f"Q2={format_range(rows, 'Q2')}, "
                f"xB={format_range(rows, 'xB')}, "
                f"t={format_range(rows, 't')}"
            )
            lines.append(
                "      best: "
                f"s={parse_float(best.get('s')):.6g}, "
                f"theta_in={parse_float(best.get('theta_in')):.6g}, "
                f"phi_e_in={parse_float(best.get('phi_in_electron')):.6g}, "
                f"phi_p_in={parse_float(best.get('phi_in')):.6g}, "
                f"qOut={parse_float(best.get('qOut')):.6g}, "
                f"phi_gamma={parse_float(best.get('phiOut')):.6g}, "
                f"|phi_e-phi_gamma|={best['electron_photon_delta']:.6g}, "
                f"Q2={parse_float(best.get('Q2')):.6g}, "
                f"xB={parse_float(best.get('xB')):.6g}, "
                f"t={parse_float(best.get('t')):.6g}"
            )
    lines.append("")
    lines.append(f"Saved log: {LOG_PATH}")
    return "\n".join(lines) + "\n"


def main():
    """Generate representative configurations from current AlignmentScan CSVs."""
    input_path = alignment_input_path()
    rows = read_csv_rows(input_path)
    grouped_clusters = []
    for key in e_gamma_columns(rows):
        for region_name, region_center in CONFIG_RELATIVE_REGIONS:
            candidates = candidate_rows(rows, key, region_name, region_center)
            if not candidates:
                continue
            grouped_clusters.append((
                spin_label_from_key(key),
                region_name,
                cluster_candidates(candidates),
            ))

    if not grouped_clusters:
        raise RuntimeError(f"No finite C_e_gamma candidates found in {input_path}.")

    examples = example_rows(grouped_clusters)
    summaries = cluster_summary_rows(grouped_clusters)
    details = representative_rows(grouped_clusters)
    momentum_rows = momentum_configuration_rows(details)
    amplitude_rows = amplitude_decomposition_rows(details)
    spin_cases = []
    for spin_case, _region_name, _clusters in grouped_clusters:
        if spin_case not in spin_cases:
            spin_cases.append(spin_case)

    write_dict_csv(EXAMPLES_CSV_PATH, examples)
    write_dict_csv(CLUSTERS_CSV_PATH, summaries)
    write_dict_csv(MOMENTA_CSV_PATH, momentum_rows)
    write_dict_csv(AMPLITUDE_CSV_PATH, amplitude_rows)
    polarization_outputs = [
        ("examples", write_polarization_csvs(EXAMPLES_CSV_PATH, examples, spin_cases)),
        ("clusters", write_polarization_csvs(CLUSTERS_CSV_PATH, summaries, spin_cases)),
        (
            "momentum configurations",
            write_polarization_csvs(MOMENTA_CSV_PATH, momentum_rows, spin_cases),
        ),
        (
            "final-state amplitude decomposition",
            write_polarization_csvs(AMPLITUDE_CSV_PATH, amplitude_rows, spin_cases),
        ),
    ]
    save_configuration_plot(grouped_clusters)
    polarization_plot_outputs = save_polarization_plots(grouped_clusters)

    log_text = build_report(
        input_path,
        len(rows),
        grouped_clusters,
        examples,
        polarization_outputs,
        polarization_plot_outputs,
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text, end="")


if __name__ == "__main__":
    main()
