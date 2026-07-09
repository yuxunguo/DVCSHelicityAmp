"""Generate high-concurrence configuration scans from AlignmentScan outputs.

The generator reads the pairwise concurrence scan written by ``AlignmentScan.py``
and builds one configuration package for each requested maximum:
``C_e_p``, ``C_p_gamma``, and ``C_e_gamma``.  Each package contains a scan plot,
representative enhanced regions, reconstructed momenta/kinematics, and a
final-state amplitude decomposition.
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
    HELICITIES,
    M,
    SPIN_CASE_TRANSVERSE_TX,
    SPIN_CASE_TRANSVERSE_TY,
    amplitude_table,
    double_transverse_final_state,
    initial_spin_states,
    outgoing_spin_states,
    transverse_electron_coefficients,
)


ALIGNMENT_CONCURRENCE_DIR = Path("Output") / "AlignmentScan" / "ConcurrenceScan"
FULL_CONCURRENCE_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_concurrence_phase_space.csv"
RANKED_CONCURRENCE_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_concurrence_top.csv"
RANKED_E_GAMMA_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_e_gamma_top.csv"
LEGACY_RANKED_C13_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_c13_top.csv"

OUTPUT_DIR = Path("Output") / "ConfigGen"
DATA_DIR = OUTPUT_DIR / "Data"
EGAMMA_CONFIG_DIR = OUTPUT_DIR / "ByEgamma"
LOG_PATH = Path("Output") / "ConfigGen.log"

CONFIG_TARGETS = (
    ("C_e_p", "c_ep"),
    ("C_p_gamma", "c_p_gamma"),
    ("C_e_gamma", "c_e_gamma"),
)
CONFIG_SPIN_CASES = (
    "unpolarized",
    "longitudinal_polarized",
    "Tx",
    "Ty",
    "double_transverse",
)
MAX_SPIN_CASES_PER_TARGET = 2
TOP_ROWS_PER_TARGET_SPIN = 80
MAX_CLUSTERS_PER_TARGET_SPIN = 4
EXAMPLES_PER_CLUSTER = 2
CLUSTER_RADIUS = 0.42
SCAN_HEATMAP_MAX_BINS = 96

DISPLAY_MOMENTA = ("k", "p", "kp", "pp", "qout")
MOMENTUM_DISPLAY_LABELS = {
    "k": r"$\ell$",
    "p": r"$P$",
    "kp": r"$\ell'$",
    "pp": r"$P'$",
    "qout": r"$q_\gamma$",
}
TARGET_FINAL_MOMENTA = {
    "C_e_p": ("kp", "pp"),
    "C_e_gamma": ("kp", "qout"),
    "C_p_gamma": ("pp", "qout"),
}

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


def clean_legacy_root_csv_outputs():
    """Remove stale root outputs from older ConfigGen layouts."""
    if not OUTPUT_DIR.exists():
        return
    for path in OUTPUT_DIR.glob("max_*.csv"):
        path.unlink()
    for path in OUTPUT_DIR.glob("max_*configuration_scan*.pdf"):
        path.unlink()
    summary_path = OUTPUT_DIR / "max_concurrence_by_egamma_configurations.pdf"
    if summary_path.exists():
        summary_path.unlink()


def clean_egamma_config_outputs():
    """Remove stale per-E_gamma PDF configuration outputs."""
    if not EGAMMA_CONFIG_DIR.exists():
        return
    for path in EGAMMA_CONFIG_DIR.glob("**/*.pdf"):
        path.unlink()
    for path in sorted(EGAMMA_CONFIG_DIR.glob("**/*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def alignment_input_path():
    """Return the best available concurrence CSV for configuration scans."""
    for path in (
        FULL_CONCURRENCE_CSV,
        RANKED_CONCURRENCE_CSV,
        RANKED_E_GAMMA_CSV,
        LEGACY_RANKED_C13_CSV,
    ):
        if path.exists():
            return path
    raise FileNotFoundError(
        "No AlignmentScan concurrence CSV found. Run AlignmentScan.py first to "
        f"create {FULL_CONCURRENCE_CSV}."
    )


def target_paths(file_tag):
    """Return output paths for one requested concurrence target."""
    prefix = f"max_{file_tag}"
    return {
        "examples": DATA_DIR / f"{prefix}_configuration_examples.csv",
        "clusters": DATA_DIR / f"{prefix}_cluster_summary.csv",
        "momenta": DATA_DIR / f"{prefix}_momentum_configurations.csv",
        "amplitudes": DATA_DIR / f"{prefix}_final_state_amplitude_decomposition.csv",
    }


def spin_label_from_key(key, observable):
    """Return the spin-case prefix from a concurrence column name."""
    suffix = f"_{observable}"
    if key.endswith(suffix):
        return key[: -len(suffix)]
    if observable == "C_e_gamma" and key.endswith("_C13"):
        return key[:-4]
    return key


def format_helicity(value):
    """Return a signed helicity label."""
    helicity = int(value)
    return f"{helicity:+d}"


def target_columns(rows, observable):
    """Return configured spin columns present for a target observable."""
    if not rows:
        return []
    names = set(rows[0])
    columns = []
    for spin_case in CONFIG_SPIN_CASES:
        key = f"{spin_case}_{observable}"
        if key in names:
            columns.append(key)
    if observable == "C_e_gamma":
        for spin_case in CONFIG_SPIN_CASES:
            key = f"{spin_case}_C13"
            if key in names:
                columns.append(key)
    return columns


def column_max(rows, key):
    """Return the finite maximum of a CSV column."""
    values = [parse_float(row.get(key)) for row in rows]
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if values.size == 0:
        return -np.inf
    return float(values.max())


def selected_target_columns(rows, observable):
    """Return the strongest spin-case columns for one target observable."""
    columns = target_columns(rows, observable)
    if not columns:
        return []
    ranked = sorted(columns, key=lambda key: column_max(rows, key), reverse=True)
    if MAX_SPIN_CASES_PER_TARGET is None:
        return ranked
    return ranked[:MAX_SPIN_CASES_PER_TARGET]


def circular_distance(a, b):
    """Return angular distance on [0, 2*pi)."""
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    diff = abs(float(a) - float(b)) % (2.0 * math.pi)
    return min(diff, 2.0 * math.pi - diff)


def vector_phi_xy(vector):
    """Return azimuth of a four-vector's transverse momentum."""
    vector = np.asarray(vector, dtype=float)
    return float(math.atan2(vector[2], vector[1]) % (2.0 * math.pi))


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


def target_pair_delta(row, observable):
    """Return transverse angular separation for the target outgoing pair."""
    first, second = TARGET_FINAL_MOMENTA[observable]
    kin = kinematics_from_config_row(row)
    return circular_distance(
        vector_phi_xy(kin["momenta"][first]),
        vector_phi_xy(kin["momenta"][second]),
    )


def source_rows_for_key(rows, key):
    """Return rows appropriate for a target/spin column."""
    if rows and "rank_group" in rows[0]:
        group_rows = [row for row in rows if row.get("rank_group") == key]
        return group_rows if group_rows else rows
    return rows


def selected_row(row, key, observable, value):
    """Return a row annotated with the selected target observable metadata."""
    item = dict(row)
    item["selected_observable"] = observable
    item["selected_observable_label"] = observable_text_label(observable)
    item["selected_spin_case"] = spin_label_from_key(key, observable)
    item["selected_concurrence_key"] = key
    item["selected_concurrence"] = value
    item["pair_delta_xy"] = target_pair_delta(item, observable)
    item["scan_phi_e_in"] = parse_float(row.get("phi_in_electron"))
    item["scan_phi_p_in"] = parse_float(row.get("phi_in"))
    item["scan_phi_gamma"] = parse_float(row.get("phiOut"))
    return item


def scan_x_phi(row):
    """Return the proton incoming azimuth used as the scan-map x coordinate."""
    value = parse_float(row.get("phi_in"))
    if np.isfinite(value):
        return value
    return parse_float(row.get("phi_in_electron"))


def candidate_rows(rows, key, observable):
    """Return ranked candidates for one target/spin concurrence column."""
    source_rows = source_rows_for_key(rows, key)
    candidates = []
    seen = set()
    for row in source_rows:
        if row.get("rank_group") and row.get("rank_group") != key and key not in row:
            continue
        value = parse_float(row.get("rank_value") if row.get("rank_group") == key else row.get(key))
        if not np.isfinite(value):
            continue
        identity = tuple(row.get(name, "") for name in KINEMATIC_COLUMNS)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(selected_row(row, key, observable, value))
    candidates.sort(key=lambda item: item["selected_concurrence"], reverse=True)
    return candidates[:TOP_ROWS_PER_TARGET_SPIN]


def row_distance(a, b):
    """Return a normalized distance between two user-frame configurations."""
    s_scale = max(abs(parse_float(a.get("s"))), abs(parse_float(b.get("s"))), 1.0)
    q_scale = max(abs(parse_float(a.get("qOut"))), abs(parse_float(b.get("qOut"))), 1.0)
    pieces = [
        (parse_float(a.get("s")) - parse_float(b.get("s"))) / s_scale,
        (parse_float(a.get("theta_in")) - parse_float(b.get("theta_in"))) / math.pi,
        circular_distance(scan_x_phi(a), scan_x_phi(b)) / math.pi,
        (parse_float(a.get("qOut")) - parse_float(b.get("qOut"))) / q_scale,
        circular_distance(parse_float(a.get("phiOut")), parse_float(b.get("phiOut"))) / math.pi,
    ]
    return float(np.sqrt(np.sum(np.asarray(pieces, dtype=float) ** 2)))


def cluster_candidates(candidates):
    """Greedily cluster high-concurrence candidate rows by scan coordinates."""
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
            if row["selected_concurrence"] > clusters[best_index]["best"]["selected_concurrence"]:
                clusters[best_index]["best"] = row
        elif len(clusters) < MAX_CLUSTERS_PER_TARGET_SPIN:
            clusters.append({"best": row, "rows": [row]})
        else:
            nearest = min(range(len(clusters)), key=lambda index: row_distance(row, clusters[index]["best"]))
            clusters[nearest]["rows"].append(row)

    for index, cluster in enumerate(clusters):
        cluster["cluster_id"] = index
        cluster["rows"].sort(key=lambda item: item["selected_concurrence"], reverse=True)
        cluster["best"] = cluster["rows"][0]
        for row in cluster["rows"]:
            row["cluster_id"] = index
            row["selected_region"] = f"enhanced_region_{index}"
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


def cluster_summary_rows(target, grouped_clusters):
    """Return dictionaries for the cluster summary CSV."""
    summaries = []
    observable, _file_tag = target
    for spin_case, clusters in grouped_clusters:
        for cluster in clusters:
            rows = cluster["rows"]
            best = cluster["best"]
            summary = {
                "selected_observable": observable,
                "selected_observable_label": observable_text_label(observable),
                "selected_spin_case": spin_case,
                "selected_region": f"enhanced_region_{cluster['cluster_id']}",
                "cluster_id": cluster["cluster_id"],
                "size": len(rows),
                "max_concurrence": f"{best['selected_concurrence']:.16e}",
                "best_kinematic_point": best.get("kinematic_point", ""),
                "best_pair_delta_xy": f"{best['pair_delta_xy']:.16e}",
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
                "pair_delta_xy",
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
    for spin_case, clusters in grouped_clusters:
        for cluster in clusters:
            for rank, row in enumerate(cluster["rows"][:EXAMPLES_PER_CLUSTER], start=1):
                item = {
                    "selected_observable": row["selected_observable"],
                    "selected_observable_label": row["selected_observable_label"],
                    "selected_spin_case": spin_case,
                    "selected_region": row["selected_region"],
                    "cluster_id": cluster["cluster_id"],
                    "example_rank": rank,
                    "selected_concurrence": f"{row['selected_concurrence']:.16e}",
                    "pair_delta_xy": f"{row['pair_delta_xy']:.16e}",
                }
                for name in KINEMATIC_COLUMNS:
                    item[name] = row.get(name, "")
                for key, value in row.items():
                    if "_C_" in key or key.endswith("_C13"):
                        item[key] = value
                examples.append(item)
    return examples


def representative_rows(target, grouped_clusters):
    """Return one characteristic row per selected target/spin/cluster."""
    rows = []
    observable, file_tag = target
    for spin_case, clusters in grouped_clusters:
        for cluster in clusters:
            row = dict(cluster["best"])
            row["selected_spin_case"] = spin_case
            row["selected_region"] = f"enhanced_region_{cluster['cluster_id']}"
            row["cluster_id"] = cluster["cluster_id"]
            row["detail_id"] = f"{file_tag}_{spin_case}_region_{cluster['cluster_id']}"
            row["selected_observable"] = observable
            row["detail_source"] = "enhanced_cluster"
            rows.append(row)
    return rows


def file_safe_label(value):
    """Return a compact filesystem-safe label."""
    text = str(value).strip() or "unknown"
    return "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")


def qout_group_key(row):
    """Return a stable grouping key for one outgoing photon energy."""
    regime = row.get("qOut_regime") or ""
    qout = parse_float(row.get("qOut"))
    if regime:
        return (regime, qout)
    if np.isfinite(qout):
        return (f"Egamma_{qout:.6g}", qout)
    return ("Egamma_unknown", np.inf)


def energy_representative_rows(target, _grouped_clusters, rows):
    """Return the best selected configuration at each sampled photon energy."""
    observable, file_tag = target
    grouped = {}
    for key in target_columns(rows, observable):
        for row in source_rows_for_key(rows, key):
            value = parse_float(
                row.get("rank_value") if row.get("rank_group") == key else row.get(key)
            )
            if not np.isfinite(value):
                continue
            group_name, qout = qout_group_key(row)
            selected = selected_row(row, key, observable, value)
            selected["qOut_group"] = group_name
            selected["qOut_group_value"] = qout
            current = grouped.get(group_name)
            if current is None or selected["selected_concurrence"] > current["selected_concurrence"]:
                grouped[group_name] = selected
    details = []
    sorted_rows = sorted(
        grouped.values(),
        key=lambda item: (
            parse_float(item.get("qOut_group_value"), default=np.inf),
            item.get("qOut_group", ""),
        ),
    )
    for index, row in enumerate(sorted_rows):
        label = file_safe_label(row.get("qOut_group") or f"Egamma_{parse_float(row.get('qOut')):.6g}")
        row["selected_region"] = row.get("qOut_group", f"Egamma_{index}")
        row["cluster_id"] = f"Egamma_{index}"
        row["detail_id"] = f"{file_tag}_{row['selected_spin_case']}_{label}"
        row["selected_observable"] = observable
        row["detail_source"] = "egamma_best"
        details.append(row)
    return details


def max_target_rows_by_egamma(rows):
    """Return one max-concurrence row per E_gamma and target observable."""
    grouped = {}
    for observable, file_tag in CONFIG_TARGETS:
        for key in target_columns(rows, observable):
            for row in source_rows_for_key(rows, key):
                value = parse_float(row.get("rank_value") if row.get("rank_group") == key else row.get(key))
                if not np.isfinite(value):
                    continue
                group_name, qout = qout_group_key(row)
                selected = selected_row(row, key, observable, value)
                selected["qOut_group"] = group_name
                selected["qOut_group_value"] = qout
                selected["selected_region"] = group_name
                selected["cluster_id"] = f"{file_tag}_{file_safe_label(group_name)}"
                selected["detail_id"] = f"{file_tag}_{file_safe_label(group_name)}"
                selected["detail_source"] = "egamma_target_max"
                current = grouped.get((group_name, observable))
                if current is None or selected["selected_concurrence"] > current["selected_concurrence"]:
                    grouped[(group_name, observable)] = selected
    return sorted(
        grouped.values(),
        key=lambda row: (
            parse_float(row.get("qOut_group_value"), default=np.inf),
            row.get("qOut_group", ""),
            [target[0] for target in CONFIG_TARGETS].index(row["selected_observable"]),
        ),
    )


def qout_groups(rows):
    """Return sorted photon-energy groups present in scan rows."""
    groups = {}
    for row in rows:
        group_name, qout = qout_group_key(row)
        groups[group_name] = qout
    return sorted(groups.items(), key=lambda item: (parse_float(item[1], default=np.inf), item[0]))


def rows_for_qout_group(rows, group_name):
    """Return scan rows belonging to one photon-energy group."""
    return [row for row in rows if qout_group_key(row)[0] == group_name]


def target_egamma_candidates(rows, target):
    """Return high-concurrence candidates for one target at one E_gamma."""
    observable, _file_tag = target
    candidates = []
    seen = set()
    for key in target_columns(rows, observable):
        for row in source_rows_for_key(rows, key):
            value = parse_float(row.get("rank_value") if row.get("rank_group") == key else row.get(key))
            if not np.isfinite(value):
                continue
            identity = (
                key,
                *tuple(row.get(name, "") for name in KINEMATIC_COLUMNS),
            )
            if identity in seen:
                continue
            seen.add(identity)
            selected = selected_row(row, key, observable, value)
            selected["qOut_group"] = qout_group_key(row)[0]
            selected["qOut_group_value"] = qout_group_key(row)[1]
            candidates.append(selected)
    candidates.sort(key=lambda item: item["selected_concurrence"], reverse=True)
    return candidates[:TOP_ROWS_PER_TARGET_SPIN]


def egamma_target_region_rows(target, group_name, clusters):
    """Return representative detail rows for one E_gamma/target region set."""
    observable, file_tag = target
    detail_rows = []
    for cluster in clusters:
        row = dict(cluster["best"])
        row["selected_observable"] = observable
        row["selected_region"] = f"{group_name}_region_{cluster['cluster_id']}"
        row["cluster_id"] = cluster["cluster_id"]
        row["detail_id"] = f"{file_tag}_{file_safe_label(group_name)}_region_{cluster['cluster_id']}"
        row["detail_source"] = "egamma_region_max"
        detail_rows.append(row)
    return detail_rows


def max_concurrence_for_row(row, observable):
    """Return the maximum target concurrence over configured spin columns for one row."""
    best_value = -np.inf
    best_key = ""
    for key in target_columns([row], observable):
        value = parse_float(row.get("rank_value") if row.get("rank_group") == key else row.get(key))
        if np.isfinite(value) and value > best_value:
            best_value = value
            best_key = key
    return best_value, best_key


def write_dict_csv(path, rows):
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        if headers:
            writer.writeheader()
            writer.writerows(rows)
    return path


def spin_file_tag(spin_case):
    """Return a filesystem-friendly tag for one selected polarization."""
    tag = "".join(char.lower() if char.isalnum() else "_" for char in str(spin_case))
    return tag.strip("_") or "unknown"


def spin_output_path(base_path, spin_case):
    """Return the per-spin output path next to an aggregate path."""
    return base_path.with_name(f"{base_path.stem}_{spin_file_tag(spin_case)}{base_path.suffix}")


def write_spin_csvs(base_path, rows, spin_cases):
    """Write one filtered CSV per selected spin case."""
    outputs = []
    for spin_case in spin_cases:
        spin_rows = [row for row in rows if row.get("selected_spin_case") == spin_case]
        outputs.append((spin_case, write_dict_csv(spin_output_path(base_path, spin_case), spin_rows)))
    return outputs


def momentum_configuration_rows(detail_rows):
    """Return one CSV row per four-momentum in each detailed configuration."""
    records = []
    for row in detail_rows:
        kin = kinematics_from_config_row(row)
        for name in DISPLAY_MOMENTA:
            vector = kin["momenta"][name]
            records.append({
                "detail_id": row["detail_id"],
                "detail_source": row.get("detail_source", ""),
                "selected_observable": row["selected_observable"],
                "selected_observable_label": row["selected_observable_label"],
                "selected_spin_case": row["selected_spin_case"],
                "selected_region": row["selected_region"],
                "cluster_id": row["cluster_id"],
                "selected_concurrence": f"{row['selected_concurrence']:.16e}",
                "pair_delta_xy": f"{row['pair_delta_xy']:.16e}",
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
        return amplitudes[in_states.index(ENTANGLEMENT_INITIAL_STATE)]
    if spin_case == "longitudinal_polarized":
        return (
            amplitudes[in_states.index((+1, proton_spin))]
            - amplitudes[in_states.index((-1, proton_spin))]
        ) / np.sqrt(2.0)
    if spin_case == "Tx":
        coefficients = transverse_electron_coefficients(SPIN_CASE_TRANSVERSE_TX)
        return sum(
            coefficients[h_in] * amplitudes[in_states.index((h_in, proton_spin))]
            for h_in in HELICITIES
        )
    if spin_case == "Ty":
        coefficients = transverse_electron_coefficients(SPIN_CASE_TRANSVERSE_TY)
        return sum(
            coefficients[h_in] * amplitudes[in_states.index((h_in, proton_spin))]
            for h_in in HELICITIES
        )
    if spin_case == "double_transverse":
        return double_transverse_final_state(amplitudes)
    raise ValueError(f"Unknown ConfigGen spin case: {spin_case}")


def selected_initial_state_label(spin_case):
    """Return a compact label for the incoming spin state behind a plot."""
    proton_spin = ENTANGLEMENT_INITIAL_STATE[1]
    proton_label = format_helicity(proton_spin)
    if spin_case == "unpolarized":
        return rf"$A(h_e={format_helicity(ENTANGLEMENT_INITIAL_STATE[0])}, h_p={proton_label})$"
    if spin_case == "longitudinal_polarized":
        return rf"$[A(h_e=+1,h_p={proton_label}) - A(h_e=-1,h_p={proton_label})]/\sqrt{{2}}$"
    if spin_case == "Tx":
        return rf"$[A(h_e=+1,h_p={proton_label}) + A(h_e=-1,h_p={proton_label})]/\sqrt{{2}}$"
    if spin_case == "Ty":
        return rf"$[A(h_e=+1,h_p={proton_label}) + i A(h_e=-1,h_p={proton_label})]/\sqrt{{2}}$"
    if spin_case == "double_transverse":
        return r"$T_x(h_e) \otimes T_x(h_p)$ coherent incoming state"
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
            "detail_source": row.get("detail_source", ""),
            "selected_observable": row["selected_observable"],
            "selected_observable_label": row["selected_observable_label"],
            "selected_spin_case": row["selected_spin_case"],
            "selected_region": row["selected_region"],
            "cluster_id": row["cluster_id"],
            "selected_concurrence": f"{row['selected_concurrence']:.16e}",
            "pair_delta_xy": f"{row['pair_delta_xy']:.16e}",
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


def _bin_edges_from_values(values, max_bins=SCAN_HEATMAP_MAX_BINS):
    """Return plotting bin edges adapted to discrete or continuous values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([0.0, 1.0])
    unique = np.unique(values)
    if unique.size == 1:
        width = max(1.0e-6, abs(unique[0]) * 1.0e-6)
        return np.asarray([unique[0] - width, unique[0] + width])
    if unique.size <= max_bins:
        midpoints = 0.5 * (unique[:-1] + unique[1:])
        first = unique[0] - 0.5 * (unique[1] - unique[0])
        last = unique[-1] + 0.5 * (unique[-1] - unique[-2])
        return np.concatenate([[first], midpoints, [last]])
    return np.linspace(values.min(), values.max(), max_bins + 1)


def _binned_mean_2d(x_values, y_values, z_values, x_edges, y_edges):
    """Return a masked 2D binned mean ``z`` on ``x``/``y`` bins."""
    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values)
    counts, _x_edges, _y_edges = np.histogram2d(
        x_values[finite],
        y_values[finite],
        bins=(x_edges, y_edges),
    )
    sums, _x_edges, _y_edges = np.histogram2d(
        x_values[finite],
        y_values[finite],
        bins=(x_edges, y_edges),
        weights=z_values[finite],
    )
    mean = np.full_like(sums, np.nan, dtype=float)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return np.ma.masked_invalid(mean.T)


def add_pi_over_two_reference_lines(ax):
    """Draw requested pi/2 reference lines on scan maps."""
    ax.axvline(0.5 * math.pi, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.axhline(0.5 * math.pi, color="white", linestyle="--", linewidth=1.0, alpha=0.9)


def plot_scan_map(plt, pdf, rows, key, observable, spin_case, clusters):
    """Append one concurrence scan page with enhanced-region markers."""
    source_rows = source_rows_for_key(rows, key)
    x = np.asarray([scan_x_phi(row) for row in source_rows], dtype=float)
    y = np.asarray([parse_float(row.get("phiOut")) for row in source_rows], dtype=float)
    z = np.asarray([
        parse_float(row.get("rank_value") if row.get("rank_group") == key else row.get(key))
        for row in source_rows
    ], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    if np.isfinite(z).sum() >= 4:
        x_edges = _bin_edges_from_values(x)
        y_edges = _bin_edges_from_values(y)
        values = _binned_mean_2d(x, y, z, x_edges, y_edges)
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        image = ax.scatter(
            x,
            y,
            c=z,
            cmap="viridis",
            s=28,
            alpha=0.78,
            vmin=0.0,
            vmax=1.0,
        )
    for cluster in clusters:
        best = cluster["best"]
        ax.scatter(
            [scan_x_phi(best)],
            [parse_float(best.get("phiOut"))],
            marker="x",
            s=90,
            color="black",
            linewidths=1.5,
        )
        ax.text(
            scan_x_phi(best),
            parse_float(best.get("phiOut")),
            f" region {cluster['cluster_id']}",
            fontsize=8,
            va="center",
        )
    ax.set_title(
        f"{spin_case}: scan for max {observable_latex_label(observable)}",
        fontsize=14,
    )
    add_pi_over_two_reference_lines(ax)
    ax.set_xlabel(r"$\phi_{P,\rm in}$ [rad]", fontsize=12)
    ax.set_ylabel(r"$\phi_{\gamma}'$ [rad]", fontsize=12)
    ax.set_xlim(0.0, 2.0 * math.pi)
    ax.set_ylim(0.0, 2.0 * math.pi)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(observable_latex_label(observable), fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def plot_egamma_target_scan_map(plt, pdf, rows, target, group_name, clusters):
    """Append one fixed-E_gamma target scan map with region markers."""
    observable, _file_tag = target
    x_values = []
    y_values = []
    z_values = []
    spin_labels = []
    for row in rows:
        value, key = max_concurrence_for_row(row, observable)
        if not np.isfinite(value):
            continue
        x_values.append(scan_x_phi(row))
        y_values.append(parse_float(row.get("phiOut")))
        z_values.append(value)
        spin_labels.append(spin_label_from_key(key, observable))
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    z = np.asarray(z_values, dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    if np.isfinite(z).sum() >= 4:
        x_edges = _bin_edges_from_values(x)
        y_edges = _bin_edges_from_values(y)
        values = _binned_mean_2d(x, y, z, x_edges, y_edges)
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        image = ax.scatter(
            x,
            y,
            c=z,
            cmap="viridis",
            s=28,
            alpha=0.78,
            vmin=0.0,
            vmax=1.0,
        )
    for cluster in clusters:
        best = cluster["best"]
        ax.scatter(
            [scan_x_phi(best)],
            [parse_float(best.get("phiOut"))],
            marker="x",
            s=90,
            color="black",
            linewidths=1.5,
        )
        ax.text(
            scan_x_phi(best),
            parse_float(best.get("phiOut")),
            f" region {cluster['cluster_id']} ({best['selected_spin_case']})",
            fontsize=8,
            va="center",
        )
    ax.set_title(
        f"{group_name}: max {observable_latex_label(observable)} regions",
        fontsize=14,
    )
    add_pi_over_two_reference_lines(ax)
    ax.set_xlabel(r"$\phi_{P,\rm in}$ [rad]", fontsize=12)
    ax.set_ylabel(r"$\phi_{\gamma}$ [rad]", fontsize=12)
    ax.set_xlim(0.0, 2.0 * math.pi)
    ax.set_ylim(0.0, 2.0 * math.pi)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(observable_latex_label(observable), fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def momentum_kind(name):
    """Return the visual particle type for a momentum label."""
    if name in {"k", "kp"}:
        return "electron"
    if name in {"p", "pp"}:
        return "proton"
    if name == "qout":
        return "photon"
    return "other"


def momentum_display_label(name):
    """Return a math display label for a momentum name."""
    return MOMENTUM_DISPLAY_LABELS.get(name, name)


def perpendicular_2d(delta):
    """Return a unit vector perpendicular to a transverse momentum."""
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-14:
        return np.asarray([0.0, 1.0])
    return np.asarray([-delta[1], delta[0]]) / norm


def _plot_arrow_2d(ax, start, end, color, linestyle="-", linewidth=1.6):
    """Draw a styled 2D arrow."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linewidth": linewidth,
            "linestyle": linestyle,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )


def _plot_wavy_2d(ax, start, end, color, amplitude):
    """Draw a wavy 2D photon line with an arrow head."""
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-14:
        return
    direction = delta / length
    normal = perpendicular_2d(delta)
    t = np.linspace(0.0, 1.0, 120)
    wave = start + t[:, np.newaxis] * delta + amplitude * np.sin(10.0 * np.pi * t)[:, np.newaxis] * normal
    ax.plot(wave[:, 0], wave[:, 1], color=color, linewidth=1.5)
    _plot_arrow_2d(ax, wave[-8], end, color, linewidth=1.2)


def _plot_vector_2d(ax, vector, label, color, line_scale):
    """Draw one styled transverse momentum vector."""
    end = np.asarray(vector, dtype=float)[1:3]
    start = np.zeros(2)
    kind = momentum_kind(label)
    if kind == "proton":
        _plot_arrow_2d(ax, start, end, color, linewidth=1.75)
    elif kind == "photon":
        _plot_wavy_2d(ax, start, end, color, amplitude=0.025 * line_scale)
    elif kind == "electron":
        _plot_arrow_2d(ax, start, end, color, linestyle="--", linewidth=1.65)
    else:
        _plot_arrow_2d(ax, start, end, color, linewidth=1.65)
    ax.text(end[0], end[1], f" {momentum_display_label(label)}", color=color, fontsize=11, va="center")


def _perpendicular_3d(delta):
    """Return a stable unit vector perpendicular to a 3D direction."""
    delta = np.asarray(delta, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm <= 1.0e-14:
        return np.asarray([0.0, 1.0, 0.0])
    direction = delta / norm
    trial = np.asarray([0.0, 0.0, 1.0])
    perp = np.cross(direction, trial)
    if np.linalg.norm(perp) <= 1.0e-12:
        trial = np.asarray([0.0, 1.0, 0.0])
        perp = np.cross(direction, trial)
    return perp / np.linalg.norm(perp)


def _plot_line_arrow_3d(ax, start, end, color, linestyle="-", linewidth=1.5):
    """Draw a styled 3D line with a short arrow head."""
    delta = end - start
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    head_start = start + 0.88 * delta
    head_delta = end - head_start
    ax.quiver(
        head_start[0],
        head_start[1],
        head_start[2],
        head_delta[0],
        head_delta[1],
        head_delta[2],
        color=color,
        arrow_length_ratio=0.45,
        linewidth=linewidth,
    )


def _plot_vector_3d(ax, vector, label, color, line_scale):
    """Draw one styled 3D momentum vector."""
    end = np.asarray(vector, dtype=float)[1:4]
    start = np.zeros(3)
    kind = momentum_kind(label)
    if kind == "proton":
        _plot_line_arrow_3d(ax, start, end, color, linewidth=1.75)
    elif kind == "photon":
        _plot_line_arrow_3d(ax, start, end, color, linestyle=":", linewidth=1.6)
    elif kind == "electron":
        _plot_line_arrow_3d(ax, start, end, color, linestyle="--", linewidth=1.55)
    else:
        _plot_line_arrow_3d(ax, start, end, color, linewidth=1.55)
    ax.text(end[0], end[1], end[2], f" {momentum_display_label(label)}", color=color, fontsize=9)


def plot_transverse_momenta(ax, kin, title="Transverse plane"):
    """Draw styled transverse momentum vectors."""
    colors = {
        "k": "tab:blue",
        "p": "tab:orange",
        "kp": "tab:cyan",
        "pp": "tab:red",
        "qout": "tab:green",
    }
    momenta = kin["momenta"]
    transverse = np.asarray([momenta[name][1:3] for name in DISPLAY_MOMENTA])
    line_scale = max(1.0, float(np.nanmax(np.abs(transverse))) * 1.20)
    for name in DISPLAY_MOMENTA:
        _plot_vector_2d(ax, momenta[name], name, colors[name], line_scale)
    ax.set_xlim(-line_scale, line_scale)
    ax.set_ylim(-line_scale, line_scale)
    ax.axhline(0.0, color="0.82", linewidth=0.8)
    ax.axvline(0.0, color="0.82", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(r"$p_x$ [GeV]")
    ax.set_ylabel(r"$p_y$ [GeV]")


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
    vectors = np.asarray([momenta[name][1:4] for name in DISPLAY_MOMENTA])
    line_scale = max(1.0, float(np.nanmax(np.abs(vectors))) * 1.15)
    ax3d = fig.add_subplot(grid_spec[0, 0], projection="3d")
    ax2d = fig.add_subplot(grid_spec[1, 0])
    for name in DISPLAY_MOMENTA:
        _plot_vector_3d(ax3d, momenta[name], name, colors[name], line_scale)
    ax3d.set_xlim(-line_scale, line_scale)
    ax3d.set_ylim(-line_scale, line_scale)
    ax3d.set_zlim(-line_scale, line_scale)
    ax3d.set_title("3D momenta")
    ax3d.set_xlabel(r"$p_x$ [GeV]")
    ax3d.set_ylabel(r"$p_y$ [GeV]")
    ax3d.set_zlabel(r"$p_z$ [GeV]")
    plot_transverse_momenta(ax2d, kin)


def format_vector_line(name, vector):
    """Return one compact four-vector line."""
    return (
        f"{momentum_display_label(name):10s} $E$={vector[0]:8.4f} "
        f"$p_x$={vector[1]:8.4f} $p_y$={vector[2]:8.4f} $p_z$={vector[3]:8.4f}"
    )


def plot_configuration_text(ax, row, kin):
    """Draw kinematic and momentum text for a detailed configuration page."""
    ax.axis("off")
    momenta = kin["momenta"]
    label = observable_text_label(row["selected_observable"])
    lines = [
        f"{row['detail_id']}  {label}={row['selected_concurrence']:.6g}",
        (
            f"region: {row['selected_region']}  "
            f"final-pair delta_xy={row['pair_delta_xy']:.6g} rad"
        ),
        f"kinematic point: {row.get('kinematic_point', '')}",
        f"incoming state: {selected_initial_state_label(row['selected_spin_case'])}",
        "",
        (
            rf"$s$={kin['s']:.6g}, $\sqrt{{s}}$={kin['sqrt_s']:.6g}, "
            rf"$|\vec{{P}}|$={kin['pIn']:.6g}, $|\vec{{P}}^{{\,\prime}}|$={kin['pOut']:.6g}"
        ),
        (
            rf"$\theta_{{\rm in}}$={kin['theta_in']:.6g}, "
            rf"$\phi_\ell$={parse_float(row.get('phi_in_electron')):.6g}, "
            rf"$\phi_P$={kin['phi_in']:.6g}"
        ),
        rf"$E_\gamma$={kin['qOut']:.6g}, $\phi_\gamma$={kin['phiOut']:.6g}",
        (
            rf"$Q^2$={kin['Q2']:.6g}, $x_B$={kin['xB']:.6g}, "
            rf"$t$={kin['t']:.6g}, $W^2$={kin['W2']:.6g}, $y$={kin['y']:.6g}"
        ),
        rf"$\theta(\ell',\gamma)$={parse_float(row.get('theta_e_gamma_deg')):.6g} deg",
        "",
        r"four-momenta [$E$, $p_x$, $p_y$, $p_z$] GeV:",
    ]
    lines.extend(format_vector_line(name, momenta[name]) for name in DISPLAY_MOMENTA)
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)


def format_amplitude_summary(row, max_rows=3):
    """Return a compact text summary of the largest amplitude fractions."""
    records = amplitude_decomposition(row)[:max_rows]
    lines = []
    for item in records:
        lines.append(
            "  "
            rf"$h_e$={format_helicity(item['hOut'])}, "
            rf"$h_p$={format_helicity(item['sOut'])}, "
            rf"$h_\gamma$={format_helicity(item['lambda'])}: "
            f"frac={parse_float(item['fraction']):.3g}, "
            f"phase={parse_float(item['amplitude_phase']):.3g}"
        )
    return lines


def plot_egamma_row_text(ax, row, kin):
    """Draw compact kinematics and amplitude text for an E-gamma row."""
    ax.axis("off")
    label = observable_text_label(row["selected_observable"])
    lines = [
        (
            rf"{row['selected_spin_case']}  {row.get('qOut_group', row.get('qOut_regime', 'Egamma'))}  "
            f"{label}={row['selected_concurrence']:.6g}"
        ),
        (
            rf"$E_\gamma$={kin['qOut']:.6g}, $\phi_\gamma$={kin['phiOut']:.6g}, "
            rf"$\phi_\ell$={parse_float(row.get('phi_in_electron')):.6g}, "
            rf"$\phi_P$={kin['phi_in']:.6g}"
        ),
        (
            rf"$Q^2$={kin['Q2']:.6g}, $x_B$={kin['xB']:.6g}, "
            rf"$t$={kin['t']:.6g}, $W^2$={kin['W2']:.6g}, $y$={kin['y']:.6g}"
        ),
        f"final-pair delta_xy={row['pair_delta_xy']:.6g} rad",
        "largest final-state spin/helicity amplitudes:",
        *format_amplitude_summary(row),
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.5)


def save_egamma_summary_pages(pdf, plt, target, energy_rows):
    """Append pages collecting one selected configuration per E-gamma row."""
    if not energy_rows:
        return
    observable, _file_tag = target
    by_spin = {}
    for row in energy_rows:
        by_spin.setdefault(row["selected_spin_case"], []).append(row)
    for spin_case, rows in by_spin.items():
        rows = sorted(rows, key=lambda row: parse_float(row.get("qOut_group_value"), default=np.inf))
        for start in range(0, len(rows), 4):
            chunk = rows[start:start + 4]
            fig_height = max(5.0, 2.25 * len(chunk))
            fig, axes = plt.subplots(
                len(chunk),
                2,
                figsize=(12.8, fig_height),
                constrained_layout=True,
                squeeze=False,
                gridspec_kw={"width_ratios": (1.0, 1.45)},
            )
            for row_index, row in enumerate(chunk):
                kin = kinematics_from_config_row(row)
                title = f"{row.get('qOut_group', 'Egamma')} momenta"
                plot_transverse_momenta(axes[row_index, 0], kin, title=title)
                plot_egamma_row_text(axes[row_index, 1], row, kin)
            fig.suptitle(
                (
                    f"{spin_case}: best E_gamma configurations for max "
                    f"{observable_latex_label(observable)}"
                ),
                fontsize=15,
            )
            pdf.savefig(fig)
            plt.close(fig)


def egamma_pdf_path(base_dir, group_name):
    """Return the standalone PDF path for one E_gamma group."""
    return base_dir / f"{file_safe_label(group_name)}_configuration.pdf"


def target_sort_index(row):
    """Return display order for target rows."""
    targets = [target[0] for target in CONFIG_TARGETS]
    return targets.index(row["selected_observable"])


def save_single_egamma_config_pdf(plt, PdfPages, rows, path):
    """Save one standalone configuration PDF for one E_gamma group."""
    rows = sorted(rows, key=target_sort_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for start in range(0, len(rows), 3):
            chunk = rows[start:start + 3]
            fig_height = max(6.0, 2.55 * len(chunk))
            fig, axes = plt.subplots(
                len(chunk),
                2,
                figsize=(12.8, fig_height),
                constrained_layout=True,
                squeeze=False,
                gridspec_kw={"width_ratios": (1.0, 1.45)},
            )
            for row_index, row in enumerate(chunk):
                kin = kinematics_from_config_row(row)
                plot_transverse_momenta(
                    axes[row_index, 0],
                    kin,
                    title=(
                        rf"max {observable_latex_label(row['selected_observable'])}  "
                        rf"{row['selected_spin_case']}  $E_\gamma$={kin['qOut']:.6g}"
                    ),
                )
                plot_egamma_row_text(axes[row_index, 1], row, kin)
            group_name = rows[0].get("qOut_group", "Egamma")
            fig.suptitle(
                rf"{group_name}: max pairwise-concurrence configurations",
                fontsize=15,
            )
            pdf.savefig(fig)
            plt.close(fig)
    return path


def save_egamma_config_pdfs(energy_rows, base_dir):
    """Write one standalone configuration PDF per sampled E_gamma value."""
    if not energy_rows:
        return []
    plt, PdfPages = _require_matplotlib()
    grouped = {}
    for row in energy_rows:
        grouped.setdefault(row.get("qOut_group", "Egamma"), []).append(row)
    outputs = []
    for group_name, rows in sorted(
        grouped.items(),
        key=lambda item: parse_float(item[1][0].get("qOut_group_value"), default=np.inf),
    ):
        path = egamma_pdf_path(base_dir, group_name)
        outputs.append((group_name, save_single_egamma_config_pdf(plt, PdfPages, rows, path)))
    return outputs


def egamma_target_pdf_path(group_name, target):
    """Return the PDF path for one fixed E_gamma and target concurrence."""
    _observable, file_tag = target
    return (
        EGAMMA_CONFIG_DIR
        / file_safe_label(group_name)
        / f"max_{file_tag}_regions.pdf"
    )


def save_egamma_target_region_pdf(target, group_name, rows):
    """Save one PDF for one E_gamma/target with clustered max-C regions."""
    candidates = target_egamma_candidates(rows, target)
    if not candidates:
        return None, []
    clusters = cluster_candidates(candidates)
    detail_rows = egamma_target_region_rows(target, group_name, clusters)
    path = egamma_target_pdf_path(group_name, target)
    plt, PdfPages = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        plot_egamma_target_scan_map(plt, pdf, rows, target, group_name, clusters)
        save_detail_pages(pdf, plt, detail_rows)
    return path, detail_rows


def save_all_egamma_target_region_pdfs(rows):
    """Write separate PDFs for every E_gamma and target concurrence."""
    outputs = []
    detail_rows = []
    for group_name, _qout in qout_groups(rows):
        group_rows = rows_for_qout_group(rows, group_name)
        for target in CONFIG_TARGETS:
            path, details = save_egamma_target_region_pdf(target, group_name, group_rows)
            if path is not None:
                outputs.append((group_name, target[0], path, len(details)))
                detail_rows.extend(details)
    return outputs, detail_rows


def plot_amplitude_decomposition(ax, row):
    """Draw final-state amplitude fractions and phases."""
    records = amplitude_decomposition(row)
    labels = [
        rf"$h_e$={format_helicity(item['hOut'])}, "
        rf"$h_p$={format_helicity(item['sOut'])}, "
        rf"$h_\gamma$={format_helicity(item['lambda'])}"
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
            (
                f"{row['selected_spin_case']} characteristic max "
                f"{observable_latex_label(row['selected_observable'])} configuration"
            ),
            fontsize=16,
        )
        pdf.savefig(fig)
        plt.close(fig)


def save_configuration_plot(target, grouped_clusters, rows, path, energy_rows=None):
    """Save scan maps and detailed momentum/amplitude pages for one target."""
    plt, PdfPages = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    observable, _file_tag = target
    detail_rows = representative_rows(target, grouped_clusters)
    energy_rows = list(energy_rows or [])
    with PdfPages(path) as pdf:
        for spin_case, clusters in grouped_clusters:
            if not clusters:
                continue
            key = clusters[0]["best"]["selected_concurrence_key"]
            plot_scan_map(plt, pdf, rows, key, observable, spin_case, clusters)
        save_egamma_summary_pages(pdf, plt, target, energy_rows)
        save_detail_pages(pdf, plt, detail_rows)
    return path


def save_spin_plots(target, grouped_clusters, rows, energy_rows, base_path):
    """Write one configuration plot PDF per selected spin case."""
    outputs = []
    for spin_case, clusters in grouped_clusters:
        path = spin_output_path(base_path, spin_case)
        spin_energy_rows = [
            row for row in energy_rows
            if row.get("selected_spin_case") == spin_case
        ]
        outputs.append((
            spin_case,
            save_configuration_plot(target, [(spin_case, clusters)], rows, path, spin_energy_rows),
        ))
    return outputs


def build_target_package(target, rows, egamma_detail_rows=()):
    """Build CSV rows and report metadata for one target observable."""
    observable, file_tag = target
    grouped_clusters = []
    for key in selected_target_columns(rows, observable):
        candidates = candidate_rows(rows, key, observable)
        if not candidates:
            continue
        grouped_clusters.append((
            spin_label_from_key(key, observable),
            cluster_candidates(candidates),
        ))
    if not grouped_clusters:
        raise RuntimeError(f"No finite {observable} candidates found.")

    paths = target_paths(file_tag)
    examples = example_rows(grouped_clusters)
    summaries = cluster_summary_rows(target, grouped_clusters)
    details = representative_rows(target, grouped_clusters)
    target_egamma_details = [
        row for row in egamma_detail_rows
        if row.get("selected_observable") == observable
    ]
    output_details = details + target_egamma_details
    momentum_rows = momentum_configuration_rows(output_details)
    amplitude_rows = amplitude_decomposition_rows(output_details)
    spin_cases = []
    for spin_case, _clusters in grouped_clusters:
        if spin_case not in spin_cases:
            spin_cases.append(spin_case)
    for row in target_egamma_details:
        if row["selected_spin_case"] not in spin_cases:
            spin_cases.append(row["selected_spin_case"])

    write_dict_csv(paths["examples"], examples)
    write_dict_csv(paths["clusters"], summaries)
    write_dict_csv(paths["momenta"], momentum_rows)
    write_dict_csv(paths["amplitudes"], amplitude_rows)
    spin_outputs = [
        ("examples", write_spin_csvs(paths["examples"], examples, spin_cases)),
        ("clusters", write_spin_csvs(paths["clusters"], summaries, spin_cases)),
        ("momentum configurations", write_spin_csvs(paths["momenta"], momentum_rows, spin_cases)),
        ("final-state amplitude decomposition", write_spin_csvs(paths["amplitudes"], amplitude_rows, spin_cases)),
    ]
    return {
        "target": target,
        "paths": paths,
        "grouped_clusters": grouped_clusters,
        "examples": examples,
        "egamma_detail_rows": target_egamma_details,
        "spin_outputs": spin_outputs,
    }


def build_report(input_path, total_rows, packages, egamma_outputs):
    """Build the text report for the generated configurations."""
    lines = [
        "Max pairwise-concurrence configuration generator from AlignmentScan results",
        f"  input csv: {input_path}",
        f"  total input rows: {total_rows}",
        f"  data csv folder: {DATA_DIR}",
        f"  per-E_gamma config folder: {EGAMMA_CONFIG_DIR}",
        f"  per-E_gamma/target PDFs: {len(egamma_outputs)}",
        f"  targets: {', '.join(observable_text_label(observable) for observable, _ in CONFIG_TARGETS)}",
        f"  max spin cases per target: {MAX_SPIN_CASES_PER_TARGET}",
        f"  top rows per target/spin case: {TOP_ROWS_PER_TARGET_SPIN}",
        f"  cluster radius: {CLUSTER_RADIUS}",
        f"  max clusters per target/spin case: {MAX_CLUSTERS_PER_TARGET_SPIN}",
        "  no PDF combines different E_gamma values",
        "  saved per-E_gamma/target config PDFs:",
    ]
    for group_name, observable, path, region_count in egamma_outputs:
        label = observable_text_label(observable)
        lines.append(f"    {group_name} {label}: {path} ({region_count} regions)")
    lines.extend([
        "",
    ])
    for package in packages:
        observable, _file_tag = package["target"]
        paths = package["paths"]
        grouped_clusters = package["grouped_clusters"]
        label = observable_text_label(observable)
        lines.extend([
            f"Target {label}:",
            f"  selected spin cases: {', '.join(spin for spin, _clusters in grouped_clusters)}",
            f"  clusters: {sum(len(clusters) for _spin, clusters in grouped_clusters)}",
            f"  per-E_gamma region detail rows: {len(package['egamma_detail_rows'])}",
            f"  examples: {len(package['examples'])}",
            f"  saved examples csv: {paths['examples']}",
            f"  saved clusters csv: {paths['clusters']}",
            f"  saved momentum configuration csv: {paths['momenta']}",
            f"  saved final-state amplitude decomposition csv: {paths['amplitudes']}",
            "  saved per-spin csvs:",
        ])
        for output_label, outputs in package["spin_outputs"]:
            lines.append(f"    {output_label}:")
            for spin_case, path in outputs:
                lines.append(f"      {spin_case}: {path}")
        lines.append("  enhanced regions:")
        for spin_case, clusters in grouped_clusters:
            for cluster in clusters:
                best = cluster["best"]
                rows = cluster["rows"]
                lines.append(
                    "    "
                    f"{spin_case} region {cluster['cluster_id']}: "
                    f"size={len(rows)}, max_{label}={best['selected_concurrence']:.6g}, "
                    f"phi_p_in={format_range(rows, 'phi_in')}, "
                    f"phi_gamma={format_range(rows, 'phiOut')}, "
                    f"final_pair_delta_xy={format_range(rows, 'pair_delta_xy')}, "
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
                    f"pair_delta_xy={best['pair_delta_xy']:.6g}, "
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
    clean_legacy_root_csv_outputs()
    clean_egamma_config_outputs()
    egamma_outputs, egamma_detail_rows = save_all_egamma_target_region_pdfs(rows)
    packages = [
        build_target_package(target, rows, egamma_detail_rows)
        for target in CONFIG_TARGETS
    ]
    log_text = build_report(input_path, len(rows), packages, egamma_outputs)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(log_text, end="")


if __name__ == "__main__":
    main()
