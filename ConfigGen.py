"""Generate high-entanglement configuration scans from AlignmentScan outputs.

The generator reads the entanglement scan written by ``AlignmentScan.py`` and
builds one configuration package for each requested maximum: ``C_e_p``,
``C_p_gamma``, ``C_e_gamma``, and ``F3``.  Each package contains a scan plot,
representative enhanced regions, reconstructed momenta/kinematics, and a
final-state amplitude decomposition that preserves the AlignmentScan
initial-spin ensemble weights.  ``F3`` is scanned for every configured
polarization; the stored purity is carried into the outputs so pure-state and
mixed-state rows can be distinguished.
"""

import csv
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from AlignmentScan import (
    SPIN_AVERAGING_VERSION,
    observable_latex_label,
    observable_text_label,
)
from FormFactors import yahl_dirac_pauli_from_t
from Kinematics import kinematics_user_from_independent
from SpinDensityMat import (
    M,
    amplitude_table,
    final_state_ensemble,
    process_density_matrix_from_amplitudes,
    contract_initial_state,
    outgoing_spin_states,
)


ALIGNMENT_CONCURRENCE_DIR = Path("Output") / "AlignmentScan" / "ConcurrenceScan"
FULL_CONCURRENCE_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_concurrence_phase_space.csv"
RANKED_CONCURRENCE_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_concurrence_top.csv"
RANKED_E_GAMMA_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_e_gamma_top.csv"
LEGACY_RANKED_C13_CSV = ALIGNMENT_CONCURRENCE_DIR / "electron_photon_c13_top.csv"

OUTPUT_DIR = Path("Output") / "ConfigGen"
DATA_DIR = OUTPUT_DIR / "Data"
EGAMMA_CONFIG_DIR = OUTPUT_DIR / "Config_Plot_By_Egamma"
LOG_PATH = Path("Output") / "ConfigGen.log"

CONFIG_TARGETS = (
    ("C_e_p", "c_ep"),
    ("C_p_gamma", "c_p_gamma"),
    ("C_e_gamma", "c_e_gamma"),
    ("F3", "f3"),
)
CONFIG_SPIN_CASES = (
    "unpolarized",
    "L_proton",
    "L_electron",
    "Tx_proton",
    "Ty_proton",
    "Tx_electron",
    "Ty_electron",
    "LL",
    "LTx",
    "LTy",
    "TxTx",
    "TxTy",
)
MAX_SPIN_CASES_PER_TARGET = None
TOP_ROWS_PER_TARGET_SPIN = 80
MAX_CLUSTERS_PER_TARGET_SPIN = 4
EXAMPLES_PER_CLUSTER = 2
CLUSTER_RADIUS = 0.42
SCAN_HEATMAP_MAX_BINS = 96
AMPLITUDE_MIN_FRACTION = 0.02
AMPLITUDE_MAX_COMPONENTS = 8

DISPLAY_MOMENTA = ("k", "p", "kp", "pp", "qout")
MOMENTUM_DISPLAY_LABELS = {
    "k": r"$\ell$", "p": r"$P$", "kp": r"$\ell'$",
    "pp": r"$P'$", "qout": r"$q_\gamma$",
}
MOMENTUM_COLORS = {
    "k": "tab:blue", "p": "tab:orange", "kp": "tab:cyan",
    "pp": "tab:red", "qout": "tab:green",
}
MOMENTUM_KIND = {
    "k": "electron", "kp": "electron",
    "p": "proton", "pp": "proton",
    "qout": "photon",
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


from PlotUtils import require_matplotlib as _require_matplotlib


def parse_float(value, default=np.nan):
    """Parse a CSV numeric field, preserving missing values as NaN."""
    if value is None or value == "":
        return default
    return float(value)


def _row_float(row, key, default=np.nan):
    """Return ``parse_float(row.get(key), default)`` in one call."""
    return parse_float(row.get(key), default)


def read_csv_rows(path):
    """Read a CSV file as dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_spin_averaging_version(rows, input_path):
    """Reject stale AlignmentScan CSVs with incompatible spin conventions."""
    versions = {
        str(row.get("initial_spin_averaging_version", "")).strip()
        for row in rows
    }
    if versions == {SPIN_AVERAGING_VERSION}:
        return
    found = ", ".join(sorted(version or "<missing>" for version in versions))
    raise ValueError(
        f"AlignmentScan CSV {input_path} uses initial-spin averaging version(s) "
        f"{found}; expected {SPIN_AVERAGING_VERSION}. Re-run AlignmentScan.py "
        "before running ConfigGen.py."
    )


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
    """Remove stale per-E_gamma PDFs."""
    if not EGAMMA_CONFIG_DIR.exists():
        return
    for path in EGAMMA_CONFIG_DIR.rglob("*.pdf"):
        path.unlink()
    for path in sorted(EGAMMA_CONFIG_DIR.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.rmdir()


def clean_data_outputs():
    """Remove generated ConfigGen CSVs before writing the organized layout."""
    if not DATA_DIR.exists():
        return
    for path in DATA_DIR.rglob("*.csv"):
        path.unlink()
    for path in sorted(
        DATA_DIR.rglob("*"), key=lambda item: len(item.parts), reverse=True
    ):
        if path.is_dir():
            path.rmdir()


def alignment_input_path():
    """Return the best available entanglement CSV for configuration scans."""
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
    """Return output paths for one requested entanglement target."""
    prefix = f"max_{file_tag}"
    combined_dir = DATA_DIR / file_tag / "combined"
    return {
        "examples": combined_dir / f"{prefix}_configuration_examples.csv",
        "clusters": combined_dir / f"{prefix}_cluster_summary.csv",
        "momenta": combined_dir / f"{prefix}_momentum_configurations.csv",
        "amplitudes": combined_dir / f"{prefix}_final_state_amplitude_decomposition.csv",
    }


def spin_label_from_key(key, observable):
    """Return the spin-case prefix from an observable column name."""
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
    """Return transverse separation for a pair target, or NaN for ``F3``."""
    pair = TARGET_FINAL_MOMENTA.get(observable)
    if pair is None:
        return np.nan
    first, second = pair
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
    spin_case = spin_label_from_key(key, observable)
    item["selected_observable"] = observable
    item["selected_observable_label"] = observable_text_label(observable)
    item["selected_spin_case"] = spin_case
    item["selected_concurrence_key"] = key
    item["selected_concurrence"] = value
    item["selected_purity"] = parse_float(row.get(f"{spin_case}_purity"))
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
    """Return ranked candidates for one target/spin observable column."""
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
    """Greedily cluster high-observable candidate rows by scan coordinates."""
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
                "best_purity": f"{best['selected_purity']:.16e}",
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
                    "selected_purity": f"{row['selected_purity']:.16e}",
                    "pair_delta_xy": f"{row['pair_delta_xy']:.16e}",
                }
                for name in KINEMATIC_COLUMNS:
                    item[name] = row.get(name, "")
                for key, value in row.items():
                    is_selected_f3 = (
                        row["selected_observable"] == "F3" and key.endswith("_F3")
                    )
                    if "_C_" in key or key.endswith("_C13") or is_selected_f3:
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


def target_egamma_candidates(rows, target, key):
    """Return high-observable candidates for one target/spin at one E_gamma."""
    observable, _file_tag = target
    candidates = []
    seen = set()
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


def egamma_target_region_rows(target, group_name, spin_case, clusters):
    """Return representative detail rows for one E_gamma/target/spin region set."""
    observable, file_tag = target
    detail_rows = []
    for cluster in clusters:
        row = dict(cluster["best"])
        row["selected_observable"] = observable
        row["selected_region"] = f"{group_name}_{spin_case}_region_{cluster['cluster_id']}"
        row["cluster_id"] = cluster["cluster_id"]
        row["detail_id"] = (
            f"{file_tag}_{file_safe_label(group_name)}_"
            f"{file_safe_label(spin_case)}_region_{cluster['cluster_id']}"
        )
        row["detail_source"] = "egamma_region_max"
        detail_rows.append(row)
    return detail_rows


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


def spin_output_path(base_path, spin_case):
    """Return a target/polarization-organized output path."""
    target_dir = base_path.parent.parent
    return target_dir / file_safe_label(spin_case) / base_path.name


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
        meta = _row_meta(row)
        meta["initial_spin_averaging_version"] = row.get(
            "initial_spin_averaging_version", SPIN_AVERAGING_VERSION
        )
        for name in DISPLAY_MOMENTA:
            vector = kin["momenta"][name]
            records.append({
                **meta,
                "momentum": name,
                "E": f"{vector[0]:.16e}",
                "px": f"{vector[1]:.16e}", "py": f"{vector[2]:.16e}", "pz": f"{vector[3]:.16e}",
                "p_abs": f"{np.linalg.norm(vector[1:4]):.16e}",
                "phi_xy": f"{vector_phi_xy(vector):.16e}",
                "s": f"{kin['s']:.16e}", "sqrt_s": f"{kin['sqrt_s']:.16e}",
                "pIn": f"{kin['pIn']:.16e}", "pOut": f"{kin['pOut']:.16e}",
                "theta_in": f"{kin['theta_in']:.16e}", "phi_in": f"{kin['phi_in']:.16e}",
                "qOut": f"{kin['qOut']:.16e}", "phiOut": f"{kin['phiOut']:.16e}",
                "Q2": f"{kin['Q2']:.16e}", "xB": f"{kin['xB']:.16e}",
                "t": f"{kin['t']:.16e}", "W2": f"{kin['W2']:.16e}", "y": f"{kin['y']:.16e}",
            })
    return records


def selected_initial_state_label(spin_case):
    """Return a compact label for the incoming spin ensemble behind a plot."""
    labels = {
        "unpolarized": r"$\frac{1}{4}\sum_{h_e,h_p}\rho(h_e,h_p)$",
        "L_proton": r"$\frac{1}{2}\sum_{h_e}\rho(h_e,L_p)$",
        "L_electron": r"$\frac{1}{2}\sum_{h_p}\rho(L_e,h_p)$",
        "Tx_proton": r"$\frac{1}{2}\sum_{h_e}\rho(h_e,T_{x,p})$",
        "Ty_proton": r"$\frac{1}{2}\sum_{h_e}\rho(h_e,T_{y,p})$",
        "Tx_electron": r"$\frac{1}{2}\sum_{h_p}\rho(T_{x,e},h_p)$",
        "Ty_electron": r"$\frac{1}{2}\sum_{h_p}\rho(T_{y,e},h_p)$",
        "LL": r"$|L_eL_p\rangle$",
        "LTx": r"$|L_eT_{x,p}\rangle$",
        "LTy": r"$|L_eT_{y,p}\rangle$",
        "TxTx": r"$|T_{x,e}T_{x,p}\rangle$",
        "TxTy": r"$|T_{x,e}T_{y,p}\rangle$",
    }
    return labels.get(spin_case, spin_case)


def _row_meta(row):
    """Return the shared metadata prefix dict for a detail row."""
    return {
        "detail_id": row["detail_id"],
        "detail_source": row.get("detail_source", ""),
        "selected_observable": row["selected_observable"],
        "selected_observable_label": row["selected_observable_label"],
        "selected_spin_case": row["selected_spin_case"],
        "selected_region": row["selected_region"],
        "cluster_id": row["cluster_id"],
        "selected_concurrence": f"{row['selected_concurrence']:.16e}",
        "selected_purity": f"{row['selected_purity']:.16e}",
        "pair_delta_xy": f"{row['pair_delta_xy']:.16e}",
        "kinematic_point": row.get("kinematic_point", ""),
    }


def amplitude_decomposition(row):
    """Return ensemble-aware final-state amplitude decomposition records."""
    kin = kinematics_from_config_row(row)
    F1, F2 = yahl_dirac_pauli_from_t(kin["t"], M)
    amplitudes = amplitude_table(kin["momenta"], M, F1, F2)
    process_rho = process_density_matrix_from_amplitudes(amplitudes)
    contracted_rho = contract_initial_state(process_rho, row["selected_spin_case"])
    ensemble = final_state_ensemble(amplitudes, row["selected_spin_case"])
    total = float(np.real_if_close(np.trace(contracted_rho), tol=1000).real)
    if total <= 1.0e-14:
        raise ZeroDivisionError(f"Zero selected amplitude norm for {row['detail_id']}.")
    meta = _row_meta(row)
    meta.update({
        "incoming_state": selected_initial_state_label(row["selected_spin_case"]),
        "initial_spin_averaging_version": row.get(
            "initial_spin_averaging_version", SPIN_AVERAGING_VERSION
        ),
    })
    records = []
    for component in ensemble:
        norms = np.abs(component["state"]) ** 2
        for index, ((h_out, s_out, lam), amplitude, norm) in enumerate(
            zip(outgoing_spin_states(), component["state"], norms)
        ):
            weighted_norm = float(component["weight"] * norm)
            records.append({
                **meta,
                "initial_component": component["label"],
                "ensemble_weight": f"{component['weight']:.16e}",
                "out_index": index,
                "hOut": h_out, "sOut": s_out, "lambda": lam,
                "amplitude_real": f"{amplitude.real:.16e}",
                "amplitude_imag": f"{amplitude.imag:.16e}",
                "amplitude_abs": f"{abs(amplitude):.16e}",
                "amplitude_phase": f"{np.angle(amplitude):.16e}",
                "amplitude_abs2": f"{norm:.16e}",
                "weighted_abs2": f"{weighted_norm:.16e}",
                "fraction": f"{weighted_norm / total:.16e}",
            })
    records.sort(key=lambda item: _row_float(item, "fraction"), reverse=True)
    selected = [r for r in records if _row_float(r, "fraction") >= AMPLITUDE_MIN_FRACTION]
    selected = selected[:AMPLITUDE_MAX_COMPONENTS]
    retained = sum(_row_float(r, "fraction") for r in selected)
    for rank, item in enumerate(selected, start=1):
        item["decomposition_rank"] = rank
        item["retained_fraction_total"] = f"{retained:.16e}"
    return selected


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
    ax.axvline(0.5 * math.pi, color="white", linestyle="--", linewidth=0.45, alpha=0.45)
    ax.axhline(0.5 * math.pi, color="white", linestyle="--", linewidth=0.45, alpha=0.45)


def plot_egamma_target_scan_map(plt, pdf, rows, target, group_name, spin_case, key, clusters):
    """Append one fixed-E_gamma/target/spin scan map with region markers."""
    observable, _file_tag = target
    x_values = []
    y_values = []
    z_values = []
    for row in rows:
        value = parse_float(row.get("rank_value") if row.get("rank_group") == key else row.get(key))
        if not np.isfinite(value):
            continue
        x_values.append(scan_x_phi(row))
        y_values.append(parse_float(row.get("phiOut")))
        z_values.append(value)
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
        f"{group_name} {spin_case}: max {observable_latex_label(observable)} regions",
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


def format_vector_line(name, vector):
    """Return one compact four-vector line."""
    label = MOMENTUM_DISPLAY_LABELS.get(name, name)
    return (
        f"{label:10s} $E$={vector[0]:8.4f} "
        f"$p_x$={vector[1]:8.4f} $p_y$={vector[2]:8.4f} $p_z$={vector[3]:8.4f}"
    )


def perpendicular_2d(delta):
    """Return a unit vector perpendicular to a transverse momentum."""
    delta = np.asarray(delta, dtype=float)
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
    kind = MOMENTUM_KIND.get(label, "other")
    if kind == "photon":
        _plot_wavy_2d(ax, start, end, color, amplitude=0.025 * line_scale)
    else:
        _plot_arrow_2d(ax, start, end, color,
                       linestyle="--" if kind == "electron" else "-",
                       linewidth=1.75 if kind == "proton" else 1.65)
    ax.text(end[0], end[1], f" {MOMENTUM_DISPLAY_LABELS.get(label, label)}",
            color=color, fontsize=11, va="center")


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
    kind = MOMENTUM_KIND.get(label, "other")
    _plot_line_arrow_3d(ax, start, end, color,
                        linestyle={ "proton": "-", "photon": ":", "electron": "--" }.get(kind, "-"),
                        linewidth={ "proton": 1.75, "photon": 1.6, "electron": 1.55 }.get(kind, 1.55))
    ax.text(end[0], end[1], end[2], f" {MOMENTUM_DISPLAY_LABELS.get(label, label)}",
            color=color, fontsize=9)


def plot_transverse_momenta(ax, kin, title="Transverse plane"):
    """Draw styled transverse momentum vectors."""
    momenta = kin["momenta"]
    transverse = np.asarray([momenta[name][1:3] for name in DISPLAY_MOMENTA])
    line_scale = max(1.0, float(np.nanmax(np.abs(transverse))) * 1.20)
    for name in DISPLAY_MOMENTA:
        _plot_vector_2d(ax, momenta[name], name, MOMENTUM_COLORS[name], line_scale)
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
    momenta = kin["momenta"]
    vectors = np.asarray([momenta[name][1:4] for name in DISPLAY_MOMENTA])
    line_scale = max(1.0, float(np.nanmax(np.abs(vectors))) * 1.15)
    ax3d = fig.add_subplot(grid_spec[0, 0], projection="3d")
    ax2d = fig.add_subplot(grid_spec[1, 0])
    for name in DISPLAY_MOMENTA:
        _plot_vector_3d(ax3d, momenta[name], name, MOMENTUM_COLORS[name], line_scale)
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
    label = MOMENTUM_DISPLAY_LABELS.get(name, name)
    return (
        f"{label:10s} $E$={vector[0]:8.4f} "
        f"$p_x$={vector[1]:8.4f} $p_y$={vector[2]:8.4f} $p_z$={vector[3]:8.4f}"
    )


def plot_configuration_text(ax, row, kin):
    """Draw kinematic and momentum text for a detailed configuration page."""
    ax.axis("off")
    momenta = kin["momenta"]
    label = observable_text_label(row["selected_observable"])
    region_line = f"region: {row['selected_region']}"
    if np.isfinite(row["pair_delta_xy"]):
        region_line += f"  final-pair delta_xy={row['pair_delta_xy']:.6g} rad"
    lines = [
        f"{row['detail_id']}  {label}={row['selected_concurrence']:.6g}",
        region_line,
        f"outgoing-state purity: {row['selected_purity']:.6g}",
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


def egamma_target_pdf_path(group_name, target, spin_case):
    """Return the PDF path under a polarization subdirectory."""
    _observable, file_tag = target
    return (
        EGAMMA_CONFIG_DIR
        / file_safe_label(spin_case)
        / f"{file_safe_label(group_name)}_{file_tag}_regions.pdf"
    )


def save_egamma_target_region_pdf(target, group_name, rows, key):
    """Save one PDF for one E_gamma/target/spin with clustered maxima."""
    observable, _file_tag = target
    spin_case = spin_label_from_key(key, observable)
    candidates = target_egamma_candidates(rows, target, key)
    if not candidates:
        return None, []
    clusters = cluster_candidates(candidates)
    detail_rows = egamma_target_region_rows(target, group_name, spin_case, clusters)
    path = egamma_target_pdf_path(group_name, target, spin_case)
    plt, PdfPages = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        plot_egamma_target_scan_map(
            plt, pdf, rows, target, group_name, spin_case, key, clusters,
        )
        save_detail_pages(pdf, plt, detail_rows)
    return path, detail_rows


def save_all_egamma_target_region_pdfs(rows):
    """Write PDFs per E_gamma × target × polarization under polarization folders."""
    outputs = []
    detail_rows = []
    for group_name, _qout in qout_groups(rows):
        group_rows = rows_for_qout_group(rows, group_name)
        for target in CONFIG_TARGETS:
            observable, _file_tag = target
            for key in target_columns(group_rows, observable):
                spin_case = spin_label_from_key(key, observable)
                path, details = save_egamma_target_region_pdf(
                    target, group_name, group_rows, key,
                )
                if path is not None and details:
                    outputs.append((group_name, observable, spin_case, path, len(details)))
                    detail_rows.extend(details)
    return outputs, detail_rows


def plot_amplitude_decomposition(ax, row):
    """Draw ensemble-weighted final-state amplitude fractions and phases."""
    records = amplitude_decomposition(row)
    labels = [
        rf"$h_e$={format_helicity(item['hOut'])}, "
        rf"$h_p$={format_helicity(item['sOut'])}, "
        rf"$h_\gamma$={format_helicity(item['lambda'])}" + "\n" + item["initial_component"]
        for item in records
    ]
    fractions = np.asarray([parse_float(item["fraction"]) for item in records], dtype=float)
    phases = np.asarray([parse_float(item["amplitude_phase"]) for item in records], dtype=float)
    y_pos = np.arange(len(records))
    bars = ax.barh(y_pos, fractions, color="tab:blue", alpha=0.72)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("ensemble-weighted final-state |A|^2 fraction")
    retained = parse_float(records[0]["retained_fraction_total"])
    ax.set_title(
        "Leading initial-ensemble/final-state components "
        f"(N={len(records)}, retained={retained:.1%})"
    )
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
    amplitude_rows = [r for row in output_details for r in amplitude_decomposition(row)]
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
        "Max entanglement-observable configuration generator from AlignmentScan results",
        f"  input csv: {input_path}",
        f"  required initial-spin averaging version: {SPIN_AVERAGING_VERSION}",
        "  unpolarized decomposition: 1/4 average over electron and proton helicities",
        "  single-particle polarization: direct prepared state with an incoherent 1/2 average over the other particle",
        "  double polarization: direct prepared electron-proton product state",
        "  no polarized category is constructed as a helicity asymmetry",
        f"  total input rows: {total_rows}",
        f"  data csv folder: {DATA_DIR}",
        f"  per-E_gamma config folder: {EGAMMA_CONFIG_DIR}",
        f"  config PDFs (one per E_gamma × target, all spin cases collected): {len(egamma_outputs)} spin entries",
        f"  targets: {', '.join(observable_text_label(observable) for observable, _ in CONFIG_TARGETS)}",
        f"  max spin cases per target: {MAX_SPIN_CASES_PER_TARGET}",
        f"  top rows per target/spin case: {TOP_ROWS_PER_TARGET_SPIN}",
        f"  cluster radius: {CLUSTER_RADIUS}",
        f"  max clusters per target/spin case: {MAX_CLUSTERS_PER_TARGET_SPIN}",
        f"  amplitude component minimum fraction: {AMPLITUDE_MIN_FRACTION:.0%}",
        f"  amplitude component maximum count: {AMPLITUDE_MAX_COMPONENTS}",
        "  data layout: Data/<target>/<polarization>/... with combined tables under combined/",
        "  PDF layout: Config_Plot_By_Egamma/<polarization>/<E_gamma>_<target>_regions.pdf",
        "  saved per-polarization config PDFs:",
    ]
    for group_name, observable, spin_case, path, region_count in egamma_outputs:
        label = observable_text_label(observable)
        lines.append(f"    {group_name} {spin_case} {label}: {path} ({region_count} regions)")
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
                pair_range = (
                    f"final_pair_delta_xy={format_range(rows, 'pair_delta_xy')}, "
                    if observable in TARGET_FINAL_MOMENTA else ""
                )
                best_pair = (
                    f"pair_delta_xy={best['pair_delta_xy']:.6g}, "
                    if observable in TARGET_FINAL_MOMENTA else ""
                )
                lines.append(
                    "    "
                    f"{spin_case} region {cluster['cluster_id']}: "
                    f"size={len(rows)}, max_{label}={best['selected_concurrence']:.6g}, "
                    f"phi_p_in={format_range(rows, 'phi_in')}, "
                    f"phi_gamma={format_range(rows, 'phiOut')}, "
                    f"purity={format_range(rows, 'selected_purity')}, "
                    f"{pair_range}"
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
                    f"purity={best['selected_purity']:.6g}, "
                    f"{best_pair}"
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
    validate_spin_averaging_version(rows, input_path)
    clean_legacy_root_csv_outputs()
    clean_egamma_config_outputs()
    clean_data_outputs()
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
