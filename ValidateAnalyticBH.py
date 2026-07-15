"""Compare generated contracted amplitudes with the numerical BH implementation."""

import json
from pathlib import Path

import numpy as np

from Algebra import photon_pol
from BHHelicityAmp import bh_amplitude_core
from Kinematics import kinematics_user_from_independent

REFERENCE_PATH = Path("Output") / "AnalyticBH_symbolic_validation.json"
MAX_ABSOLUTE_DIFFERENCE = 1.0e-10


def main():
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    inputs = reference["inputs"]
    kin = kinematics_user_from_independent(
        inputs["s"],
        np.pi / 2.0,
        inputs["alpha"],
        inputs["w"],
        inputs["phi"],
        inputs["m"],
    )
    mom = kin["momenta"]
    maximum = 0.0
    for label, pair in reference["amplitudes"].items():
        h, hp, s, spout, lam = map(int, label.split(","))
        numerical = bh_amplitude_core(
            mom["k"], mom["kp"], mom["qout"], mom["p"], mom["pp"],
            photon_pol(mom["qout"], lam), h, hp, s, spout,
            inputs["m"], inputs["F1"], inputs["F2"],
        )
        symbolic = complex(*pair)
        maximum = max(maximum, abs(numerical - symbolic))
    print(f"maximum absolute difference over 32 amplitudes: {maximum:.16e}")
    if maximum > MAX_ABSOLUTE_DIFFERENCE:
        raise SystemExit("Contracted symbolic amplitudes failed validation.")


if __name__ == "__main__":
    main()
