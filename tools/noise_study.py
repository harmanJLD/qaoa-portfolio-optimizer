"""Noise study: how both mixers hold up under gate noise, with and without error detection.

Runs off-platform (needs qiskit-aer) and writes data/noise_study.json, which main.py
plots as dashboard panel F. Run from the project folder:

    pip install qiskit qiskit-aer scipy numpy
    python tools/noise_study.py

Noise model: depolarizing error on every gate, with probability e on each CX and
e / 10 on each single-qubit gate. Error detection (post-selection) keeps only shots
with exactly `budget` assets. With the XY mixer the ideal circuit never leaves that
subspace, so any other shot must contain an error and can be discarded for free.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel, depolarizing_error  # noqa: E402

from portfolio import load_assets, build_qubo, all_costs  # noqa: E402
from qaoa import Simulator, tune_depths, qubo_to_ising, build_circuit  # noqa: E402

BUDGET, RISK, SHOTS = 4, 0.5, 8000
ERRORS = [0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01]
CONFIGS = [("x", 2), ("xy", 1), ("xy", 2), ("xy", 3)]


def main():
    names, mu, sigma = load_assets()
    n = len(names)
    Qp, cp, _ = build_qubo(mu, sigma, BUDGET, RISK)
    Q0, c0, _ = build_qubo(mu, sigma, BUDGET, RISK, penalty=0.0)
    costs_p, bits = all_costs(Qp, cp)
    costs_0, _ = all_costs(Q0, c0)
    feasible = bits.sum(axis=1) == BUDGET
    opt = int(np.argmin(np.where(feasible, costs_0, np.inf)))

    circuits = {}
    for m, p in CONFIGS:
        Q, c, cost = (Q0, c0, costs_0) if m == "xy" else (Qp, cp, costs_p)
        sim = Simulator(cost, n, m, BUDGET)
        tuned, _ = tune_depths(sim, p, extra_restarts=8)
        g, b, _ = tuned[p]
        h, J, _ = qubo_to_ising(Q, c)
        circuits[(m, p)] = build_circuit(h, J, sim.to_physical(g), b, m, BUDGET)

    rows = []
    for e in ERRORS:
        nm = NoiseModel()
        if e:
            nm.add_all_qubit_quantum_error(depolarizing_error(e / 10, 1), ["h", "rx", "ry", "rz"])
            nm.add_all_qubit_quantum_error(depolarizing_error(e, 2), ["cx"])
        backend = AerSimulator(noise_model=nm if e else None, seed_simulator=1)
        for (m, p), qc in circuits.items():
            counts = backend.run(qc, shots=SHOTS).result().get_counts()
            s = {int(k, 2): v for k, v in counts.items()}
            kept = sum(v for i, v in s.items() if feasible[i])
            hit = s.get(opt, 0)
            rows.append({"cx_error": e, "mixer": m, "layers": p, "cx_gates": qc.count_ops().get("cx", 0),
                         "p_opt_raw": hit / SHOTS, "kept_share": kept / SHOTS,
                         "p_opt_postselected": hit / kept if kept else 0.0})
            print(f"e={e:<7} {m:<2} p={p}  raw {hit / SHOTS:.3f}  kept {kept / SHOTS:.2f}  "
                  f"post-selected {rows[-1]['p_opt_postselected']:.3f}")

    out = {"budget": BUDGET, "risk_aversion": RISK, "shots": SHOTS,
           "noise": "depolarizing, e on CX, e/10 on 1-qubit gates", "rows": rows}
    with open("data/noise_study.json", "w") as f:
        json.dump(out, f, indent=1)
    print("Saved data/noise_study.json")


if __name__ == "__main__":
    main()
