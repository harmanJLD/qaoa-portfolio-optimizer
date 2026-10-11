"""QAOA Portfolio Optimizer: budget-preserving XY mixer vs standard QAOA

1. Classical baselines: exact brute force and a greedy heuristic.
2. Local study (no quantum jobs): tune both mixers at depths p = 1..4 on an exact
   statevector, map the p = 1 energy landscape, and compare circuit costs.
3. Efficient frontier: re-tune QAOA for several risk levels.
4. ONE quantum job: the main circuit plus every frontier circuit go to the selected
   backend in a single backend.run call, which keeps the project inside IonQ's
   one-job-per-run limit.
5. A 6-panel dashboard (qaoa_portfolio_dashboard.png) and a JSON summary.
"""
import json
import time
import warnings
import numpy as np

# Quiet a harmless Matplotlib deprecation notice raised by the browser image viewer
warnings.filterwarnings("ignore", message=r"The (width|height) parameter as float")
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation

from portfolio import load_assets, build_qubo, all_costs, brute_force, greedy, portfolio_stats
from qaoa import (Simulator, tune_depths, qubo_to_ising, build_circuit,
                  dicke_circuit, dicke_target, BASIS)

ORANGE, BLUE, GREY, INK = "#e8692c", "#3b6fb6", "#b9b9b9", "#222222"
MAX_STUDY_DEPTH = 4
NOISE_FILE = "data/noise_study.json"


def fmt(x, names):
    return ", ".join(nm for nm, b in zip(names, x) if b) or "(none)"


def index_of(x):
    return int(sum(int(b) << i for i, b in enumerate(x)))


def to_samples(counts):
    """Qiskit puts qubit 0 on the right of each bitstring, matching our bit order."""
    out = {}
    for key, v in counts.items():
        i = int(key.replace(" ", ""), 2)
        out[i] = out.get(i, 0) + v
    return out


def approx_ratio(samples, costs, bits, budget, c_opt):
    feas = {i: v for i, v in samples.items() if bits[i].sum() == budget}
    if not feas:
        return float("nan")
    worst = max(costs[i] for i in range(len(costs)) if bits[i].sum() == budget)
    mean = sum(costs[i] * v for i, v in feas.items()) / sum(feas.values())
    return (worst - mean) / (worst - c_opt) if worst != c_opt else 1.0


def frontier_levels(mu, sigma, budget, n_points):
    """Risk-aversion values spread along the frontier. Each sits at the geometric middle
    of the q-range where its classical optimum holds, away from switch-over points where
    two portfolios tie and no optimizer can separate them."""
    grid = np.geomspace(0.05, 20, 80)
    ranges = {}
    for q in grid:
        Q, c, _ = build_qubo(mu, sigma, budget, q, penalty=0.0)
        x, _ = brute_force(Q, c, budget)
        lo, hi = ranges.get(index_of(x), (q, q))
        ranges[index_of(x)] = (min(lo, q), max(hi, q))
    # Skip optima that hold over a very narrow band, then spread the picks evenly
    wide = sorted((np.sqrt(lo * hi), hi / lo) for lo, hi in ranges.values())
    wide = [q for q, width in wide if width > 1.15] or [q for q, _ in wide]
    if len(wide) <= n_points:
        return wide
    pick = np.linspace(0, len(wide) - 1, n_points).round().astype(int)
    return [wide[i] for i in pick]


def relative_gap(costs, feasible):
    v = np.sort(costs[feasible])
    return float((v[1] - v[0]) / (v[-1] - v[0]))


def main(shots: int = 2000, layers: int = 2, budget: int = 4, risk_aversion: float = 0.5, mixer: str = "xy", frontier_points: int = 4, analysis: bool = True):
    t0 = time.time()
    names, mu, sigma = load_assets()
    n = len(names)
    mixer = mixer.strip().lower()
    if mixer not in ("xy", "x"):
        print('mixer must be "xy" or "x".')
        return
    if not 1 <= budget < n:
        print(f"budget must be between 1 and {n - 1}.")
        return
    layers = max(1, min(layers, 6))
    frontier_points = max(0, min(frontier_points, 6))

    print("=" * 72)
    print(f" QAOA portfolio optimizer | pick {budget} of {n} assets | q = {risk_aversion} "
          f"| p = {layers} | mixer = {mixer}")
    print("=" * 72)

    # ---------------------------------------------------------- problem + baselines
    Qp, cp, penalty = build_qubo(mu, sigma, budget, risk_aversion)      # with penalty ("x")
    Q0, c0, _ = build_qubo(mu, sigma, budget, risk_aversion, penalty=0)  # objective only ("xy")
    costs_p, bits = all_costs(Qp, cp)
    costs_0, _ = all_costs(Q0, c0)
    feasible = bits.sum(axis=1) == budget
    n_feasible = int(feasible.sum())

    x_opt, c_opt = brute_force(Q0, c0, budget)
    x_gr, c_gr = greedy(mu, sigma, budget, risk_aversion, Q0, c0)
    opt = index_of(x_opt)
    print(f"\nSearch space: {2 ** n} bitstrings, {n_feasible} portfolios meet the budget")
    print("\nClassical baselines")
    for label, x, cv in (("Exact (brute force)", x_opt, c_opt), ("Greedy heuristic", x_gr, c_gr)):
        r, s = portfolio_stats(x, mu, sigma)
        print(f"  {label:<20} {fmt(x, names):<26} cost {cv:+.4f}  return {r:.3f}  risk {s:.3f}")
    print(f"  Greedy matches the optimum: {'yes' if index_of(x_gr) == opt else 'NO'}")

    # ---------------------------------------------------------- local study
    sims = {"x": Simulator(costs_p, n, "x", budget), "xy": Simulator(costs_0, n, "xy", budget)}
    study_depth = max(layers, MAX_STUDY_DEPTH) if analysis else layers
    tuned, landscapes = {}, {}
    for m in (("x", "xy") if analysis else (mixer,)):
        tuned[m], landscapes[m] = tune_depths(sims[m], study_depth, extra_restarts=8, target=opt)

    depth_table = {}
    if analysis:
        print("\nLocal ideal study: probability of the optimal portfolio")
        print(f"  {'p':>3}  {'standard (x)':>13}  {'XY mixer':>9}  {'XY gain':>8}")
        for p in range(1, study_depth + 1):
            row = {}
            for m in ("x", "xy"):
                g, b, _ = tuned[m][p]
                pr = sims[m].probs(g, b)
                row[m] = (float(pr[opt]), float(pr[feasible].sum()))
            depth_table[p] = row
            print(f"  {p:>3}  {row['x'][0]:>13.3f}  {row['xy'][0]:>9.3f}  {row['xy'][0] / row['x'][0]:>7.1f}x")
        print(f"  random guess: {1 / 2 ** n:.4f}   uniform over budget-valid portfolios: {1 / n_feasible:.4f}")

    sim = sims[mixer]
    g_main, b_main, _ = tuned[mixer][layers]
    p_ideal = sim.probs(g_main, b_main)

    # ---------------------------------------------------------- circuits
    run_Q, run_c = (Q0, c0) if mixer == "xy" else (Qp, cp)
    h, J, _ = qubo_to_ising(run_Q, run_c)
    circuits = [build_circuit(h, J, sim.to_physical(g_main), b_main, mixer, budget, name="main")]

    gate_report = {}
    if analysis:
        def cx(qc):
            return transpile(qc, basis_gates=BASIS, optimization_level=1).count_ops().get("cx", 0)
        generic = QuantumCircuit(n)
        generic.append(StatePreparation(dicke_target(n, budget)), range(n))
        gate_report = {
            "Dicke prep\n(generic)": cx(generic),
            "Dicke prep\n(this project)": cx(dicke_circuit(n, budget)),
            f"Full circuit\n{mixer}, p={layers}": circuits[0].count_ops().get("cx", 0),
        }
        print("\nCircuit cost (CX gates)")
        for k, v in gate_report.items():
            print(f"  {k.replace(chr(10), ' '):<28} {v}")

    frontier = []
    if frontier_points:
        for q in frontier_levels(mu, sigma, budget, frontier_points):
            Qf, cf, _ = build_qubo(mu, sigma, budget, q, penalty=0.0 if mixer == "xy" else None)
            cost_f, _ = all_costs(Qf, cf)
            sim_f = Simulator(cost_f, n, mixer, budget)
            tuned_f, _ = tune_depths(sim_f, layers, n_grid=24, extra_restarts=3)
            gf, bf, _ = tuned_f[layers]
            hf, Jf, _ = qubo_to_ising(Qf, cf)
            circuits.append(build_circuit(hf, Jf, sim_f.to_physical(gf), bf, mixer, budget, name=f"q={q:.2f}"))
            Q0f, c0f, _ = build_qubo(mu, sigma, budget, q, penalty=0.0)
            xf, _ = brute_force(Q0f, c0f, budget)
            probs_f = sim_f.probs(gf, bf)
            frontier.append({"q": q, "opt": index_of(xf), "gap": relative_gap(cost_f, feasible),
                             "costs": cost_f, "probs": probs_f, "p_ideal": float(probs_f[index_of(xf)])})

    # IonQ backends accept one circuit per job. There, only the main circuit goes to the
    # device, and frontier points are sampled locally from their ideal statevectors.
    backend_id = (type(backend).__module__ + " " + str(getattr(backend, "name", ""))).lower()
    single_circuit = "ionq" in backend_id
    to_send = circuits[:1] if single_circuit else circuits
    frontier_source = "same job" if frontier and not single_circuit else "ideal local samples"

    c0_main = circuits[0]
    print(f"\nMain circuit: {n} qubits, depth {c0_main.depth()}, {c0_main.count_ops().get('cx', 0)} CX")
    if single_circuit and frontier:
        print("This backend accepts one circuit per job: sending the main circuit only. "
              "Frontier points use ideal local samples.")
    print(f"Local work done in {time.time() - t0:.1f}s. Sending ONE job with {len(to_send)} "
          f"circuit(s) x {shots} shots to the selected backend...")

    # ---------------------------------------------------------- the single quantum job
    result = backend.run(to_send if len(to_send) > 1 else to_send[0], shots=shots).result()
    all_samples = [to_samples(result.get_counts(i)) for i in range(len(to_send))]
    if single_circuit and frontier:
        rng = np.random.default_rng(11)
        for f in frontier:
            drawn = rng.multinomial(shots, f["probs"] / f["probs"].sum())
            all_samples.append({i: int(v) for i, v in enumerate(drawn) if v})

    # ---------------------------------------------------------- main results
    samples = all_samples[0]
    total = sum(samples.values())
    top = max(samples, key=samples.get)
    best = min(samples, key=lambda i: costs_0[i] if feasible[i] else np.inf)
    ratio = approx_ratio(samples, costs_0, bits, budget, c_opt)
    p_meas = samples.get(opt, 0) / total
    rank = 1 + sorted(samples, key=samples.get, reverse=True).index(opt) if opt in samples else None

    print("\nQuantum results (main circuit)")
    print(f"  Most frequent portfolio:   {fmt(bits[top], names)}  ({samples[top] / total:.1%} of shots)")
    print(f"  Exact optimum found:       {'YES' if best == opt else 'no'}  (rank {rank} by frequency)")
    print(f"  P(optimal): measured {p_meas:.3f} | ideal {p_ideal[opt]:.3f} | random {1 / 2 ** n:.4f}")
    print(f"  Shots meeting the budget:  {sum(v for i, v in samples.items() if feasible[i]) / total:.1%}")
    print(f"  Approximation ratio:       {ratio:.3f}  (1.0 = every valid shot is optimal, 0.5 = random)")

    # Error detection: the budget is a symmetry of the XY circuit, so on a noisy backend any
    # shot with the wrong number of assets must contain an error and can be discarded.
    kept = sum(v for i, v in samples.items() if feasible[i])
    p_post = samples.get(opt, 0) / kept if kept else 0.0
    print("\nError detection (post-selection on the budget)")
    if mixer == "xy":
        print(f"  Shots flagged as errors:   {1 - kept / total:.1%}  (0% on an ideal simulator)")
    else:
        print(f"  Shots discarded:           {1 - kept / total:.1%}  (standard QAOA leaves the budget by design)")
    print(f"  P(optimal) raw {p_meas:.3f}  ->  post-selected {p_post:.3f}")

    noise = None
    try:
        with open(NOISE_FILE) as fh:
            noise = json.load(fh)
    except (OSError, ValueError):
        pass
    if noise:
        print("\nNoise study (precomputed with Qiskit Aer, see tools/noise_study.py)")
        print(f"  {'CX error':>8}  {'standard p=2 + detection':>25}  {'XY p=2 raw':>11}  {'XY p=2 + detection':>19}")
        for e in sorted({r["cx_error"] for r in noise["rows"]}):
            get = {(r["mixer"], r["layers"]): r for r in noise["rows"] if r["cx_error"] == e}
            if ("x", 2) in get and ("xy", 2) in get:
                print(f"  {e:>8.4f}  {get[('x', 2)]['p_opt_postselected']:>25.3f}  {get[('xy', 2)]['p_opt_raw']:>11.3f}"
                      f"  {get[('xy', 2)]['p_opt_postselected']:>19.3f}")

    if frontier:
        print(f"\nEfficient frontier ({frontier_source})")
        print("  QAOA is used as a sampler: the pick is the lowest-cost portfolio among the shots.")
        print(f"  {'q':>6}  {'classical optimum':<26} {'QAOA pick':<26} match  P(opt)  vs random  gap")
        for f, smp in zip(frontier, all_samples[1:]):
            tot = sum(smp.values())
            f["top"] = min((i for i in smp if feasible[i]), key=lambda i: f["costs"][i])
            f["p_meas"] = smp.get(f["opt"], 0) / tot
            print(f"  {f['q']:>6.2f}  {fmt(bits[f['opt']], names):<26} {fmt(bits[f['top']], names):<26} "
                  f"{'yes' if f['top'] == f['opt'] else 'no ':<5}  {f['p_meas']:.3f}  {f['p_meas'] * n_feasible:>8.1f}x"
                  f"  {f['gap']:.3f}")
        print("  gap = cost difference between the best and second-best portfolio, as a share of the"
              " full range. Small gaps are harder for QAOA.")

    # ---------------------------------------------------------- dashboard
    rows = 3 if noise else 2
    fig = plt.figure(figsize=(16, 4.75 * rows))
    gs = fig.add_gridspec(rows, 3, hspace=0.45, wspace=0.38)

    ax = fig.add_subplot(gs[0, :2])
    order = sorted(samples, key=samples.get, reverse=True)[:12]
    colors = [ORANGE if i == opt else (BLUE if feasible[i] else GREY) for i in order]
    ax.bar(range(len(order)), [samples[i] / total for i in order], color=colors)
    ax.plot(range(len(order)), [p_ideal[i] for i in order], "_", color=INK, markersize=24, mew=2,
            label="ideal prediction")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["\n".join(nm for nm, b in zip(names, bits[i]) if b) for i in order], fontsize=7.5)
    ax.set_ylabel("share of shots")
    ax.set_title(f"A. Measured portfolios, mixer '{mixer}', p = {layers}   "
                 f"(orange = optimal, blue = meets budget, grey = breaks it)", loc="left", fontsize=10)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    feas_idx = np.where(feasible)[0]
    risk = np.array([portfolio_stats(bits[i], mu, sigma)[1] for i in feas_idx])
    ret = np.array([portfolio_stats(bits[i], mu, sigma)[0] for i in feas_idx])
    ax.scatter(risk, ret, s=12, color=GREY, label="all valid portfolios")
    if frontier:
        fr = sorted(frontier, key=lambda f: portfolio_stats(bits[f["opt"]], mu, sigma)[1])
        pts = np.array([portfolio_stats(bits[f["opt"]], mu, sigma)[::-1] for f in fr])
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=INK, ms=5, lw=1.2, label="classical frontier")
        qp = np.array([portfolio_stats(bits[f["top"]], mu, sigma)[::-1] for f in fr])
        ax.scatter(qp[:, 0], qp[:, 1], s=110, facecolors="none", edgecolors=ORANGE, lw=2,
                   label="QAOA picks")
    r_o, s_o = portfolio_stats(x_opt, mu, sigma)
    ax.scatter([s_o], [r_o], marker="*", s=180, color=ORANGE, zorder=5, label=f"optimum at q = {risk_aversion}")
    ax.set_xlabel("risk (volatility)")
    ax.set_ylabel("expected return")
    ax.set_title(f"B. Efficient frontier ({frontier_source})", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    if analysis:
        ax = fig.add_subplot(gs[1, 0])
        ps = list(depth_table)
        ax.plot(ps, [depth_table[p]["xy"][0] for p in ps], "-o", color=ORANGE, label="XY mixer")
        ax.plot(ps, [depth_table[p]["x"][0] for p in ps], "-o", color=BLUE, label="standard X mixer")
        ax.axhline(1 / n_feasible, ls="--", color=GREY, lw=1, label="random valid portfolio")
        ax.axhline(1 / 2 ** n, ls=":", color=GREY, lw=1, label="random bitstring")
        ax.set_xticks(ps)
        ax.set_xlabel("QAOA depth p")
        ax.set_ylabel("ideal P(optimal)")
        ax.set_yscale("log")
        ax.set_title("C. Depth scaling (INTERP warm starts)", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5)

        vmax = max(landscapes[m][3].max() for m in ("x", "xy"))
        for col, m, title in ((1, "x", "D. p = 1 landscape, standard"), (2, "xy", "E. p = 1 landscape, XY mixer")):
            ax = fig.add_subplot(gs[1, col])
            g_axis, b_axis, _, P = landscapes[m]
            im = ax.imshow(P, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                           extent=[g_axis[0], g_axis[-1], b_axis[0], b_axis[-1]])
            g1, b1, _ = tuned[m][1]
            ax.plot(g1[0] % (2 * np.pi), b1[0] % np.pi, "*", color="white", ms=13, mec=INK)
            ax.set_xlabel("gamma (scaled to the cost range)")
            ax.set_ylabel("beta")
            ax.set_title(f"{title}  (peak {P.max():.3f})", loc="left", fontsize=10)
            fig.colorbar(im, ax=ax, label="P(optimal), shared scale", fraction=0.046, pad=0.03)

    if noise:
        ax = fig.add_subplot(gs[2, :2])
        styles = [(("x", 2), "raw", BLUE, "--", "standard X, p = 2, raw"),
                  (("x", 2), "post", BLUE, "-", "standard X, p = 2, with error detection"),
                  (("xy", 2), "raw", ORANGE, "--", "XY, p = 2, raw"),
                  (("xy", 2), "post", ORANGE, "-", "XY, p = 2, with error detection"),
                  (("xy", 3), "post", "#8a3b12", "-", "XY, p = 3, with error detection")]
        for key, kind, col, ls, lab in styles:
            pts = sorted((r["cx_error"], r["p_opt_postselected" if kind == "post" else "p_opt_raw"])
                         for r in noise["rows"] if (r["mixer"], r["layers"]) == key)
            if pts:
                xs, ys = zip(*pts)
                ax.plot([100 * x for x in xs], ys, ls, marker="o", color=col, label=lab)
        ax.axhline(1 / n_feasible, ls="--", color=GREY, lw=1, label="random valid portfolio")
        ax.set_xlabel("two-qubit (CX) error rate, %")
        ax.set_ylabel("P(optimal)")
        ax.set_title("F. Noise robustness (depolarizing noise, Qiskit Aer, precomputed)", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)

        ax = fig.add_subplot(gs[2, 2])
        labels_b = ["raw", "with error\ndetection"]
        ax.bar(labels_b, [p_meas, p_post], color=[GREY, ORANGE])
        for xpos, v in enumerate([p_meas, p_post]):
            ax.text(xpos, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("P(optimal)")
        ax.set_title(f"G. This run: error detection ({1 - kept / total:.0%} flagged)", loc="left", fontsize=10)

    fig.suptitle(f"QAOA portfolio optimizer: pick {budget} of {n} assets", fontsize=14, x=0.01, ha="left")
    plt.savefig("qaoa_portfolio_dashboard.png", dpi=110, bbox_inches="tight")

    # ---------------------------------------------------------- JSON summary
    summary = {
        "assets": names, "budget": budget, "risk_aversion": risk_aversion, "mixer": mixer,
        "layers": layers, "shots": shots, "circuits_in_job": len(to_send), "frontier_source": frontier_source if frontier else None,
        "classical_optimum": fmt(x_opt, names), "greedy_matches_optimum": bool(index_of(x_gr) == opt),
        "qaoa_top_pick": fmt(bits[top], names), "optimum_found": bool(best == opt),
        "p_optimal_measured": round(p_meas, 4), "p_optimal_ideal": round(float(p_ideal[opt]), 4),
        "approximation_ratio": round(float(ratio), 4),
        "shots_flagged_as_errors": round(1 - kept / total, 4), "p_optimal_postselected": round(p_post, 4),
        "depth_study": {p: {m: round(v[0], 4) for m, v in row.items()} for p, row in depth_table.items()},
        "cx_gates": {k.replace("\n", " "): v for k, v in gate_report.items()},
        "frontier": [{"q": round(float(f["q"]), 3), "classical": fmt(bits[f["opt"]], names),
                      "qaoa": fmt(bits[f["top"]], names), "p_opt": round(f["p_meas"], 4),
                      "gap": round(f["gap"], 4)} for f in frontier],
    }
    try:
        from js import sendFile
        sendFile("summary.json", "application/json", json.dumps(summary, indent=2))
    except ImportError:
        with open("summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s. Dashboard: qaoa_portfolio_dashboard.png")
