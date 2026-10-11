**Budget-Preserving QAOA for Portfolio Selection: 21x better than standard QAOA on real market data, and the constraint doubles as free error detection**

![Results](docs/thumbnail.png)


Theme: **Quantum Impact**

**The problem**

An investor wants to hold exactly 4 of 8 US sector ETFs and balance return against risk. Standard QAOA handles the "exactly 4" rule with a penalty term. On this problem, the penalty swamps the risk/return signal, and standard QAOA finds the best portfolio *less often than random guessing*.

This project fixes that by building the rule into the circuit itself, then shows that the same rule detects hardware errors for free.

**Results at a glance (built-in simulator, 2000 shots, p = 2, risk aversion 0.5)**

| | Random valid portfolio | Standard QAOA | **Budget-preserving QAOA** |
|---|---|---|---|
| P(optimal portfolio) | 0.014 | 0.009 | **about 0.19** (ideal 0.189) |
| Shots that meet the budget | 100% | 63% | **100%** |
| Rank of the optimal portfolio | | 30th | **1st** |
| Optimal portfolio at 0.5% CX error, with error detection | 0.014 | 0.009 | **0.111** |
| On IonQ Forte 1 noise model, with error detection | 0.014 | | **0.055 to 0.062** (raw 0.018 to 0.021) |

- Optimal portfolio: **TECH, FIN, UTIL, INDU** (equal-weight return 20.0%, volatility 14.3%).
- Efficient frontier: QAOA recovers the classical optimum at **all 4** extra risk levels (same job on the built-in simulator; ideal local samples on IonQ).
- Measured results match the ideal statevector prediction, which validates the circuit end to end.

**What is new**

1. **Constraint built into the circuit.** The circuit starts in a Dicke state, an equal superposition of all 70 portfolios that hold exactly 4 assets. It mixes with XY swaps on a ring, so an asset can leave only when another joins. Every shot meets the budget and no penalty is needed (Hadfield et al., 2019).
2. **Gate-efficient Dicke state.** I built it with the split-and-cyclic-shift construction (Bärtschi and Eidenbenz, 2019) and hand-written 4-CX double-controlled rotations. This needs **118 CX gates, versus 247** for Qiskit's generic state preparation. The cost grows as O(n·k) instead of O(2ⁿ).
3. **The constraint is a free error detector.** On an ideal device the circuit never leaves the 4-asset subspace, so any shot with the wrong number of assets must contain an error. Discarding those shots costs no extra quantum resources. At a 0.5% CX error rate it **doubles** the success rate (0.052 to 0.111).
4. **Fair comparison.** Both mixers get identical tuning: a p = 1 grid search, COBYLA, INTERP warm starts for deeper circuits (Zhou et al., 2020), and the same restarts. Both also get the same error detection in the noise study.
5. **Exactly one quantum job per run.** Angles are tuned on a local statevector. On backends that accept batches, the main circuit and 4 frontier circuits go in a single `backend.run` call. IonQ backends accept one circuit per job, so there the code detects the backend, sends only the main circuit, and labels the frontier as ideal local samples.

**Result on IonQ Forte 1 (noise-model simulator, 2000 shots, p = 2, two independent runs)**

| | Value |
|---|---|
| Rank of the optimal portfolio | **1st**, even under device noise |
| Shots flagged as errors (wrong number of assets) | 66 to 68% |
| P(optimal), raw | 0.018 to 0.021 |
| **P(optimal), with error detection** | **0.055 to 0.062 (3x the raw value)** |
| For reference: random valid portfolio | 0.014 |
| For reference: standard QAOA, even with no noise | 0.009 |

- Error detection triples the success rate on IonQ's own device model, at no extra quantum cost.
- The result sits close to the Aer study below at a 1% CX error rate (0.048), so the two independent noise tests agree.
- Deeper circuits lose more to noise. Shorter Dicke preparation and fewer XY layers are the clearest routes to better hardware results.

**Noise study (depolarizing noise, Qiskit Aer, 8000 shots)**

| CX error rate | Standard QAOA + detection | Budget-preserving, raw | **Budget-preserving + detection** |
|---|---|---|---|
| 0% | 0.012 | 0.201 | **0.201** |
| 0.1% | 0.011 | 0.157 | **0.190** |
| 0.3% | 0.010 | 0.090 | **0.151** |
| 0.5% | 0.009 | 0.052 | **0.111** |
| 1.0% | 0.011 | 0.016 | **0.048** |

- Budget-preserving QAOA with detection stays well above random (0.014) up to about 1% CX error.
- Standard QAOA stays below random at every noise level.
- Reproduce with `tools/noise_study.py`. The results ship as `data/noise_study.json` because Qollab runs have no noise simulator for this.

**Dashboard from one run (built-in simulator)**

![Dashboard](docs/dashboard.png)

**What one run produces**

- A console report: classical baselines, a depth study for both mixers, CX gate counts, quantum results, error detection, the noise table and the frontier table.
- `qaoa_portfolio_dashboard.png` with 7 panels:
  - **A.** Measured portfolio distribution against the ideal prediction
  - **B.** Efficient frontier: all 70 valid portfolios, the classical frontier, and QAOA's picks
  - **C.** P(optimal) against depth p = 1 to 4 for both mixers
  - **D, E.** p = 1 landscapes of P(optimal) for both mixers, on a shared colour scale
  - **F.** Noise robustness, with and without error detection
  - **G.** Error detection in this run (0% flagged on an ideal simulator; try a noisy backend)
- `summary.json` with every key number

**How to run**

On Qollab (Python 3.14, Qiskit 2.x in the browser), press Run and pick a backend. To run locally instead, see "Running locally" below.


1. Press **Run** and pick a backend in **Select QPU**.
2. Optional run settings:

| Setting | Default | Meaning |
|---|---|---|
| `shots` | 2000 | shots per circuit |
| `layers` | 2 | QAOA depth p (1 to 6) |
| `budget` | 4 | number of assets to hold |
| `risk_aversion` | 0.5 | weight on risk |
| `mixer` | `xy` | `xy` (budget-preserving) or `x` (standard) |
| `frontier_points` | 4 | extra risk levels in the same job (0 to 6) |
| `analysis` | true | depth study, landscapes, gate comparison |

3. A full run takes about 45 seconds locally and a few minutes in the browser. For a quick run, set `analysis` to false and `frontier_points` to 0.
4. On IonQ simulators the frontier is computed from ideal local samples automatically, because IonQ accepts one circuit per job. Keep the browser tab in front while it runs, since background tabs run slower.

**Running locally**

```
pip install qiskit numpy scipy matplotlib
python run_local.py            # built-in Qiskit BasicSimulator
```

**The math**

- Choose x ∈ {0,1}⁸ with Σxᵢ = 4 to minimize q · xᵀΣx − μᵀx, where μ is the expected annual return, Σ the covariance, and q the risk aversion.
- Standard QAOA adds λ(Σxᵢ − 4)² and starts from |+⟩⊗⁸ with an RX mixer.
- Budget-preserving QAOA starts from the Dicke state |D(8,4)⟩ and applies exp(−iβ(XᵢXⱼ + YᵢYⱼ)/2) on each ring pair. That operator commutes with the total number of selected assets.

**Data**

Annualized expected return, volatility and correlation for 8 SPDR sector ETFs, computed from 753 trading days of adjusted closing prices (Oct 10 2023 to Oct 9 2026, Yahoo Finance via `yfinance`).

| Label | ETF | Label | ETF |
|---|---|---|---|
| TECH | XLK | UTIL | XLU |
| HLTH | XLV | STAP | XLP |
| ENRG | XLE | INDU | XLI |
| FIN | XLF | MATL | XLB |

- Expected returns are historical averages, not forecasts. This is not investment advice.
- To refresh the data or use other tickers, run `tools/make_dataset.py` on your own computer.

**Files**

| File | Purpose |
|---|---|
| `main.py` | entry point: baselines, study, single quantum job, error detection, dashboard |
| `qaoa.py` | Ising mapping, fast simulator for both mixers, INTERP tuning, Dicke circuit, circuit builder |
| `portfolio.py` | data loading, QUBO, brute force and greedy baselines |
| `data/assets.csv` | real ETF statistics |
| `data/noise_study.json` | precomputed noise study |
| `tools/make_dataset.py` | rebuilds `assets.csv` from market prices (off-platform) |
| `tools/noise_study.py` | reproduces the noise study with Qiskit Aer (off-platform) |

**Limitations**

- At 8 assets, brute force is instant and greedy also finds the optimum. The contribution is making QAOA itself work under a hard constraint, which matters at sizes where brute force is impossible.
- Angles are tuned on an exact local statevector, which does not scale past roughly 20 qubits. Larger problems would need parameter transfer or on-device tuning.
- The budget-preserving circuit is deeper (292 CX at p = 2, against 112 for standard). Error detection recovers much of the loss, but not all of it.
- The noise study uses a simple depolarizing model, not a calibrated device model.
- The tuner minimizes expected cost rather than P(optimal), so tuned angles do not always sit at the P(optimal) peak.

**References**

- Hadfield et al., "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz", Algorithms, 2019
- Bärtschi and Eidenbenz, "Deterministic Preparation of Dicke States", FCT 2019
- Zhou et al., "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices", PRX 2020

**What existed before the build window**

- No prior code, templates or forked projects were used. All code was written during the hackathon window.
- Libraries: Qiskit, NumPy, SciPy and Matplotlib (in the project); qiskit-aer and yfinance (in the off-platform tools).


**License**

MIT
