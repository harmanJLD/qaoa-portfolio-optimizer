"""QAOA for budget-constrained portfolio selection.

Two mixers:
  "x"  : standard QAOA. Uniform start, RX mixer, budget enforced by a penalty term.
  "xy" : constraint-preserving QAOA (Hadfield et al., 2019). Starts in a Dicke state,
         the equal superposition of every portfolio holding exactly `budget` assets,
         and mixes with XY swaps on a ring. An asset can leave only when another
         joins, so every shot meets the budget and no penalty is needed.

Angles are tuned on a fast local numpy statevector (no quantum jobs), with INTERP
warm starts between depths (Zhou et al., 2020). Final circuits are compiled to
H/RX/RY/RZ/CX so any backend accepts them.

Bit convention everywhere: asset i <-> qubit i <-> bit i of the basis-state index.
"""
import itertools
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile

BASIS = ["h", "rx", "ry", "rz", "cx"]


# ----------------------------------------------------------------- problem mapping
def qubo_to_ising(Q, c):
    """Map x = (1 - z) / 2 so cost = sum h_i z_i + sum_{i<j} J_ij z_i z_j + offset."""
    n = Q.shape[0]
    Qs = (Q + Q.T) / 2
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = c
    for i in range(n):
        offset += Qs[i, i] / 2
        h[i] -= Qs[i, i] / 2
        for j in range(n):
            if j != i:
                offset += Qs[i, j] / 4
                h[i] -= Qs[i, j] / 2
                if j > i:
                    J[i, j] += Qs[i, j] / 2
    return h, J, offset


def ring_pairs(n):
    return [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]


# ----------------------------------------------------------------- fast simulator
class Simulator:
    """Exact QAOA statevector for one problem, with precomputed index tables."""

    def __init__(self, costs, n, mixer, budget):
        self.n, self.mixer, self.budget = n, mixer, budget
        idx = np.arange(2 ** n)
        weights = np.array([bin(i).count("1") for i in idx])
        if mixer == "xy":
            self.support = weights == budget
            self.psi0 = self.support / np.sqrt(self.support.sum()) + 0j
            self.pairs = []
            for i, j in ring_pairs(n):
                s = idx[((idx >> i) & 1) != ((idx >> j) & 1)]
                self.pairs.append((s, s ^ (1 << i) ^ (1 << j)))
        else:
            self.support = np.ones(2 ** n, bool)
            self.psi0 = np.full(2 ** n, 1 / np.sqrt(2 ** n), dtype=complex)
        ref = costs[self.support]
        self.c_min, self.scale = ref.min(), (ref.max() - ref.min()) or 1.0
        self.norm = (costs - self.c_min) / self.scale  # 0 = best, 1 = worst feasible

    def state(self, gammas, betas):
        """gammas are in normalized units (see to_physical)."""
        psi = self.psi0.copy()
        n = self.n
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * self.norm)
            cb, sb = np.cos(b), -1j * np.sin(b)
            if self.mixer == "xy":
                for s, t in self.pairs:
                    psi_t = psi[t]
                    psi[s] = cb * psi[s] + sb * psi_t
            else:
                psi = psi.reshape([2] * n)
                for q in range(n):
                    ax = n - 1 - q
                    a0, a1 = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
                    psi = np.stack([cb * a0 + sb * a1, sb * a0 + cb * a1], axis=ax)
                psi = psi.reshape(-1)
        return psi

    def probs(self, gammas, betas):
        return np.abs(self.state(gammas, betas)) ** 2

    def energy(self, theta):
        p = len(theta) // 2
        return float(self.probs(theta[:p], theta[p:]) @ self.norm)

    def to_physical(self, gammas):
        return np.asarray(gammas) / self.scale

    def landscape(self, n_grid=36, target=None):
        """p = 1 grid over (gamma, beta): expected normalized cost, and P(target) if given."""
        g_axis = np.linspace(0, 2 * np.pi, n_grid)
        b_axis = np.linspace(0, np.pi, n_grid)
        E = np.zeros((n_grid, n_grid))
        P = np.zeros((n_grid, n_grid))
        for r, b in enumerate(b_axis):
            for c, g in enumerate(g_axis):
                pr = self.probs([g], [b])
                E[r, c] = pr @ self.norm
                if target is not None:
                    P[r, c] = pr[target]
        return g_axis, b_axis, E, P


# ----------------------------------------------------------------- angle tuning
def _interp(v):
    """INTERP rule: angles for depth p+1 from the optimum at depth p."""
    p = len(v)
    padded = np.concatenate([[0.0], v, [0.0]])
    return np.array([(i / p) * padded[i] + ((p - i) / p) * padded[i + 1] for i in range(p + 1)])


def _local(sim, x0, maxiter=400):
    res = minimize(sim.energy, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
    return res.x, res.fun


def tune_depths(sim, max_layers, n_grid=36, extra_restarts=2, seed=7, target=None):
    """Tune p = 1..max_layers. p = 1 starts from the best point on a landscape grid;
    each deeper level starts from INTERP of the level below plus a few random starts.
    Returns {p: (gammas, betas, energy)}."""
    rng = np.random.default_rng(seed)
    g_axis, b_axis, E, P = sim.landscape(n_grid, target)
    bi, gi = np.unravel_index(np.argmin(E), E.shape)
    x, f = _local(sim, np.array([g_axis[gi], b_axis[bi]]))
    out = {1: (x[:1], x[1:], f)}
    for p in range(2, max_layers + 1):
        g_prev, b_prev, _ = out[p - 1]
        starts = [np.concatenate([_interp(g_prev), _interp(b_prev)]),
                  # an extra zero layer reproduces depth p-1 exactly, so deeper is never worse
                  np.concatenate([g_prev, [0.0], b_prev, [0.0]])]
        for _ in range(extra_restarts):
            starts.append(np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)]))
        best = min((_local(sim, s) for s in starts), key=lambda r: r[1])
        out[p] = (best[0][:p], best[0][p:], best[1])
    return out, (g_axis, b_axis, E, P)


def tune_from(sim, gammas, betas, extra_restarts=1, seed=3):
    """Re-tune angles for a related problem, warm-started from known good angles."""
    rng = np.random.default_rng(seed)
    p = len(gammas)
    starts = [np.concatenate([gammas, betas])]
    for _ in range(extra_restarts):
        starts.append(np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)]))
    best = min((_local(sim, s) for s in starts), key=lambda r: r[1])
    return best[0][:p], best[0][p:]


# ----------------------------------------------------------------- circuits
def _ccry(qc, theta, a, b, t):
    """Doubly controlled RY with 4 CX (exact: angle is theta only when a = b = 1)."""
    qc.ry(theta / 4, t); qc.cx(a, t)
    qc.ry(-theta / 4, t); qc.cx(b, t)
    qc.ry(theta / 4, t); qc.cx(a, t)
    qc.ry(-theta / 4, t); qc.cx(b, t)


def _scs(qc, m, k):
    """Split-and-cyclic-shift block of Baertschi & Eidenbenz (2019), qubits 0..m-1."""
    th = 2 * np.arccos(np.sqrt(1 / m))
    a, t = m - 1, m - 2
    qc.cx(t, a)
    qc.ry(th / 2, t); qc.cx(a, t); qc.ry(-th / 2, t); qc.cx(a, t)  # CRY(th), control a
    qc.cx(t, a)
    for l in range(2, k + 1):
        t, ctrl2 = m - 1 - l, m - l
        qc.cx(t, m - 1)
        _ccry(qc, 2 * np.arccos(np.sqrt(l / m)), m - 1, ctrl2, t)
        qc.cx(t, m - 1)


def dicke_circuit(n, k):
    """Prepare the Dicke state |D(n, k)> with O(n k) gates (no generic state prep)."""
    qc = QuantumCircuit(n)
    for q in range(n - k, n):
        qc.x(q)
    for m in range(n, k, -1):
        _scs(qc, m, k)
    for m in range(k, 1, -1):
        _scs(qc, m, m - 1)
    return qc


def build_circuit(h, J, gammas, betas, mixer="x", budget=None, name="qaoa"):
    """gammas here are physical (already divided by the cost scale)."""
    n = len(h)
    qc = QuantumCircuit(n, n, name=name)
    if mixer == "xy":
        qc.compose(dicke_circuit(n, budget), inplace=True)
    else:
        qc.h(range(n))
    for g, b in zip(gammas, betas):
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    qc.cx(i, j)
                    qc.rz(2 * g * J[i, j], j)
                    qc.cx(i, j)
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * g * h[i], i)
        if mixer == "xy":
            for i, j in ring_pairs(n):
                qc.rxx(b, i, j)
                qc.ryy(b, i, j)
        else:
            qc.rx(2 * b, range(n))
    qc.measure(range(n), range(n))
    return transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=1)


def dicke_target(n, k):
    v = np.zeros(2 ** n)
    for combo in itertools.combinations(range(n), k):
        v[sum(1 << q for q in combo)] = 1
    return v / np.linalg.norm(v)
