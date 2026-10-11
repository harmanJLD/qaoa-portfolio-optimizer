"""Portfolio problem: load data, build the QUBO, and solve it classically."""
import csv
import itertools
import numpy as np

DATA_FILE = "data/assets.csv"


def load_assets():
    """Read asset names, expected returns and the covariance matrix."""
    with open(DATA_FILE) as f:
        rows = list(csv.DictReader(f))
    names = [r["asset"] for r in rows]
    mu = np.array([float(r["expected_return"]) for r in rows])
    vol = np.array([float(r["volatility"]) for r in rows])
    corr = np.array([[float(r["corr_" + n]) for n in names] for r in rows])
    sigma = np.outer(vol, vol) * corr
    return names, mu, sigma


def build_qubo(mu, sigma, budget, risk_aversion, penalty=None):
    """Objective to minimize over binary x:
        q * x^T Sigma x  -  mu^T x  +  lambda * (sum(x) - budget)^2
    Returns the matrix Q and constant c so that cost(x) = x^T Q x + c.
    """
    n = len(mu)
    if penalty is None:
        # Large enough that breaking the budget never pays off.
        penalty = 2.0 * (np.abs(mu).sum() + risk_aversion * np.abs(sigma).sum())
    Q = risk_aversion * sigma.copy()
    Q -= np.diag(mu)
    # lambda * (sum x - B)^2 = lambda * (x^T 1 1^T x - 2B sum x + B^2), with x_i^2 = x_i
    Q += penalty * np.ones((n, n))
    Q -= 2.0 * penalty * budget * np.eye(n)
    c = penalty * budget ** 2
    return Q, c, penalty


def qubo_cost(x, Q, c):
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + c)


def portfolio_stats(x, mu, sigma):
    """Expected return and volatility of an equal-weight portfolio over the chosen assets."""
    x = np.asarray(x, dtype=float)
    w = x / x.sum() if x.sum() else x
    return float(mu @ w), float(np.sqrt(w @ sigma @ w))


def all_costs(Q, c):
    """Cost of every bitstring. Index i <-> bits of i with asset 0 as the lowest bit."""
    n = Q.shape[0]
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)) & 1).astype(float)
    return np.einsum("ki,ij,kj->k", bits, Q, bits) + c, bits


def brute_force(Q, c, budget):
    """Exact classical baseline: check every portfolio with `budget` assets."""
    n = Q.shape[0]
    best, best_x = None, None
    for combo in itertools.combinations(range(n), budget):
        x = np.zeros(n)
        x[list(combo)] = 1
        val = qubo_cost(x, Q, c)
        if best is None or val < best:
            best, best_x = val, x
    return best_x, best


def greedy(mu, sigma, budget, risk_aversion, Q, c):
    """Fast classical heuristic: add the asset that lowers the cost most, one at a time."""
    n = len(mu)
    x = np.zeros(n)
    for _ in range(budget):
        options = []
        for i in range(n):
            if x[i] == 0:
                y = x.copy()
                y[i] = 1
                options.append((qubo_cost(y, Q, c), i))
        x[min(options)[1]] = 1
    return x, qubo_cost(x, Q, c)
