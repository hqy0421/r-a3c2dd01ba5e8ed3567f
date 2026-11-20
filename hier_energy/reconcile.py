"""Weighted non-negative reconciliation of base predictions.

Solves, for leaf values ``x``,

    min  sum_n w_n * (S_n . x - yhat_n)^2     subject to  x >= 0

where ``S`` is the hierarchy summing matrix (rows = nodes, columns = leaves)
and ``w_n > 0`` the given node weights.  With ``A = diag(sqrt(w)) @ S`` and
``b = sqrt(w) * yhat`` this is exactly ``min ||A x - b||`` s.t. ``x >= 0`` --
a convex QP solved to its *global* optimum by the NNLS active-set algorithm.
This is not a clip-at-zero heuristic: mass is shifted across leaves so the
weighted squared change over *all* nodes is minimised, and every node value
is then the sum of the reconciled leaves beneath it (coherent by
construction, non-negative because ``x >= 0`` and ``S >= 0``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


class ReconciliationError(RuntimeError):
    """The reconciliation QP could not be solved."""


def reconcile(base: pd.Series, weights: pd.Series, S: pd.DataFrame) -> dict:
    """Reconcile per-node base predictions into non-negative leaf values.

    ``base``: base predictions indexed by node. ``weights``: positive weight
    per node. ``S``: summing matrix (nodes x leaves).  Returns a JSON-able
    dict with reconciled leaves/nodes, the per-node cost breakdown, KKT
    optimality diagnostics and the clip-at-zero baseline for comparison.
    Raises :class:`ReconciliationError` on any solver problem.
    """
    nodes = list(base.index)
    leaves = list(S.columns)
    if list(S.index) != nodes:
        S = S.reindex(index=nodes)
    if S.isna().any().any():
        raise ReconciliationError("summing matrix does not cover every node")
    w = weights.reindex(nodes)
    if w.isna().any() or (w.to_numpy(dtype=float) <= 0).any():
        raise ReconciliationError("weights must be given and positive for every node")

    b = base.to_numpy(dtype=float)
    if not np.isfinite(b).all():
        raise ReconciliationError("base predictions contain NaN/inf")
    Sv = S.to_numpy(dtype=float)
    wv = w.to_numpy(dtype=float)
    sw = np.sqrt(wv)
    A = Sv * sw[:, None]
    bw = b * sw

    try:
        x, _ = nnls(A, bw)
    except Exception as exc:  # noqa: BLE001 - re-raised as ReconciliationError
        raise ReconciliationError(f"NNLS solver failed: {exc}") from exc
    if x is None or not np.isfinite(np.asarray(x, dtype=float)).all():
        raise ReconciliationError("NNLS returned non-finite leaf values")
    x = np.maximum(np.asarray(x, dtype=float), 0.0)  # guard against -1e-17 jitter

    node_values = Sv @ x
    delta = node_values - b
    contrib = wv * delta**2
    total_cost = float(contrib.sum())

    # KKT conditions for min f(x) s.t. x >= 0 with g = grad f (up to the
    # irrelevant factor 2):  x_i > 0 => g_i = 0 ;  x_i = 0 => g_i >= 0.
    g = Sv.T @ (wv * delta)
    active = x > 1e-10
    kkt = {
        "stationarity_max_abs": float(np.max(np.abs(g[active]))) if active.any() else 0.0,
        "dual_max_violation": float(np.max(-g[~active])) if (~active).any() else 0.0,
        "primal_min_leaf": float(x.min()),
    }

    # Naive baseline: clip negative leaf predictions at zero, aggregate
    # bottom-up.  Its cost is always >= the optimum found above.
    base_leaves = base.reindex(leaves).to_numpy(dtype=float)
    x_clip = np.maximum(base_leaves, 0.0)
    clip_nodes = Sv @ x_clip
    clip_total = float(np.sum(wv * (clip_nodes - b) ** 2))

    per_node = [
        {
            "node": n,
            "weight": float(wv[i]),
            "base": float(b[i]),
            "reconciled": float(node_values[i]),
            "delta": float(delta[i]),
            "weighted_sq_change": float(contrib[i]),
            "cost_share": float(contrib[i] / total_cost) if total_cost > 0 else 0.0,
        }
        for i, n in enumerate(nodes)
    ]
    return {
        "leaves": {l: float(x[i]) for i, l in enumerate(leaves)},
        "nodes": {n: float(node_values[i]) for i, n in enumerate(nodes)},
        "base": {n: float(b[i]) for i, n in enumerate(nodes)},
        "weights": {n: float(wv[i]) for i, n in enumerate(nodes)},
        "per_node": per_node,
        "total_cost": total_cost,
        "kkt": kkt,
        "clip_baseline": {
            "leaves": {l: float(x_clip[i]) for i, l in enumerate(leaves)},
            "total_cost": clip_total,
        },
    }
