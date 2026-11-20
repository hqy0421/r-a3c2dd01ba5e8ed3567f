"""Deterministic synthetic sub-meter demo.

Run:  python -m hier_energy.demo [--out DIR] [--also-failure]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .hierarchy import Hierarchy
from .pipeline import run_pipeline

TREE = {
    "total": ["building_a", "building_b"],
    "building_a": ["floor_a1", "floor_a2"],
    "building_b": ["floor_b1"],
}

# Given positive reconciliation weights: the total matters most, then the
# buildings, then the individual floors.
WEIGHTS = {
    "total": 3.0,
    "building_a": 2.0,
    "building_b": 2.0,
    "floor_a1": 1.0,
    "floor_a2": 1.0,
    "floor_b1": 1.0,
}

WINDOW_SIZE = 24   # hourly samples -> one day per window
N_WINDOWS = 42     # 41 labelled training windows + 1 window to predict
SEED = 20261120


def _ar1(rng, n, phi, scale):
    eps = rng.normal(0.0, scale, n)
    out = np.empty(n)
    acc = 0.0
    for i, e in enumerate(eps):
        acc = phi * acc + e
        out[i] = acc
    return out


def synthetic_meters(seed: int = SEED, n_windows: int = N_WINDOWS,
                     window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """Long-format (time, meter, value) synthetic leaf signals, all >= 0.

    A weekly pattern over windows makes next-window demand learnable from
    current-window features; per-leaf AR(1) noise makes the per-node models
    disagree enough that reconciliation does real work.
    """
    rng = np.random.default_rng(seed)
    n = n_windows * window_size
    times = pd.date_range("2026-01-01", periods=n, freq="h")
    t = np.arange(n)
    win = t // window_size
    hour = t % window_size
    weekly = 1.0 + 0.30 * np.sin(2.0 * np.pi * (win % 7) / 7.0)

    def leaf(base, phase, noise_scale):
        daily = 1.0 + 0.45 * np.sin(2.0 * np.pi * (hour - phase) / window_size)
        signal = base * weekly * daily + _ar1(rng, n, 0.6, noise_scale)
        return np.clip(signal, 0.1, None)

    series = {
        "floor_a1": leaf(12.0, phase=8, noise_scale=0.8),
        "floor_a2": leaf(8.0, phase=10, noise_scale=0.6),
        "floor_b1": leaf(20.0, phase=6, noise_scale=1.2),
    }
    frames = [
        pd.DataFrame({"time": times, "meter": name, "value": values})
        for name, values in series.items()
    ]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/hier_energy_output")
    parser.add_argument(
        "--also-failure",
        action="store_true",
        help="additionally run a missing-leaf scenario to show honest failure reporting",
    )
    args = parser.parse_args()
    out = Path(args.out)

    raw = synthetic_meters()
    hierarchy = Hierarchy(TREE)
    result = run_pipeline(
        raw, hierarchy, WEIGHTS, window_size=WINDOW_SIZE, out_dir=out
    )

    print(f"success: {result['success']}")
    win = result["windowing"]
    print(f"windows: {win['n_windows']} (train {len(win['train_windows'])}, "
          f"predict {win['predict_window']})")
    print("base predictions:")
    for node in hierarchy.nodes:
        print(f"  {node:<12} yhat={result['base_predictions'][node]:12.3f} "
              f"(train R2={result['train_r2'][node]:.3f})")
    rec = result["reconciliation"]
    print("reconciled leaves:")
    for leaf, value in rec["leaves"].items():
        print(f"  {leaf:<12} x*={value:12.3f}")
    print(f"total cost: {rec['total_cost']:.6f} "
          f"(clip-at-zero baseline: {rec['clip_baseline']['total_cost']:.6f})")
    print(f"coherence: root={rec['coherence']['root_value']:.4f} == "
          f"leaf sum={rec['coherence']['leaf_sum']:.4f}")
    print(f"report: {result['report_path']}")

    if args.also_failure:
        broken = raw[raw["meter"] != "floor_b1"]  # drop one leaf meter
        failed = run_pipeline(
            broken, hierarchy, WEIGHTS, window_size=WINDOW_SIZE,
            out_dir=out / "missing_leaf_demo",
        )
        print(f"\nmissing-leaf scenario success: {failed['success']}")
        for f in failed["failures"]:
            print(f"  [{f['stage']}] {f['code']}: {f['message']}")
        print(f"status: {failed['status_path']}")


if __name__ == "__main__":
    main()
