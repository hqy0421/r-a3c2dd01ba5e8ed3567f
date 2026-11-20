"""tsfresh feature extraction on the *aggregated* node series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from tsfresh import extract_features

# A deliberately mixed set: linear statistics plus non-linear ones
# (abs_energy, c3, fft coefficients, ...).  The non-linear entries are exactly
# why summing leaf features would be wrong: e.g. abs_energy(a + b) !=
# abs_energy(a) + abs_energy(b).  Features are therefore computed per node on
# the node's own aggregated raw series.
FC_PARAMETERS = {
    "mean": None,
    "median": None,
    "standard_deviation": None,
    "variance": None,
    "minimum": None,
    "maximum": None,
    "sum_values": None,
    "abs_energy": None,
    "root_mean_square": None,
    "mean_abs_change": None,
    "mean_change": None,
    "skewness": None,
    "kurtosis": None,
    "quantile": [{"q": 0.25}, {"q": 0.75}],
    "first_location_of_maximum": None,
    "last_location_of_maximum": None,
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    "number_peaks": [{"n": 3}],
    "autocorrelation": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
    "agg_autocorrelation": [{"f_agg": "mean", "maxlag": 4}],
    "c3": [{"lag": 1}],
    "linear_trend": [
        {"attr": "slope"},
        {"attr": "intercept"},
        {"attr": "rvalue"},
    ],
    "fft_coefficient": [
        {"coeff": 1, "attr": "abs"},
        {"coeff": 2, "attr": "abs"},
    ],
}


def extract_node_features(node: str, window_series: dict, fc_parameters=None) -> pd.DataFrame:
    """Extract tsfresh features for one node from its aggregated series.

    ``window_series`` maps window id -> pd.Series of the node's aggregated raw
    signal within that window.  Returns a DataFrame indexed by window id with
    one column per extracted feature.  Extraction is serial (``n_jobs=0``) so
    results are deterministic.
    """
    settings = fc_parameters if fc_parameters is not None else FC_PARAMETERS
    frames = []
    for wid, series in window_series.items():
        values = np.asarray(series, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "id": wid,
                    "time": np.arange(len(values)),
                    "value": values,
                }
            )
        )
    if not frames:
        raise ValueError(f"node {node!r}: no windows to extract features from")
    long = pd.concat(frames, ignore_index=True)
    features = extract_features(
        long,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=settings,
        n_jobs=0,
        show_warnings=False,
        disable_progressbar=True,
    )
    features.index.name = "window"
    # tsfresh prefixes columns with the value-column kind ("value__"); strip it
    features.columns = [c.removeprefix("value__") for c in features.columns]
    return features.sort_index().sort_index(axis=1)
