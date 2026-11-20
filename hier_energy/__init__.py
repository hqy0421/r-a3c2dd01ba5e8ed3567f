"""Hierarchical sub-meter energy-demand forecasting with tsfresh.

Pipeline contract
-----------------
1. Leaf raw signals are aggregated *bottom-up* along the declared tree for the
   same window and the same timestamps; tsfresh features are then extracted
   from each node's own aggregated series.  Leaf (non-linear) features are
   never summed to synthesise a node's features.
2. Every node fits an independent regressor on its own training features with
   next-period labels composed from its leaves.  Preprocessing (imputation,
   variance filter, scaling) is fitted on the training windows only and
   re-applied verbatim to new windows.
3. Base predictions are reconciled by solving, exactly and globally,
       min_{x >= 0}  sum_n w_n * (S_n . x - yhat_n)^2
   via non-negative least squares.  Every node value is then the sum of the
   reconciled leaves beneath it, so the hierarchy is coherent by construction
   and no demand is negative.  Clipping at zero is *not* used.
4. Missing leaves, population misalignment, fit failures and solver failures
   are recorded and always yield an overall non-success status.
"""

from .hierarchy import Hierarchy, HierarchyError
from .windows import DataError
from .features import FC_PARAMETERS, extract_node_features
from .model import NodeModel
from .reconcile import ReconciliationError, reconcile
from .pipeline import run_pipeline

__all__ = [
    "Hierarchy",
    "HierarchyError",
    "DataError",
    "FC_PARAMETERS",
    "extract_node_features",
    "NodeModel",
    "ReconciliationError",
    "reconcile",
    "run_pipeline",
]
