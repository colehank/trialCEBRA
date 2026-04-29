"""trial_cebra: Trial-aware contrastive learning wrapper for CEBRA.

Provides trial-aware extensions to the official CEBRA package without
modifying CEBRA's source code.

Usage::

    import numpy as np
    from trial_cebra import TrialCEBRA

    X = np.random.randn(40, 50, 64).astype("float32")  # (ntrial, ntime, nneuro)
    y = np.random.randn(40, 8).astype("float32")        # (ntrial, nd)

    model = TrialCEBRA(
        conditional="delta",
        time_offsets=10,
        delta=0.1,
        max_iterations=1000,
        batch_size=512,
        output_dimension=3,
    )
    model.fit(X, y)

    emb  = model.transform(X)          # (ntrial, ntime, 3) — shape preserved
    gof  = model.goodness_of_fit_score(X, y)
    hist = model.goodness_of_fit_history()
"""

from importlib.metadata import PackageNotFoundError, version

from trial_cebra.cebra import TrialCEBRA
from trial_cebra.dataset import TrialTensorDataset
from trial_cebra.distribution import TrialAwareDistribution
from trial_cebra.epochs import flatten_epochs, flatten_epochs_multisession
from trial_cebra.multisession import (
    TrialAwareMultisessionLoader,
    TrialAwareMultisessionSampler,
)

__all__ = [
    "TrialCEBRA",
    "TrialTensorDataset",
    "TrialAwareDistribution",
    "TrialAwareMultisessionSampler",
    "TrialAwareMultisessionLoader",
    "flatten_epochs",
    "flatten_epochs_multisession",
]

try:
    __version__ = version("TrialCEBRA")
except PackageNotFoundError:
    __version__ = "unknown"
