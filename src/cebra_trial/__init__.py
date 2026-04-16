"""cebra_trial: Trial-aware contrastive learning wrapper for CEBRA.

Provides trial-aware extensions to the official CEBRA package without
modifying CEBRA's source code.

Usage::

    from cebra_trial import TrialCEBRA

    model = TrialCEBRA(
        conditional="trial_delta",
        time_offsets=10,
        delta=0.1,
        max_iterations=1000,
        output_dimension=3,
    )
    model.fit(X, y, trial_starts=trial_starts, trial_ends=trial_ends)
    embeddings = model.transform(X)
"""

from cebra_trial.cebra import TrialCEBRA
from cebra_trial.dataset import TrialTensorDataset
from cebra_trial.distribution import TrialAwareDistribution

__all__ = ["TrialCEBRA", "TrialTensorDataset", "TrialAwareDistribution"]
__version__ = "0.1.0"
