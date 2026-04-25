"""Shared test fixtures for trial_cebra tests."""

import torch


def make_epoch_data(ntrial=8, ntime=50, nneuro=32, nd=10, device="cpu"):
    """Create synthetic epoch-format data for new 3-conditional design.

    Returns:
        X:   (ntrial, ntime, nneuro) — neural data
        y2d: (ntrial, nd)            — trial-level labels (constant within trial)
        y3d: (ntrial, ntime, nd)     — timepoint-level labels (varies within trial)
    """
    X = torch.randn(ntrial, ntime, nneuro, device=device)
    # y2d: each trial has a distinct constant label; make pairs similar for delta tests
    y2d = torch.randn(ntrial, nd, device=device)
    half = ntrial // 2
    for i in range(min(2, half)):
        y2d[half + i] = y2d[i] + 0.01 * torch.randn(nd, device=device)
    # y3d: smooth ramp within each trial so Δstim is non-zero
    t_idx = torch.linspace(0, 1, ntime, device=device).unsqueeze(-1)  # (ntime, 1)
    y3d = y2d.unsqueeze(1) * t_idx.unsqueeze(0)  # (ntrial, ntime, nd)
    y3d = y3d + 0.01 * torch.randn(ntrial, ntime, nd, device=device)
    return X, y2d, y3d


def make_trial_data(
    num_trials=10, trial_length=80, embedding_dim=16, gap=20, num_similar_pairs=2, device="cpu"
):
    """Create synthetic trial data with controlled stimulus similarity.

    Returns:
        Tuple of (continuous, trial_starts, trial_ends, total_samples).
    """
    total_samples = num_trials * (trial_length + gap)
    continuous = torch.randn(total_samples, embedding_dim, device=device)

    trial_starts = []
    trial_ends = []
    for i in range(num_trials):
        start = i * (trial_length + gap) + gap
        end = start + trial_length
        trial_starts.append(start)
        trial_ends.append(end)
        continuous[start:end] = torch.randn(1, embedding_dim, device=device)

    trial_starts = torch.tensor(trial_starts, dtype=torch.long, device=device)
    trial_ends = torch.tensor(trial_ends, dtype=torch.long, device=device)

    # Make similar pairs: trial i and trial num_trials//2 + i share embeddings
    half = num_trials // 2
    for i in range(min(num_similar_pairs, half)):
        s_src = int(trial_starts[i])
        s_tgt = int(trial_starts[half + i])
        continuous[s_tgt : s_tgt + trial_length] = continuous[
            s_src : s_src + trial_length
        ] + 0.001 * torch.randn(trial_length, embedding_dim, device=device)

    return continuous, trial_starts, trial_ends, total_samples
