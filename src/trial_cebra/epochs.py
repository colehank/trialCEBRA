"""Utilities for epoch-format neural data (ntrial × ntime × nneuro)."""

from typing import Tuple

import numpy as np
import numpy.typing as npt


def flatten_epochs(
    X: npt.NDArray,
    *y: npt.NDArray,
) -> Tuple[npt.NDArray, Tuple[npt.NDArray, ...], npt.NDArray, npt.NDArray]:
    """Convert epoch-format data to flat format with trial boundaries.

    Args:
        X: Neural data, shape ``(ntrial, ntime, nneuro)``.
        y: Label arrays (continuous **or** discrete, any order).  Each array
           is broadcast or reshaped according to its shape:

           **Continuous (float)**

           * ``(ntrial, nd)`` where ``nd ≠ ntime`` — per-trial; tiled to
             ``(ntrial * ntime, nd)``
           * ``(ntrial, ntime, nd)`` — per-timepoint; reshaped to
             ``(ntrial * ntime, nd)``

           **Discrete (int)**

           * ``(ntrial,)`` — per-trial class; tiled to ``(ntrial * ntime,)``
           * ``(ntrial, ntime)`` — per-timepoint class; reshaped to
             ``(ntrial * ntime,)``

           .. note::
               A 2-D array of shape ``(ntrial, ntime)`` is always treated as
               **per-timepoint** (already expanded), regardless of dtype.  If
               you have a per-trial continuous label whose feature dimension
               happens to equal ``ntime``, expand it to ``(ntrial, 1, ntime)``
               first.

    Returns:
        X_flat:       Neural data, shape ``(ntrial * ntime, nneuro)``.
        y_flat:       Tuple of flattened label arrays (dtypes preserved).
        trial_starts: Start index of each trial, shape ``(ntrial,)``.
        trial_ends:   End index (exclusive) of each trial, shape ``(ntrial,)``.

    Raises:
        ValueError: If ``X`` is not 3-D, or a label array has incompatible
            shape.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (ntrial, ntime, nneuro), got shape {X.shape}")
    ntrial, ntime, nneuro = X.shape

    X_flat = X.reshape(ntrial * ntime, nneuro)
    trial_starts = np.arange(ntrial, dtype=np.int64) * ntime
    trial_ends = trial_starts + ntime

    y_flat = []
    for i, yi in enumerate(y):
        yi = np.asarray(yi)
        if yi.ndim == 1:
            # (ntrial,) → per-trial discrete
            if yi.shape[0] != ntrial:
                raise ValueError(
                    f"y[{i}] has shape {yi.shape}; expected ({ntrial},) for per-trial labels"
                )
            y_flat.append(np.repeat(yi, ntime))
        elif yi.ndim == 2:
            if yi.shape == (ntrial, ntime):
                # per-timepoint, flatten
                y_flat.append(yi.reshape(ntrial * ntime))
            elif yi.shape[0] == ntrial:
                # (ntrial, d) with d != ntime → per-trial continuous
                y_flat.append(np.repeat(yi, ntime, axis=0))
            else:
                raise ValueError(
                    f"y[{i}] has shape {yi.shape}; expected ({ntrial}, ...) or ({ntrial}, {ntime})"
                )
        elif yi.ndim == 3:
            if yi.shape[:2] != (ntrial, ntime):
                raise ValueError(f"y[{i}] has shape {yi.shape}; expected ({ntrial}, {ntime}, d)")
            y_flat.append(yi.reshape(ntrial * ntime, yi.shape[2]))
        else:
            raise ValueError(f"y[{i}] must be 1-D, 2-D, or 3-D, got {yi.ndim}-D array")

    return X_flat, tuple(y_flat), trial_starts, trial_ends


def flatten_epochs_multisession(
    X_list,
    *y_lists,
):
    """Multi-session variant of :py:func:`flatten_epochs`.

    Args:
        X_list: list of 3-D arrays, each ``(ntrial_s, ntime_s, nneuro_s)``.
            Sessions may have heterogeneous ``ntrial_s`` / ``ntime_s`` / ``nneuro_s``.
        *y_lists: variadic; each is a list of per-session label arrays
            (length must match ``len(X_list)``).  Per-session labels follow
            the same broadcasting rules as :py:func:`flatten_epochs`.

    Returns:
        list of dicts, one per session, each with keys:
        ``"X_flat"``, ``"y_flat"`` (tuple), ``"trial_starts"``, ``"trial_ends"``,
        ``"ntrial"``, ``"ntime"``.

    Raises:
        ValueError: if ``X_list`` has fewer than 2 sessions, or any ``y_list``
            length does not match.
    """
    if not isinstance(X_list, list):
        raise ValueError(
            f"X_list must be a list of 3-D arrays for multisession; got {type(X_list).__name__}"
        )
    num_sessions = len(X_list)
    if num_sessions < 2:
        raise ValueError(f"multisession requires at least 2 sessions; got {num_sessions}")
    for i, yi_list in enumerate(y_lists):
        if not isinstance(yi_list, list) or len(yi_list) != num_sessions:
            raise ValueError(
                f"y[{i}] must be a list of length {num_sessions} (one per session); "
                f"got {type(yi_list).__name__} of length "
                f"{len(yi_list) if hasattr(yi_list, '__len__') else 'N/A'}"
            )

    out = []
    for s in range(num_sessions):
        Xs = np.asarray(X_list[s])
        if Xs.ndim != 3:
            raise ValueError(
                f"X_list[{s}] must be 3-D (ntrial_s, ntime_s, nneuro_s); got shape {Xs.shape}"
            )
        ys_s = tuple(yi_list[s] for yi_list in y_lists)
        Xs_flat, ys_flat, ts, te = flatten_epochs(Xs, *ys_s)
        out.append(
            dict(
                X_flat=Xs_flat,
                y_flat=ys_flat,
                trial_starts=ts,
                trial_ends=te,
                ntrial=Xs.shape[0],
                ntime=Xs.shape[1],
            )
        )
    return out
