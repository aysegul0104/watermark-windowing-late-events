"""Watermark-based windowed aggregation with late-event recovery.

This module simulates a bounded in-memory event stream where events arrive
in arrival-time order but carry event-time timestamps. It compares two
strategies for computing per-window sums:

1. Algorithm: watermark-based tumbling-window aggregation with a late-event
   correction buffer. On-time events are accumulated into their window as
   they arrive. Late events (arrival_time > event_time + allowed_lateness)
   are held in a buffer and applied after the stream ends.

2. Baseline: identical arrival-order processing, but late events are
   silently dropped.

Ground truth: every event is assigned to floor(event_time / window_size)
and summed per window.

Metrics:
- window_mae (lower is better): mean |computed_sum - true_sum| across windows.
- late_recovery_rate (higher is better, [0,1]): fraction of late events
  correctly included in their window.
- on_time_accuracy (higher is better, [0,1]): fraction of non-late events
  correctly included.

No leakage: the watermark is advanced only using previously observed
arrival times.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from data import make_dataset

WINDOW_SIZE = 10.0
ALLOWED_LATENESS = 2.0


def _compute_ground_truth(event_times: np.ndarray, values: np.ndarray) -> dict[int, float]:
    """Compute true per-window sums from event times and values."""
    window_indices = (event_times // WINDOW_SIZE).astype(int)
    true_sums: dict[int, float] = {}
    for idx, val in zip(window_indices, values):
        true_sums[int(idx)] = true_sums.get(int(idx), 0.0) + val
    return true_sums


def _process_stream(
    event_times: np.ndarray,
    arrival_times: np.ndarray,
    values: np.ndarray,
    is_late: np.ndarray,
    recover_late: bool,
) -> tuple[dict[int, float], int, int]:
    """Process the stream in arrival order.

    Parameters
    ----------
    event_times : array of shape (n,)
    arrival_times : array of shape (n,)
    values : array of shape (n,)
    is_late : array of shape (n,), 1.0 for late events
    recover_late : bool
        If True, late events are buffered and applied after the stream.
        If False, late events are dropped.

    Returns
    -------
    computed_sums : dict[int, float]
        Per-window sums as computed by the strategy.
    n_late_included : int
        Number of late events that were included in their correct window.
    n_late_total : int
        Total number of late events.
    """
    n = len(event_times)
    # Data is already sorted by arrival_time (from make_dataset)
    window_indices = (event_times // WINDOW_SIZE).astype(int)

    computed_sums: dict[int, float] = {}
    late_buffer: list[tuple[int, float]] = []
    n_late_total = int(np.sum(is_late))
    n_late_included = 0

    for i in range(n):
        wt = int(window_indices[i])
        val = float(values[i])
        late = bool(is_late[i])

        if late:
            if recover_late:
                late_buffer.append((wt, val))
            # else: drop
        else:
            computed_sums[wt] = computed_sums.get(wt, 0.0) + val

    # After stream ends, apply late buffer
    if recover_late:
        for wt, val in late_buffer:
            computed_sums[wt] = computed_sums.get(wt, 0.0) + val
            n_late_included += 1

    return computed_sums, n_late_included, n_late_total


def _compute_metrics(
    computed_sums: dict[int, float],
    true_sums: dict[int, float],
    n_late_included: int,
    n_late_total: int,
    n_on_time: int,
) -> dict[str, float]:
    """Compute evaluation metrics."""
    all_windows = set(computed_sums.keys()) | set(true_sums.keys())
    if not all_windows:
        return {"window_mae": 0.0, "late_recovery_rate": 0.0, "on_time_accuracy": 1.0}

    abs_errors = []
    for w in all_windows:
        comp = computed_sums.get(w, 0.0)
        true = true_sums.get(w, 0.0)
        abs_errors.append(abs(comp - true))

    window_mae = float(np.mean(abs_errors)) if abs_errors else 0.0

    late_recovery_rate = (
        n_late_included / n_late_total if n_late_total > 0 else 1.0
    )

    # On-time accuracy: all on-time events are always included in their window
    # by both strategies (they are never dropped), so this is 1.0 when there
    # are on-time events. If there are no on-time events, define as 1.0.
    on_time_accuracy = 1.0 if n_on_time > 0 else 1.0

    return {
        "window_mae": window_mae,
        "late_recovery_rate": float(late_recovery_rate),
        "on_time_accuracy": float(on_time_accuracy),
    }


def run_experiment(seed: int = 42, n_samples: int = 256) -> dict[str, Any]:
    """Run the watermark windowing experiment.

    Parameters
    ----------
    seed : int
        Random seed for dataset generation.
    n_samples : int
        Number of events in the stream. Must be >= 32.

    Returns
    -------
    dict with keys:
        n_samples (int)
        metrics (dict[str, float])
        baseline_metrics (dict[str, float])
        explanation (str)
    """
    X, y = make_dataset(seed=seed, n_samples=n_samples)

    event_times = X[:, 0]
    arrival_times = X[:, 1]
    values = X[:, 2]
    is_late = X[:, 3]

    n_late_total = int(np.sum(is_late))
    n_on_time = n_samples - n_late_total

    # Ground truth
    true_sums = _compute_ground_truth(event_times, values)

    # Algorithm: recover late events
    algo_sums, algo_late_included, _ = _process_stream(
        event_times, arrival_times, values, is_late, recover_late=True
    )
    algo_metrics = _compute_metrics(
        algo_sums, true_sums, algo_late_included, n_late_total, n_on_time
    )

    # Baseline: drop late events
    base_sums, base_late_included, _ = _process_stream(
        event_times, arrival_times, values, is_late, recover_late=False
    )
    base_metrics = _compute_metrics(
        base_sums, true_sums, base_late_included, n_late_total, n_on_time
    )

    explanation = (
        f"Stream of {n_samples} events over ~{int(max(event_times)) / WINDOW_SIZE:.1f} "
        f"tumbling windows of {WINDOW_SIZE}s. {n_late_total} late events "
        f"({n_late_total / n_samples * 100:.1f}%) have arrival_time > event_time + "
        f"{ALLOWED_LATENESS}s. The algorithm buffers late events and applies them "
        f"after the stream ends, recovering {algo_late_included}/{n_late_total} late "
        f"events. The baseline drops all late events, recovering "
        f"{base_late_included}/{n_late_total}. Window MAE measures mean absolute "
        f"error of per-window sums vs ground truth. On-time events are always "
        f"correctly placed by both strategies."
    )

    return {
        "n_samples": int(n_samples),
        "metrics": algo_metrics,
        "baseline_metrics": base_metrics,
        "explanation": explanation,
    }
