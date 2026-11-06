"""Synthetic event-stream dataset for watermark windowing experiments."""
from __future__ import annotations

import numpy as np


def make_dataset(seed: int = 42, n_samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Generate a bounded in-memory event stream.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 4), dtype float64
        Columns: [event_time, arrival_time, value, is_late].
        event_time and arrival_time are in seconds (floats).
        value is a positive magnitude in [1, 10].
        is_late is 1.0 if the event is late, else 0.0.
    y : np.ndarray, shape (n_samples,), dtype int64
        True window index: floor(event_time / window_size).
        window_size is fixed at 10.0 seconds.

    Notes
    -----
    - ~15% of events are late: arrival_time > event_time + allowed_lateness.
      allowed_lateness is fixed at 2.0 seconds.
    - arrival_time is always >= event_time (no negative skew).
    - Events are returned in arrival_time order (simulating stream arrival).
    - No disk I/O; purely in-memory.
    """
    if n_samples < 32:
        raise ValueError(f"n_samples must be >= 32, got {n_samples}")

    rng = np.random.default_rng(seed)

    window_size = 10.0
    allowed_lateness = 2.0
    late_fraction = 0.15

    # Generate event times spread over a range large enough for multiple windows
    # Use a range of about 5 windows so we have ~5 distinct windows
    n_windows = max(3, n_samples // 50)
    total_time = n_windows * window_size

    # Event times: uniform over [0, total_time)
    event_times = rng.uniform(0.0, total_time, size=n_samples)

    # Values: positive magnitudes
    values = rng.uniform(1.0, 10.0, size=n_samples)

    # Determine which events are late (~15%)
    n_late = int(round(n_samples * late_fraction))
    late_mask = np.zeros(n_samples, dtype=bool)
    if n_late > 0:
        late_indices = rng.choice(n_samples, size=n_late, replace=False)
        late_mask[late_indices] = True

    # Arrival times:
    # - On-time events: arrival_time = event_time + small non-negative jitter
    #   (jitter in [0, allowed_lateness])
    # - Late events: arrival_time = event_time + jitter in (allowed_lateness, allowed_lateness + 5]
    jitter_on_time = rng.uniform(0.0, allowed_lateness, size=n_samples)
    jitter_late = rng.uniform(allowed_lateness + 0.1, allowed_lateness + 5.0, size=n_samples)

    arrival_times = event_times + jitter_on_time
    arrival_times[late_mask] = event_times[late_mask] + jitter_late[late_mask]

    # Ensure arrival_time >= event_time (should already hold, but guard)
    arrival_times = np.maximum(arrival_times, event_times)

    # True window index
    y = (event_times // window_size).astype(np.int64)

    # is_late as float for X
    is_late_float = late_mask.astype(np.float64)

    # Build X
    X = np.column_stack([event_times, arrival_times, values, is_late_float])

    # Sort by arrival_time to simulate stream arrival order
    sort_idx = np.argsort(arrival_times, kind="stable")
    X = X[sort_idx]
    y = y[sort_idx]

    return X, y
