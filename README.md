# Watermark-Based Windowed Aggregation with Late-Event Recovery

This project simulates a bounded in-memory event stream to compare watermark-based tumbling-window aggregation with late-event correction against a naive drop-late baseline. It measures window MAE and late-recovery rate to quantify the impact of late-event handling strategies on stream processing accuracy.

## Problem Statement

In real-time stream processing, events often arrive out of order relative to their event-time timestamps. A watermark-based approach tracks the maximum observed event time minus an allowed lateness threshold to determine when a window can be safely closed. Events arriving after the watermark has passed their window are considered "late."

This project addresses the following questions:
1. How does buffering and correcting late events after stream completion compare to silently dropping them?
2. What is the impact of late-event handling on per-window aggregation accuracy?
3. Can a simple correction buffer recover late events without requiring complex state management?

## Implementation

### Architecture

The project consists of four main components:

1. **`data.py`**: Generates a synthetic event stream with controlled lateness. Each event has an event time, arrival time, value, and a late flag. Events are sorted by arrival time to simulate stream order.

2. **`app.py`**: Implements the core experiment:
   - **Algorithm**: Processes events in arrival order. On-time events are immediately accumulated into their window. Late events are held in a correction buffer and applied after the stream ends.
   - **Baseline**: Identical arrival-order processing, but late events are silently dropped.
   - **Ground Truth**: Every event is assigned to `floor(event_time / window_size)` and summed per window.

3. **`test_project.py`**: Pytest test suite validating dataset generation, determinism, edge cases, and metric correctness.

4. **`main.py`**: CLI entry point (supplied by the host) that runs the experiment and writes results to a JSON file.

### Algorithm Details

**Watermark Advancement**: The watermark is conceptually advanced as `max_event_time_seen - allowed_lateness`. In this bounded simulation, the watermark is not explicitly tracked per-event; instead, lateness is determined by comparing `arrival_time` to `event_time + allowed_lateness`. This is equivalent for a bounded stream where all events are available.

**Window Assignment**: Each event is assigned to window `floor(event_time / window_size)`, where `window_size = 10.0` seconds.

**Late-Event Detection**: An event is late if `arrival_time > event_time + allowed_lateness`, where `allowed_lateness = 2.0` seconds.

**Correction Buffer**: Late events are stored in a list of `(window_index, value)` tuples. After all events have been processed, the buffer is drained and each late event's value is added to its window's sum.

**Baseline**: Late events are not buffered or applied. They are simply ignored during stream processing.

### Synthetic Dataset

The dataset is generated in-memory using NumPy's random number generator with a fixed seed for reproducibility.

**Assumptions**:
- Event times are uniformly distributed over a range of approximately 5 windows (50 seconds for the default 256 samples).
- Values are uniformly distributed in `[1.0, 10.0]`.
- Approximately 15% of events are late.
- On-time events have `arrival_time = event_time + jitter` where `jitter ∈ [0, 2.0]`.
- Late events have `arrival_time = event_time + jitter` where `jitter ∈ (2.1, 7.0]`.
- Events are sorted by `arrival_time` to simulate stream arrival order.
- No negative skew: `arrival_time >= event_time` always holds.

**Limitations**:
- The stream is bounded and in-memory; it does not simulate unbounded streaming, backpressure, or distributed state.
- Lateness is deterministic per seed; real-world streams may have variable and unpredictable lateness patterns.
- The correction buffer assumes all late events can be held in memory until stream completion, which may not scale to very large streams.
- The watermark is not explicitly advanced per-event; lateness is determined by a fixed threshold comparison.

## Algorithm vs. Baseline

| Aspect | Algorithm (Correction Buffer) | Baseline (Drop Late) |
|--------|-------------------------------|----------------------|
| Late events | Buffered and applied after stream ends | Silently dropped |
| On-time events | Immediately accumulated | Immediately accumulated |
| Window sums | Include late event values | Exclude late event values |
| Memory overhead | O(n_late) for the buffer | O(1) additional |
| Complexity | Slightly higher due to buffer management | Lower |

The algorithm is expected to recover late events that the baseline drops, resulting in higher `late_recovery_rate` and lower `window_mae`. However, this is a measured result, not a guaranteed invariant. The baseline may win in scenarios where late events are rare or where the correction buffer introduces errors (e.g., if the buffer is incorrectly managed).

## Metrics

All metrics are computed per experiment run.

| Metric | Direction | Description |
|--------|-----------|-------------|
| `window_mae` | Lower is better | Mean absolute error between computed window sums and ground truth sums across all windows. |
| `late_recovery_rate` | Higher is better, range [0,1] | Fraction of late events correctly included in their window. |
| `on_time_accuracy` | Higher is better, range [0,1] | Fraction of non-late events correctly included in their window. |

**No Leakage**: The watermark (and lateness determination) uses only previously observed arrival times. Ground truth is computed independently from event times and values, without access to the processing strategy.

## Reproducibility

The project is fully deterministic for a given seed and sample size. Repeated calls to `run_experiment(seed, n_samples)` produce byte-identical output.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Running the Experiment

```bash
python main.py --seed 42 --n-samples 256 --output results.json
```

This writes the experiment results to `results.json`, including `n_samples`, `metrics`, `baseline_metrics`, and an `explanation` string.

### Running Tests

```bash
python -m pytest -q
```

The test suite validates:
- Dataset generation (shape, dtype, seed sensitivity, invalid input handling).
- Determinism (repeated calls produce identical output).
- Edge cases (zero late events, minimum sample size).
- Metric correctness (ground truth comparison, late-recovery behavior).

## Python Version

The project requires Python 3.11, 3.12, or 3.13. It uses `from __future__ import annotations` for forward-reference compatibility and relies on NumPy 2.x and Pytest 8.x as specified in `requirements.txt`.

## Limitations

- **Bounded Stream**: The simulation is bounded and in-memory. It does not model unbounded streaming, backpressure, or distributed state management.
- **Simplified Watermark**: The watermark is not explicitly advanced per-event. Lateness is determined by a fixed threshold comparison, which is equivalent for a bounded stream but may not capture all nuances of real-time watermark advancement.
- **Memory Assumption**: The correction buffer assumes all late events can be held in memory. For very large streams, this may not be feasible.
- **Synthetic Data**: The dataset is synthetic with controlled lateness. Real-world streams may have more complex and variable lateness patterns.
- **No Production Readiness**: This project is an educational simulation, not a production-ready stream processing system. It does not handle failures, retries, or distributed coordination.

## Validation

Refer to `validation_report.json` and `example_results.json` for measured output from the experiment. The project does not claim superiority of the algorithm over the baseline without measured evidence. The baseline may win in certain scenarios, and the metrics reflect the actual performance of each strategy.

## Automated Test Limitations

The test suite validates correctness of the implementation but does not cover:
- Performance under high load or large streams.
- Edge cases in watermark advancement that may arise in unbounded streams.
- Interaction with external systems (e.g., message brokers, state backends).
- Failure modes (e.g., buffer overflow, memory exhaustion).

The tests are designed to catch bugs in the core logic, not to certify production readiness.

## Recorded automated validation

Host contract tests and project tests passed (20 tests, 0 skipped). Demo completed on Python 3.13.16. See `validation_report.json` and `example_results.json`. These checks validate the execution contract, not scientific novelty or every algorithmic claim.
