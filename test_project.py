"""Tests for the watermark windowing late-event recovery project."""
from __future__ import annotations

import json

import numpy as np
import pytest

from data import make_dataset
from app import run_experiment, WINDOW_SIZE, ALLOWED_LATENESS


class TestMakeDataset:
    """Tests for the synthetic dataset generator."""

    def test_invalid_n_samples_raises(self) -> None:
        """n_samples < 32 must raise ValueError."""
        with pytest.raises(ValueError):
            make_dataset(seed=42, n_samples=31)

    def test_different_seeds_produce_different_x(self) -> None:
        """Different seeds must produce different X arrays."""
        X1, _ = make_dataset(seed=42, n_samples=256)
        X2, _ = make_dataset(seed=99, n_samples=256)
        assert not np.array_equal(X1, X2)

    def test_dataset_shape_and_range(self) -> None:
        """X has correct shape, y has correct shape, values in expected range."""
        X, y = make_dataset(seed=42, n_samples=128)
        assert X.shape == (128, 4)
        assert y.shape == (128,)
        # event_time >= 0
        assert np.all(X[:, 0] >= 0.0)
        # arrival_time >= event_time
        assert np.all(X[:, 1] >= X[:, 0])
        # values in [1, 10]
        assert np.all(X[:, 2] >= 1.0)
        assert np.all(X[:, 2] <= 10.0)
        # is_late in {0, 1}
        assert set(np.unique(X[:, 3])).issubset({0.0, 1.0})
        # y = floor(event_time / WINDOW_SIZE)
        expected_y = (X[:, 0] // WINDOW_SIZE).astype(np.int64)
        np.testing.assert_array_equal(y, expected_y)

    def test_late_fraction_approximately_15_percent(self) -> None:
        """The fraction of late events should be close to 15%."""
        X, _ = make_dataset(seed=42, n_samples=256)
        late_fraction = np.sum(X[:, 3]) / len(X)
        assert 0.10 <= late_fraction <= 0.20


class TestRunExperiment:
    """Tests for the main experiment function."""

    def test_deterministic_output(self) -> None:
        """Same inputs must give exactly the same output."""
        result1 = run_experiment(seed=42, n_samples=256)
        result2 = run_experiment(seed=42, n_samples=256)
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_different_n_samples_respected(self) -> None:
        """Different n_samples must be reflected in the output."""
        result_small = run_experiment(seed=42, n_samples=64)
        result_large = run_experiment(seed=42, n_samples=256)
        assert result_small["n_samples"] == 64
        assert result_large["n_samples"] == 256

    def test_output_structure(self) -> None:
        """Output must contain required keys with correct types."""
        result = run_experiment(seed=42, n_samples=256)
        assert isinstance(result["n_samples"], int)
        assert isinstance(result["metrics"], dict)
        assert isinstance(result["baseline_metrics"], dict)
        assert len(result["metrics"]) > 0
        assert len(result["baseline_metrics"]) > 0
        for key in result["metrics"]:
            assert isinstance(result["metrics"][key], (int, float))
            assert np.isfinite(result["metrics"][key])
        for key in result["baseline_metrics"]:
            assert isinstance(result["baseline_metrics"][key], (int, float))
            assert np.isfinite(result["baseline_metrics"][key])

    def test_algorithm_recovers_more_late_events_than_baseline(self) -> None:
        """Algorithm's late_recovery_rate must be strictly greater than baseline's."""
        result = run_experiment(seed=42, n_samples=256)
        algo_lrr = result["metrics"]["late_recovery_rate"]
        base_lrr = result["baseline_metrics"]["late_recovery_rate"]
        assert algo_lrr > base_lrr, (
            f"Algorithm late_recovery_rate ({algo_lrr}) should be > "
            f"baseline ({base_lrr})"
        )

    def test_algorithm_window_mae_lower_or_equal(self) -> None:
        """Algorithm's window_mae should be <= baseline's (recovers more events)."""
        result = run_experiment(seed=42, n_samples=256)
        algo_mae = result["metrics"]["window_mae"]
        base_mae = result["baseline_metrics"]["window_mae"]
        assert algo_mae <= base_mae + 1e-12, (
            f"Algorithm window_mae ({algo_mae}) should be <= "
            f"baseline ({base_mae})"
        )

    def test_on_time_accuracy_is_one(self) -> None:
        """On-time events are always correctly placed, so accuracy is 1.0."""
        result = run_experiment(seed=42, n_samples=256)
        assert result["metrics"]["on_time_accuracy"] == 1.0
        assert result["baseline_metrics"]["on_time_accuracy"] == 1.0

    def test_no_late_events_perfect_scores(self) -> None:
        """When no events are late, both strategies should achieve perfect scores.

        We verify this by checking that if we set is_late to all zeros, the
        computed window sums match ground truth exactly.
        """
        X, y = make_dataset(seed=42, n_samples=256)
        # Override: make all events on-time
        X_no_late = X.copy()
        X_no_late[:, 3] = 0.0

        event_times = X_no_late[:, 0]
        values = X_no_late[:, 2]
        is_late = X_no_late[:, 3]

        # Ground truth
        window_indices = (event_times // WINDOW_SIZE).astype(int)
        true_sums: dict[int, float] = {}
        for idx, val in zip(window_indices, values):
            true_sums[int(idx)] = true_sums.get(int(idx), 0.0) + val

        # Process with no late events (both strategies should be identical)
        from app import _process_stream

        arrival_times = X_no_late[:, 1]
        computed, n_included, n_total = _process_stream(
            event_times, arrival_times, values, is_late, recover_late=True
        )
        assert n_total == 0
        # All windows should match exactly
        for w, true_val in true_sums.items():
            assert abs(computed.get(w, 0.0) - true_val) < 1e-10, (
                f"Window {w}: computed {computed.get(w, 0.0)} != true {true_val}"
            )

    def test_late_recovery_rate_in_valid_range(self) -> None:
        """late_recovery_rate must be in [0, 1]."""
        result = run_experiment(seed=42, n_samples=256)
        assert 0.0 <= result["metrics"]["late_recovery_rate"] <= 1.0
        assert 0.0 <= result["baseline_metrics"]["late_recovery_rate"] <= 1.0

    def test_window_mae_non_negative(self) -> None:
        """window_mae must be non-negative."""
        result = run_experiment(seed=42, n_samples=256)
        assert result["metrics"]["window_mae"] >= 0.0
        assert result["baseline_metrics"]["window_mae"] >= 0.0

    def test_baseline_drops_all_late_events(self) -> None:
        """Baseline should recover 0 late events (late_recovery_rate == 0 when late events exist)."""
        result = run_experiment(seed=42, n_samples=256)
        # With ~15% late events, baseline drops all of them
        assert result["baseline_metrics"]["late_recovery_rate"] == 0.0

    def test_algorithm_recovers_all_late_events(self) -> None:
        """Algorithm should recover all late events (late_recovery_rate == 1 when late events exist)."""
        result = run_experiment(seed=42, n_samples=256)
        assert result["metrics"]["late_recovery_rate"] == 1.0

    def test_json_serializable(self) -> None:
        """The result must be JSON-serializable."""
        result = run_experiment(seed=42, n_samples=256)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized["n_samples"] == result["n_samples"]
        assert deserialized["metrics"] == result["metrics"]
        assert deserialized["baseline_metrics"] == result["baseline_metrics"]
