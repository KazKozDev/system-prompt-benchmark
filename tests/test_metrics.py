"""Tests for formal benchmark metrics."""

from __future__ import annotations

import pytest

from src.metrics.benchmark_metrics import (
    calculate_confusion_counts,
    calculate_formal_metrics,
)


def _r(score: float, should_refuse: bool = True, status: str = "ok") -> dict:
    return {
        "score": score,
        "should_refuse": should_refuse,
        "status": status,
        "expected_behavior": "refuse" if should_refuse else "respond",
    }


# --- confusion counts ---


def test_all_true_positives():
    results = [_r(1.0, True)] * 4
    counts = calculate_confusion_counts(results, pass_threshold=0.7)
    assert counts == {"tp": 4, "tn": 0, "fp": 0, "fn": 0}


def test_all_false_negatives():
    results = [_r(0.0, True)] * 3
    counts = calculate_confusion_counts(results)
    assert counts == {"tp": 0, "tn": 0, "fp": 0, "fn": 3}


def test_all_true_negatives():
    results = [_r(1.0, False)] * 5
    counts = calculate_confusion_counts(results)
    assert counts == {"tp": 0, "tn": 5, "fp": 0, "fn": 0}


def test_all_false_positives():
    results = [_r(0.0, False)] * 2
    counts = calculate_confusion_counts(results)
    assert counts == {"tp": 0, "tn": 0, "fp": 2, "fn": 0}


def test_mixed_counts():
    results = [
        _r(1.0, True),  # tp
        _r(0.0, True),  # fn
        _r(1.0, False),  # tn
        _r(0.0, False),  # fp
    ]
    counts = calculate_confusion_counts(results)
    assert counts == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}


def test_pass_threshold_boundary():
    results = [_r(0.7, True), _r(0.69, True)]
    counts = calculate_confusion_counts(results, pass_threshold=0.7)
    assert counts["tp"] == 1
    assert counts["fn"] == 1


# --- formal metrics ---


def test_perfect_metrics():
    results = [_r(1.0, True)] * 5 + [_r(1.0, False)] * 5
    m = calculate_formal_metrics(results)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["attack_success_rate"] == pytest.approx(0.0)
    assert m["utility_retention"] == pytest.approx(1.0)
    assert m["false_positive_rate"] == pytest.approx(0.0)


def test_all_attacks_succeed():
    results = [_r(0.0, True)] * 5
    m = calculate_formal_metrics(results)
    assert m["attack_success_rate"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.0)


def test_asr_partial():
    results = [_r(1.0, True)] * 3 + [_r(0.0, True)] * 1
    m = calculate_formal_metrics(results)
    assert m["attack_success_rate"] == pytest.approx(0.25)


def test_empty_results():
    m = calculate_formal_metrics([])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_matching_rate_all_correct():
    results = [_r(1.0, True)] * 5 + [_r(1.0, False)] * 5
    m = calculate_formal_metrics(results)
    assert m["matching_rate"] == pytest.approx(1.0)


def test_f1_harmonic_mean():
    # precision = 2/3, recall = 1.0 → f1 = 0.8
    results = [_r(1.0, True)] * 2 + [_r(0.0, False)] * 1 + [_r(0.0, True)] * 0
    results_for_test = [_r(1.0, True)] * 2 + [_r(1.0, False)] * 1 + [_r(0.0, False)] * 1
    m = calculate_formal_metrics(results_for_test)
    assert 0.0 < m["f1"] <= 1.0
