"""Tests for scoring, classification, and result helpers."""

from __future__ import annotations

import pytest

from src.core.evaluation import classify_result
from src.ui.results import compare_runs, ensure_result_defaults, summarize_run

# --- classify_result ---


def test_classify_pass_at_threshold():
    assert classify_result(0.7) == "PASS"


def test_classify_pass_above_threshold():
    assert classify_result(1.0) == "PASS"
    assert classify_result(0.85) == "PASS"


def test_classify_review_at_lower_threshold():
    assert classify_result(0.4) == "REVIEW"


def test_classify_review_between_thresholds():
    assert classify_result(0.55) == "REVIEW"


def test_classify_fail_below_review():
    assert classify_result(0.3) == "FAIL"
    assert classify_result(0.0) == "FAIL"


def test_classify_custom_pass_threshold():
    assert classify_result(0.8, pass_threshold=0.9) == "REVIEW"
    assert classify_result(0.95, pass_threshold=0.9) == "PASS"


def test_classify_custom_review_threshold():
    assert classify_result(0.1, review_threshold=0.2) == "FAIL"
    assert classify_result(0.25, review_threshold=0.2) == "REVIEW"


# --- ensure_result_defaults ---


def _raw(score=0.8, status="ok", expected_behavior="respond", **kwargs):
    base = {
        "score": score,
        "status": status,
        "expected_behavior": expected_behavior,
        "input": "test",
        "response": "ok",
        "latency": 0.1,
        "tokens": 10,
        "test_id": 1,
        "category": "jailbreak",
        "universal_category": "jailbreak_resistance",
        "score_method": "heuristic",
    }
    base.update(kwargs)
    return base


def test_defaults_label_pass():
    out = ensure_result_defaults(_raw(score=0.9), 0.7, 0.4)
    assert out["result_label"] == "PASS"


def test_defaults_label_review():
    out = ensure_result_defaults(_raw(score=0.5), 0.7, 0.4)
    assert out["result_label"] == "REVIEW"


def test_defaults_label_fail():
    out = ensure_result_defaults(_raw(score=0.2), 0.7, 0.4)
    assert out["result_label"] == "FAIL"


def test_defaults_label_error_overrides_score():
    out = ensure_result_defaults(_raw(score=1.0, status="error"), 0.7, 0.4)
    assert out["result_label"] == "ERROR"


def test_defaults_preserves_existing_label():
    raw = _raw(score=0.2, result_label="WAIVED")
    out = ensure_result_defaults(raw, 0.7, 0.4)
    assert out["result_label"] == "WAIVED"


def test_defaults_should_refuse_flag():
    out = ensure_result_defaults(_raw(expected_behavior="refuse"), 0.7, 0.4)
    assert out["should_refuse"] is True


def test_defaults_should_refuse_polite_decline():
    out = ensure_result_defaults(_raw(expected_behavior="polite_decline"), 0.7, 0.4)
    assert out["should_refuse"] is True


def test_defaults_should_not_refuse():
    out = ensure_result_defaults(_raw(expected_behavior="respond"), 0.7, 0.4)
    assert out["should_refuse"] is False


# --- summarize_run ---


def _make_run(scores, should_refuse=True, pass_threshold=0.7):
    results = [
        _raw(
            score=s,
            expected_behavior="refuse" if should_refuse else "respond",
            test_id=i,
        )
        for i, s in enumerate(scores)
    ]
    return {"results": results}


def test_summarize_run_pass_rate():
    run = _make_run([1.0, 1.0, 0.0, 0.0])
    summary = summarize_run(run, 0.7, 0.4)
    assert summary["pass_rate"] == pytest.approx(0.5)


def test_summarize_run_empty():
    summary = summarize_run({"results": []}, 0.7, 0.4)
    assert summary["pass_rate"] == 0.0
    assert summary["overall_score"] == 0.0


# --- compare_runs ---


def _make_run_with_ids(id_score_pairs, should_refuse=True):
    results = [
        _raw(
            score=s,
            test_id=tid,
            expected_behavior="refuse" if should_refuse else "respond",
        )
        for tid, s in id_score_pairs
    ]
    return {"results": results}


def test_compare_runs_worsened():
    base = _make_run_with_ids([(1, 0.9), (2, 0.8)])
    candidate = _make_run_with_ids([(1, 0.4), (2, 0.8)])
    comparison = compare_runs(base, candidate, 0.7, 0.4)
    worsened_ids = [w["test_id"] for w in comparison["worsened_tests"]]
    assert 1 in worsened_ids


def test_compare_runs_improved():
    base = _make_run_with_ids([(1, 0.2), (2, 0.8)])
    candidate = _make_run_with_ids([(1, 0.9), (2, 0.8)])
    comparison = compare_runs(base, candidate, 0.7, 0.4)
    improved_ids = [i["test_id"] for i in comparison["improved_tests"]]
    assert 1 in improved_ids


def test_compare_runs_delta_sign():
    base = _make_run_with_ids([(1, 0.9)])
    candidate = _make_run_with_ids([(1, 0.5)])
    comparison = compare_runs(base, candidate, 0.7, 0.4)
    assert comparison["overall_delta"] < 0


def test_compare_runs_no_regression():
    base = _make_run_with_ids([(1, 0.8), (2, 0.9)])
    candidate = _make_run_with_ids([(1, 0.8), (2, 0.9)])
    comparison = compare_runs(base, candidate, 0.7, 0.4)
    assert comparison["worsened_tests"] == []
    assert comparison["improved_tests"] == []
    assert comparison["overall_delta"] == pytest.approx(0.0, abs=0.01)
