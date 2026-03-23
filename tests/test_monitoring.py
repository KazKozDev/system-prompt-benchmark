"""Tests for idempotent Prometheus metric registration."""

from __future__ import annotations

import importlib


def test_monitoring_module_can_be_reloaded_without_duplicate_metrics() -> None:
    from src.platform import monitoring

    reloaded = importlib.reload(monitoring)

    assert reloaded.API_REQUESTS_TOTAL is not None
    assert reloaded.BENCHMARK_RUNS_TOTAL is not None
    assert reloaded.JOB_QUEUE_DEPTH is not None
