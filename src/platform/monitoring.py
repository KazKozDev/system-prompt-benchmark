"""Prometheus metrics for the API service."""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response


def _lookup_registered_metric(name: str) -> Any | None:
    """Return an existing collector when the metric was already registered."""
    registry_map = getattr(REGISTRY, "_names_to_collectors", {})
    candidates = [name]
    if name.endswith("_total"):
        base_name = name[: -len("_total")]
        candidates.extend([base_name, f"{base_name}_created"])
    for candidate in candidates:
        collector = registry_map.get(candidate)
        if collector is not None:
            return collector
    return None


def _get_or_create_metric(metric_type, name: str, documentation: str, *args):
    """Create a metric once and reuse it on repeated imports."""
    existing = _lookup_registered_metric(name)
    if existing is not None:
        return existing
    return metric_type(name, documentation, *args)


API_REQUESTS_TOTAL = _get_or_create_metric(
    Counter,
    "spb_api_requests_total",
    "Count of API requests",
    ["endpoint", "method", "status"],
)
BENCHMARK_RUNS_TOTAL = _get_or_create_metric(
    Counter,
    "spb_benchmark_runs_total",
    "Count of benchmark runs created",
    ["status"],
)
WEBHOOKS_TOTAL = _get_or_create_metric(
    Counter,
    "spb_webhooks_total",
    "Count of webhook callback attempts",
    ["status"],
)
WEBHOOK_RETRIES_TOTAL = _get_or_create_metric(
    Counter,
    "spb_webhook_retries_total",
    "Count of webhook retry attempts",
    ["reason"],
)
WEBHOOK_SIGNED_TOTAL = _get_or_create_metric(
    Counter,
    "spb_webhook_signed_total",
    "Count of signed webhook deliveries",
    ["mode"],
)
WEBHOOK_DELIVERY_DURATION_SECONDS = _get_or_create_metric(
    Histogram,
    "spb_webhook_delivery_duration_seconds",
    "Duration of webhook callback delivery attempts",
    ["outcome"],
)
BENCHMARK_RUN_DURATION_SECONDS = _get_or_create_metric(
    Histogram,
    "spb_benchmark_run_duration_seconds",
    "Duration of benchmark runs",
)
JOB_QUEUE_DEPTH = _get_or_create_metric(
    Gauge,
    "spb_job_queue_depth",
    "Current depth of the in-process job queue",
)
JOB_STATUS_COUNT = _get_or_create_metric(
    Gauge,
    "spb_job_status_count",
    "Current persisted job counts by status",
    ["status"],
)
WORKER_HEARTBEATS_TOTAL = _get_or_create_metric(
    Counter,
    "spb_worker_heartbeats_total",
    "Count of worker lease renewals",
    ["worker_id"],
)
WORKER_POLL_TOTAL = _get_or_create_metric(
    Counter,
    "spb_worker_poll_total",
    "Count of external worker polling cycles",
    ["outcome"],
)
WORKER_ACTIVE = _get_or_create_metric(
    Gauge,
    "spb_worker_active",
    "Worker liveness and current lease information",
    ["worker_id"],
)
JOB_RETRIES_TOTAL = _get_or_create_metric(
    Counter,
    "spb_job_retries_total",
    "Count of job retries after failed attempts",
    ["reason"],
)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
