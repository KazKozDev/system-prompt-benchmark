"""Prometheus metrics for the API service."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response


API_REQUESTS_TOTAL = Counter(
    "spb_api_requests_total",
    "Count of API requests",
    ["endpoint", "method", "status"],
)
BENCHMARK_RUNS_TOTAL = Counter(
    "spb_benchmark_runs_total",
    "Count of benchmark runs created",
    ["status"],
)
WEBHOOKS_TOTAL = Counter(
    "spb_webhooks_total",
    "Count of webhook callback attempts",
    ["status"],
)
WEBHOOK_RETRIES_TOTAL = Counter(
    "spb_webhook_retries_total",
    "Count of webhook retry attempts",
    ["reason"],
)
WEBHOOK_SIGNED_TOTAL = Counter(
    "spb_webhook_signed_total",
    "Count of signed webhook deliveries",
    ["mode"],
)
WEBHOOK_DELIVERY_DURATION_SECONDS = Histogram(
    "spb_webhook_delivery_duration_seconds",
    "Duration of webhook callback delivery attempts",
    ["outcome"],
)
BENCHMARK_RUN_DURATION_SECONDS = Histogram(
    "spb_benchmark_run_duration_seconds",
    "Duration of benchmark runs",
)
JOB_QUEUE_DEPTH = Gauge(
    "spb_job_queue_depth",
    "Current depth of the in-process job queue",
)
JOB_STATUS_COUNT = Gauge(
    "spb_job_status_count",
    "Current persisted job counts by status",
    ["status"],
)
WORKER_HEARTBEATS_TOTAL = Counter(
    "spb_worker_heartbeats_total",
    "Count of worker lease renewals",
    ["worker_id"],
)
WORKER_POLL_TOTAL = Counter(
    "spb_worker_poll_total",
    "Count of external worker polling cycles",
    ["outcome"],
)
WORKER_ACTIVE = Gauge(
    "spb_worker_active",
    "Worker liveness and current lease information",
    ["worker_id"],
)
JOB_RETRIES_TOTAL = Counter(
    "spb_job_retries_total",
    "Count of job retries after failed attempts",
    ["reason"],
)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
