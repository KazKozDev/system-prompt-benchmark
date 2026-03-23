"""REST API for benchmark execution and dataset validation."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.config import benchmark_config_from_dict
from src.core.run_universal_benchmark import load_result_file
from src.datasets import load_dataset_manifest, validate_dataset_file
from src.platform.api_auth import AuthContext, require_role
from src.platform.job_store import JobStore
from src.platform.monitoring import (
    API_REQUESTS_TOTAL,
    BENCHMARK_RUNS_TOTAL,
    metrics_response,
)
from src.platform.worker_backend import WORKER_BACKEND, WorkerBackend


def _get_or_create_job_store() -> JobStore:
    existing = globals().get("JOB_STORE")
    if existing is not None:
        return existing
    return JobStore(os.getenv("SPB_API_DB", "results/api.sqlite3"))


def _get_or_create_worker_backend(job_store: JobStore) -> WorkerBackend:
    existing = globals().get("WORKERS")
    if existing is not None:
        return existing
    return WorkerBackend(job_store)


JOB_STORE = _get_or_create_job_store()
WORKERS = _get_or_create_worker_backend(JOB_STORE)


class RunRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    webhook_url: str | None = None


class DatasetValidateRequest(BaseModel):
    path: str
    format: str | None = None


def _upsert_job(job: dict[str, Any]) -> None:
    WORKERS.upsert_job(job)


def _get_job(job_id: str) -> dict[str, Any] | None:
    return WORKERS.get_job(job_id)


def _refresh_job_status_metrics() -> None:
    WORKERS.refresh_metrics()


def _start_workers() -> None:
    if WORKER_BACKEND == "inprocess":
        WORKERS.start_inprocess_workers()
    else:
        _refresh_job_status_metrics()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _start_workers()
    yield


app = FastAPI(
    title="system-prompt-benchmark API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    API_REQUESTS_TOTAL.labels(endpoint="/health", method="GET", status="200").inc()
    return {
        "status": "ok",
        "worker_backend": WORKER_BACKEND,
        "workers": len(WORKERS.worker_threads),
        "queue_depth": JOB_STORE.queued_count(),
        "worker_status": WORKERS.worker_status(),
    }


@app.get("/workers/status")
def workers_status(_: Annotated[AuthContext, Depends(require_role("admin"))]) -> dict[str, Any]:
    API_REQUESTS_TOTAL.labels(endpoint="/workers/status", method="GET", status="200").inc()
    _refresh_job_status_metrics()
    return WORKERS.worker_status()


@app.get("/workers/redis/pending")
def workers_redis_pending(
    _: Annotated[AuthContext, Depends(require_role("admin"))],
    limit: int = Query(default=100, ge=1, le=500),
    consumer: str | None = Query(default=None),
) -> dict[str, Any]:
    API_REQUESTS_TOTAL.labels(endpoint="/workers/redis/pending", method="GET", status="200").inc()
    entries = WORKERS.list_redis_pending_entries(limit=limit, consumer=consumer)
    return {"entries": entries, "count": len(entries)}


@app.post("/workers/redis/pending/{job_id}/replay")
def workers_redis_pending_replay(
    job_id: str,
    _: Annotated[AuthContext, Depends(require_role("admin"))],
) -> dict[str, Any]:
    replayed = WORKERS.replay_redis_pending_job(job_id)
    if not replayed:
        API_REQUESTS_TOTAL.labels(endpoint="/workers/redis/pending/{job_id}/replay", method="POST", status="404").inc()
        raise HTTPException(status_code=404, detail=f"Redis pending entry not found for job: {job_id}")
    API_REQUESTS_TOTAL.labels(endpoint="/workers/redis/pending/{job_id}/replay", method="POST", status="200").inc()
    return replayed


@app.post("/runs/{job_id}/replay")
def replay_run(
    job_id: str,
    _: Annotated[AuthContext, Depends(require_role("admin"))],
) -> dict[str, Any]:
    replayed = WORKERS.replay_dead_letter_job(job_id)
    if not replayed:
        API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}/replay", method="POST", status="404").inc()
        raise HTTPException(status_code=404, detail=f"Dead-letter job not found or not replayable: {job_id}")
    API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}/replay", method="POST", status="200").inc()
    return {
        "job_id": job_id,
        "status": replayed["status"],
        "status_url": f"/runs/{job_id}",
        "result_url": f"/runs/{job_id}/result",
    }


@app.get("/runs/webhook-failures")
def list_webhook_failures(
    _: Annotated[AuthContext, Depends(require_role("admin"))],
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    API_REQUESTS_TOTAL.labels(endpoint="/runs/webhook-failures", method="GET", status="200").inc()
    jobs = WORKERS.list_webhook_failures(limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@app.post("/runs/{job_id}/webhook/replay")
def replay_webhook(
    job_id: str,
    _: Annotated[AuthContext, Depends(require_role("admin"))],
) -> dict[str, Any]:
    replayed = WORKERS.replay_webhook_delivery(job_id)
    if not replayed:
        API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}/webhook/replay", method="POST", status="404").inc()
        raise HTTPException(status_code=404, detail=f"Webhook delivery not replayable for job: {job_id}")
    API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}/webhook/replay", method="POST", status="200").inc()
    return {
        "job_id": job_id,
        "webhook_status": replayed.get("webhook_status"),
        "webhook_attempts": replayed.get("webhook_attempts"),
        "webhook_last_status_code": replayed.get("webhook_last_status_code"),
        "webhook_last_error": replayed.get("webhook_last_error"),
        "webhook_delivery_id": replayed.get("webhook_delivery_id"),
    }


@app.get("/metrics")
def metrics(_: Annotated[AuthContext, Depends(require_role("admin"))]) -> Any:
    API_REQUESTS_TOTAL.labels(endpoint="/metrics", method="GET", status="200").inc()
    _refresh_job_status_metrics()
    return metrics_response()


@app.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def create_run(
    request: RunRequest,
    auth: Annotated[AuthContext, Depends(require_role("runner"))],
) -> dict[str, Any]:
    try:
        config = benchmark_config_from_dict(request.config or {})
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "status": "queued",
            "owner": auth.principal,
            "created_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
            "output_path": None,
            "result_id": None,
            "summary": None,
            "error": None,
            "config": config.to_dict(),
            "webhook_url": request.webhook_url,
            "webhook_status": None,
            "webhook_attempted_at": None,
            "webhook_attempts": 0,
            "webhook_last_error": None,
            "webhook_last_status_code": None,
            "webhook_delivery_id": None,
            "webhook_delivered_at": None,
            "worker_backend": WORKER_BACKEND,
            "locked_by": None,
            "lease_until": None,
            "started_at": None,
            "attempts": 0,
        }
        _upsert_job(job)
        WORKERS.enqueue_job(job_id)
        _refresh_job_status_metrics()
        API_REQUESTS_TOTAL.labels(endpoint="/runs", method="POST", status="202").inc()
        BENCHMARK_RUNS_TOTAL.labels(status="queued").inc()
        return {
            "job_id": job_id,
            "status": "queued",
            "owner": auth.principal,
            "status_url": f"/runs/{job_id}",
            "result_url": f"/runs/{job_id}/result",
        }
    except Exception as exc:
        API_REQUESTS_TOTAL.labels(endpoint="/runs", method="POST", status="400").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/runs")
def list_runs(
    auth: Annotated[AuthContext, Depends(require_role("viewer"))],
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    jobs = JOB_STORE.list_jobs(limit=limit)
    if auth.role != "admin":
        jobs = [job for job in jobs if job.get("owner") == auth.principal]
    API_REQUESTS_TOTAL.labels(endpoint="/runs", method="GET", status="200").inc()
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/runs/{job_id}")
def get_run(job_id: str, auth: Annotated[AuthContext, Depends(require_role("viewer"))]) -> dict[str, Any]:
    job = _get_job(job_id)
    if not job:
        API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}", method="GET", status="404").inc()
        raise HTTPException(status_code=404, detail=f"Run job not found: {job_id}")
    if auth.role != "admin" and job.get("owner") != auth.principal:
        API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}", method="GET", status="403").inc()
        raise HTTPException(status_code=403, detail="Forbidden")
    API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}", method="GET", status="200").inc()
    return job


@app.get("/runs/{job_id}/result")
def get_run_result(job_id: str, auth: Annotated[AuthContext, Depends(require_role("viewer"))]) -> dict[str, Any]:
    job = _get_job(job_id)
    if not job:
        API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}/result", method="GET", status="404").inc()
        raise HTTPException(status_code=404, detail=f"Run job not found: {job_id}")
    if auth.role != "admin" and job.get("owner") != auth.principal:
        API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}/result", method="GET", status="403").inc()
        raise HTTPException(status_code=403, detail="Forbidden")
    if job.get("status") in {"failed", "dead_letter"}:
        API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}/result", method="GET", status="409").inc()
        raise HTTPException(status_code=409, detail=job.get("error", "run failed"))
    if job.get("status") != "completed" or not job.get("output_path"):
        API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}/result", method="GET", status="202").inc()
        raise HTTPException(status_code=202, detail="run is not completed yet")
    API_REQUESTS_TOTAL.labels(endpoint="/runs/{job_id}/result", method="GET", status="200").inc()
    return load_result_file(job["output_path"])


@app.get("/results/{result_id}")
def get_result(result_id: str, _: Annotated[AuthContext, Depends(require_role("viewer"))]) -> dict[str, Any]:
    candidate_paths = [
        Path("results") / f"{result_id}.json",
        Path(result_id),
    ]
    for path in candidate_paths:
        if path.exists():
            API_REQUESTS_TOTAL.labels(endpoint="/results/{result_id}", method="GET", status="200").inc()
            return load_result_file(str(path))
    API_REQUESTS_TOTAL.labels(endpoint="/results/{result_id}", method="GET", status="404").inc()
    raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")


@app.post("/datasets/validate")
def validate_dataset(
    request: DatasetValidateRequest,
    _: Annotated[AuthContext, Depends(require_role("viewer"))],
) -> dict[str, Any]:
    try:
        rows, issues = validate_dataset_file(request.path, file_format=request.format)
        metadata = load_dataset_manifest(request.path)
        API_REQUESTS_TOTAL.labels(endpoint="/datasets/validate", method="POST", status="200").inc()
        return {
            "valid": not issues,
            "rows_loaded": len(rows),
            "issues": issues,
            "metadata": metadata,
        }
    except Exception as exc:
        API_REQUESTS_TOTAL.labels(endpoint="/datasets/validate", method="POST", status="400").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
