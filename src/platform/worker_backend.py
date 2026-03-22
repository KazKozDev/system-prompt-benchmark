"""Shared worker backend logic for API and external worker processes."""

from __future__ import annotations

import os
import queue
import threading
import time
import uuid
import hmac
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
try:
    import redis
except ModuleNotFoundError:
    redis = None

from src.config import benchmark_config_from_dict
from src.core.run_universal_benchmark import run_benchmark_from_config
from src.platform.job_store import JobStore
from src.platform.monitoring import (
    BENCHMARK_RUN_DURATION_SECONDS,
    BENCHMARK_RUNS_TOTAL,
    JOB_RETRIES_TOTAL,
    JOB_QUEUE_DEPTH,
    JOB_STATUS_COUNT,
    WEBHOOK_DELIVERY_DURATION_SECONDS,
    WEBHOOK_RETRIES_TOTAL,
    WEBHOOK_SIGNED_TOTAL,
    WEBHOOKS_TOTAL,
    WORKER_ACTIVE,
    WORKER_HEARTBEATS_TOTAL,
    WORKER_POLL_TOTAL,
)


JOB_DIR = Path("results/api-jobs")
JOB_DIR.mkdir(parents=True, exist_ok=True)
WEBHOOK_TIMEOUT_SECONDS = float(os.getenv("SPB_WEBHOOK_TIMEOUT_SECONDS", "10"))
WEBHOOK_MAX_ATTEMPTS = max(1, int(os.getenv("SPB_WEBHOOK_MAX_ATTEMPTS", "3")))
WEBHOOK_RETRY_BACKOFF_SECONDS = float(os.getenv("SPB_WEBHOOK_RETRY_BACKOFF_SECONDS", "5"))
WEBHOOK_SIGNING_SECRET = os.getenv("SPB_WEBHOOK_SIGNING_SECRET", "")
WORKER_BACKEND = os.getenv("SPB_WORKER_BACKEND", "inprocess").strip().lower()
WORKER_COUNT = max(1, int(os.getenv("SPB_API_WORKERS", "2")))
WORKER_POLL_INTERVAL_SECONDS = float(os.getenv("SPB_WORKER_POLL_INTERVAL_SECONDS", "2"))
JOB_LEASE_SECONDS = float(os.getenv("SPB_WORKER_JOB_LEASE_SECONDS", "7200"))
MAX_JOB_ATTEMPTS = max(1, int(os.getenv("SPB_WORKER_MAX_ATTEMPTS", "3")))
WORKER_HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("SPB_WORKER_HEARTBEAT_INTERVAL_SECONDS", "30"))
RETRY_BACKOFF_SECONDS = float(os.getenv("SPB_WORKER_RETRY_BACKOFF_SECONDS", "15"))
REDIS_URL = os.getenv("SPB_REDIS_URL", "redis://localhost:6379/0")
REDIS_QUEUE_KEY = os.getenv("SPB_REDIS_QUEUE_KEY", "spb:jobs:queue")
REDIS_STREAM_KEY = os.getenv("SPB_REDIS_STREAM_KEY", "spb:jobs:stream")
REDIS_CONSUMER_GROUP = os.getenv("SPB_REDIS_CONSUMER_GROUP", "spb-workers")
REDIS_STREAM_BLOCK_MILLISECONDS = max(100, int(float(os.getenv("SPB_REDIS_STREAM_BLOCK_MILLISECONDS", "2000"))))
REDIS_STREAM_CLAIM_IDLE_MILLISECONDS = max(1000, int(float(os.getenv("SPB_REDIS_STREAM_CLAIM_IDLE_MILLISECONDS", "60000"))))


class WorkerBackend:
    def __init__(self, job_store: JobStore) -> None:
        self.job_store = job_store
        self.job_queue: queue.Queue[str] = queue.Queue()
        self.worker_threads: list[threading.Thread] = []
        self.redis_client = None
        if WORKER_BACKEND == "redis":
            if redis is None:
                raise RuntimeError("redis package is required for SPB_WORKER_BACKEND=redis")
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

    @staticmethod
    def _job_json_path(job_id: str) -> Path:
        return JOB_DIR / f"{job_id}.json"

    def _write_job_file(self, job: dict[str, Any]) -> None:
        with self._job_json_path(job["job_id"]).open("w", encoding="utf-8") as handle:
            import json

            json.dump(job, handle, indent=2, ensure_ascii=False)

    def upsert_job(self, job: dict[str, Any]) -> None:
        self.job_store.upsert_job(job)
        self._write_job_file(job)
        self.refresh_metrics()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.job_store.get_job(job_id)

    def refresh_metrics(self) -> None:
        counts = self.job_store.status_counts()
        known_statuses = {"queued", "running", "completed", "failed", "dead_letter"}
        for status_name in known_statuses | set(counts):
            JOB_STATUS_COUNT.labels(status=status_name).set(counts.get(status_name, 0))
        JOB_QUEUE_DEPTH.set(self.job_store.queued_count())

    def _redis_ping(self) -> bool:
        if not self.redis_client:
            return False
        try:
            return bool(self.redis_client.ping())
        except Exception:
            return False

    def _ensure_redis_group(self) -> None:
        if not self.redis_client:
            return
        try:
            self.redis_client.xgroup_create(
                name=REDIS_STREAM_KEY,
                groupname=REDIS_CONSUMER_GROUP,
                id="0-0",
                mkstream=True,
            )
        except Exception as exc:
            message = str(exc)
            if "BUSYGROUP" not in message:
                raise

    def _redis_pending_summary(self) -> dict[str, Any] | None:
        if not self.redis_client or WORKER_BACKEND != "redis":
            return None
        try:
            summary = self.redis_client.xpending(REDIS_STREAM_KEY, REDIS_CONSUMER_GROUP)
        except Exception:
            return None
        if isinstance(summary, dict):
            return summary
        if isinstance(summary, (list, tuple)) and len(summary) >= 4:
            return {
                "pending": summary[0],
                "min": summary[1],
                "max": summary[2],
                "consumers": summary[3],
            }
        return None

    def list_redis_pending_entries(self, limit: int = 100, consumer: str | None = None) -> list[dict[str, Any]]:
        if not self.redis_client or WORKER_BACKEND != "redis":
            return []
        self._ensure_redis_group()
        pending_entries: list[Any] = []
        try:
            pending_entries = self.redis_client.xpending_range(
                REDIS_STREAM_KEY,
                REDIS_CONSUMER_GROUP,
                min="-",
                max="+",
                count=limit,
                consumername=consumer,
            )
        except TypeError:
            try:
                pending_entries = self.redis_client.xpending_range(
                    REDIS_STREAM_KEY,
                    REDIS_CONSUMER_GROUP,
                    "-",
                    "+",
                    limit,
                    consumername=consumer,
                )
            except Exception:
                return []
        except Exception:
            return []

        normalized: list[dict[str, Any]] = []
        for entry in pending_entries or []:
            message_id = str(entry.get("message_id") or entry.get("messageId") or entry.get("id") or "")
            if not message_id:
                continue
            payload = self._redis_fetch_stream_payload(message_id)
            normalized.append(
                {
                    "message_id": message_id,
                    "consumer": entry.get("consumer"),
                    "idle_ms": entry.get("idle") or entry.get("idle_ms") or entry.get("time_since_delivered"),
                    "deliveries": entry.get("times_delivered") or entry.get("times_delivered_count") or entry.get("delivery_count"),
                    "job_id": payload.get("job_id"),
                    "payload": payload,
                }
            )
        return normalized

    def _redis_fetch_stream_payload(self, message_id: str) -> dict[str, Any]:
        if not self.redis_client:
            return {}
        try:
            rows = self.redis_client.xrange(REDIS_STREAM_KEY, min=message_id, max=message_id, count=1)
        except TypeError:
            try:
                rows = self.redis_client.xrange(REDIS_STREAM_KEY, message_id, message_id, 1)
            except Exception:
                return {}
        except Exception:
            return {}
        if not rows:
            return {}
        _, payload = rows[0]
        return dict(payload or {})

    def _redis_ack_and_delete(self, message_id: str) -> None:
        if not self.redis_client:
            return
        self.redis_client.xack(REDIS_STREAM_KEY, REDIS_CONSUMER_GROUP, message_id)
        try:
            self.redis_client.xdel(REDIS_STREAM_KEY, message_id)
        except Exception:
            pass

    def _redis_requeue_if_needed(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job and job.get("status") == "queued":
            self.enqueue_job(job_id)

    def replay_redis_pending_job(self, job_id: str) -> dict[str, Any] | None:
        if not self.redis_client or WORKER_BACKEND != "redis":
            return None
        pending = self.list_redis_pending_entries(limit=500)
        match = next((entry for entry in pending if entry.get("job_id") == job_id), None)
        if not match:
            return None
        self._redis_ack_and_delete(str(match["message_id"]))
        self.enqueue_job(job_id)
        return {
            "job_id": job_id,
            "replayed": True,
            "old_message_id": match["message_id"],
            "consumer": match.get("consumer"),
        }

    def invoke_webhook(self, job: dict[str, Any]) -> None:
        webhook_url = job.get("webhook_url")
        if not webhook_url:
            return
        payload = {
            "job_id": job["job_id"],
            "status": job["status"],
            "owner": job.get("owner"),
            "completed_at": job.get("completed_at"),
            "result_id": job.get("result_id"),
            "summary": job.get("summary"),
            "error": job.get("error"),
        }
        delivery_id = uuid.uuid4().hex
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "X-SPB-Delivery-ID": delivery_id,
            "X-SPB-Timestamp": timestamp,
        }
        if WEBHOOK_SIGNING_SECRET:
            signature = hmac.new(
                WEBHOOK_SIGNING_SECRET.encode("utf-8"),
                f"{timestamp}.".encode("utf-8") + body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-SPB-Signature"] = f"sha256={signature}"
            WEBHOOK_SIGNED_TOTAL.labels(mode="hmac-sha256").inc()
        else:
            WEBHOOK_SIGNED_TOTAL.labels(mode="unsigned").inc()
        job["webhook_delivery_id"] = delivery_id
        job["webhook_attempts"] = 0
        job["webhook_last_error"] = None
        job["webhook_last_status_code"] = None
        job["webhook_delivered_at"] = None
        last_error: str | None = None
        for attempt in range(1, WEBHOOK_MAX_ATTEMPTS + 1):
            job["webhook_attempts"] = attempt
            job["webhook_attempted_at"] = datetime.now(UTC).isoformat()
            try:
                with WEBHOOK_DELIVERY_DURATION_SECONDS.labels(outcome="attempt").time():
                    response = requests.post(
                        webhook_url,
                        data=body,
                        headers=headers,
                        timeout=WEBHOOK_TIMEOUT_SECONDS,
                    )
                job["webhook_last_status_code"] = response.status_code
                if response.ok:
                    job["webhook_status"] = f"delivered:{response.status_code}"
                    job["webhook_last_error"] = None
                    job["webhook_delivered_at"] = datetime.now(UTC).isoformat()
                    WEBHOOKS_TOTAL.labels(status="success").inc()
                    break
                last_error = f"http_{response.status_code}"
                job["webhook_last_error"] = last_error
                if not self._should_retry_webhook_status(response.status_code, attempt):
                    job["webhook_status"] = f"failed:{last_error}"
                    WEBHOOKS_TOTAL.labels(status="failure").inc()
                    break
                WEBHOOK_RETRIES_TOTAL.labels(reason=last_error).inc()
                time.sleep(self._webhook_retry_delay_seconds(attempt))
            except Exception as exc:
                last_error = str(exc)
                job["webhook_last_error"] = last_error
                if attempt >= WEBHOOK_MAX_ATTEMPTS:
                    job["webhook_status"] = f"failed:{last_error}"
                    WEBHOOKS_TOTAL.labels(status="failure").inc()
                    break
                WEBHOOK_RETRIES_TOTAL.labels(reason="transport_error").inc()
                time.sleep(self._webhook_retry_delay_seconds(attempt))
        else:
            job["webhook_status"] = f"failed:{last_error or 'unknown'}"
            WEBHOOKS_TOTAL.labels(status="failure").inc()
        self.upsert_job(job)

    def replay_webhook_delivery(self, job_id: str) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        if not job or not job.get("webhook_url"):
            return None
        if job.get("status") not in {"completed", "failed", "dead_letter"}:
            return None
        self.invoke_webhook(job)
        return self.get_job(job_id)

    def list_webhook_failures(self, limit: int = 100) -> list[dict[str, Any]]:
        failed: list[dict[str, Any]] = []
        for job in self.job_store.list_jobs(limit=limit):
            webhook_status = str(job.get("webhook_status") or "")
            if job.get("webhook_url") and webhook_status.startswith("failed:"):
                failed.append(job)
        return failed

    def run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        worker_id = job.get("locked_by")
        heartbeat_stop = threading.Event()
        heartbeat_thread = None
        if worker_id:
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(job_id, worker_id, heartbeat_stop),
                daemon=True,
                name=f"heartbeat-{worker_id}",
            )
            heartbeat_thread.start()
        try:
            config = benchmark_config_from_dict(job.get("config") or {})
            with BENCHMARK_RUN_DURATION_SECONDS.time():
                benchmark, output_path = run_benchmark_from_config(config)
            summary = benchmark.build_summary(fail_threshold=config.fail_threshold)
            job.update(
                {
                    "status": "completed",
                    "completed_at": datetime.now(UTC).isoformat(),
                    "output_path": str(output_path),
                    "result_id": Path(output_path).stem,
                    "summary": summary,
                    "error": None,
                    "lease_until": None,
                }
            )
            job = self.job_store.release_job(
                job_id,
                worker_id=worker_id or "",
                status="completed",
                completed_at=job["completed_at"],
                retryable=False,
                max_attempts=MAX_JOB_ATTEMPTS,
            ) or job
            BENCHMARK_RUNS_TOTAL.labels(status="completed").inc()
        except Exception as exc:
            completed_at = datetime.now(UTC).isoformat()
            released = self.job_store.release_job(
                job_id,
                worker_id=worker_id or "",
                status="failed",
                error=str(exc),
                completed_at=completed_at,
                retryable=True,
                max_attempts=MAX_JOB_ATTEMPTS,
                retry_delay_seconds=self._retry_delay_seconds(int((job.get("attempts") or 1))),
            )
            job = released or job
            job["completed_at"] = job.get("completed_at") or completed_at
            job["error"] = str(exc)
            if job.get("status") == "queued":
                JOB_RETRIES_TOTAL.labels(reason="job_failure").inc()
            elif job.get("status") == "dead_letter":
                BENCHMARK_RUNS_TOTAL.labels(status="dead_letter").inc()
            else:
                BENCHMARK_RUNS_TOTAL.labels(status="failed").inc()
        finally:
            if heartbeat_thread:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=5)
        self.upsert_job(job)
        if job.get("status") in {"completed", "failed", "dead_letter"}:
            self.invoke_webhook(job)
        elif job.get("status") == "queued" and WORKER_BACKEND == "redis":
            self._redis_requeue_if_needed(job_id)

    def _heartbeat_loop(self, job_id: str, worker_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(WORKER_HEARTBEAT_INTERVAL_SECONDS):
            renewed = self.job_store.renew_lease(job_id, worker_id, lease_seconds=JOB_LEASE_SECONDS)
            if not renewed:
                return
            WORKER_HEARTBEATS_TOTAL.labels(worker_id=worker_id).inc()
            WORKER_ACTIVE.labels(worker_id=worker_id).set(time.time())
            self.refresh_metrics()

    def _inprocess_worker_loop(self) -> None:
        while True:
            job_id = self.job_queue.get()
            self.refresh_metrics()
            try:
                job = self.get_job(job_id)
                if job and job.get("status") in {"queued", "running"}:
                    job["status"] = "running"
                    if not job.get("started_at"):
                        job["started_at"] = datetime.now(UTC).isoformat()
                    self.upsert_job(job)
                    self.run_job(job_id)
            finally:
                self.job_queue.task_done()
                self.refresh_metrics()

    def start_inprocess_workers(self) -> None:
        if self.worker_threads:
            return
        for index in range(WORKER_COUNT):
            worker = threading.Thread(target=self._inprocess_worker_loop, name=f"spb-api-worker-{index + 1}", daemon=True)
            worker.start()
            self.worker_threads.append(worker)
        for job in self.job_store.list_pending_jobs():
            self.job_queue.put(job["job_id"])
        self.refresh_metrics()

    def enqueue_job(self, job_id: str) -> None:
        if WORKER_BACKEND == "inprocess":
            self.job_queue.put(job_id)
        elif WORKER_BACKEND == "redis":
            if not self.redis_client:
                raise RuntimeError("Redis backend is not initialized")
            self._ensure_redis_group()
            self.redis_client.xadd(
                REDIS_STREAM_KEY,
                {
                    "job_id": job_id,
                    "queued_at": datetime.now(UTC).isoformat(),
                },
            )
        self.refresh_metrics()

    def run_external_worker_loop(self, once: bool = False, worker_id: str | None = None) -> int:
        effective_worker_id = worker_id or f"spb-worker-{uuid.uuid4().hex[:8]}"
        processed = 0
        while True:
            WORKER_ACTIVE.labels(worker_id=effective_worker_id).set(time.time())
            job = self.job_store.claim_next_job(effective_worker_id, lease_seconds=JOB_LEASE_SECONDS)
            self.refresh_metrics()
            if not job:
                WORKER_POLL_TOTAL.labels(outcome="empty").inc()
                if once:
                    return processed
                time.sleep(WORKER_POLL_INTERVAL_SECONDS)
                continue
            WORKER_POLL_TOTAL.labels(outcome="claimed").inc()
            self.run_job(job["job_id"])
            processed += 1
            if once:
                return processed

    def run_redis_worker_loop(self, once: bool = False, worker_id: str | None = None) -> int:
        if not self.redis_client:
            raise RuntimeError("Redis backend is not initialized")
        self._ensure_redis_group()
        effective_worker_id = worker_id or f"spb-redis-worker-{uuid.uuid4().hex[:8]}"
        processed = 0
        while True:
            WORKER_ACTIVE.labels(worker_id=effective_worker_id).set(time.time())
            streams = self.redis_client.xreadgroup(
                groupname=REDIS_CONSUMER_GROUP,
                consumername=effective_worker_id,
                streams={REDIS_STREAM_KEY: ">"},
                count=1,
                block=1000 if once else REDIS_STREAM_BLOCK_MILLISECONDS,
            )
            claimed_from_pending = False
            if not streams:
                reclaimed = self._redis_claim_stale_messages(effective_worker_id)
                if reclaimed:
                    streams = reclaimed
                    claimed_from_pending = True
            if not streams:
                WORKER_POLL_TOTAL.labels(outcome="empty").inc()
                if once:
                    return processed
                continue
            _, entries = streams[0]
            if not entries:
                WORKER_POLL_TOTAL.labels(outcome="empty").inc()
                if once:
                    return processed
                continue
            message_id, payload = entries[0]
            job_id = str(payload.get("job_id", "")).strip()
            if not job_id:
                self._redis_ack_and_delete(message_id)
                WORKER_POLL_TOTAL.labels(outcome="invalid_stream_item").inc()
                if once:
                    return processed
                continue
            job = self.job_store.claim_job(job_id, effective_worker_id, lease_seconds=JOB_LEASE_SECONDS)
            self.refresh_metrics()
            if not job:
                self._redis_ack_and_delete(message_id)
                WORKER_POLL_TOTAL.labels(outcome="orphaned_queue_item").inc()
                if once:
                    return processed
                continue
            WORKER_POLL_TOTAL.labels(outcome="reclaimed" if claimed_from_pending else "claimed").inc()
            self.run_job(job["job_id"])
            self._redis_ack_and_delete(message_id)
            processed += 1
            if once:
                return processed

    def _redis_claim_stale_messages(self, worker_id: str) -> list[Any]:
        if not self.redis_client:
            return []
        try:
            result = self.redis_client.xautoclaim(
                REDIS_STREAM_KEY,
                REDIS_CONSUMER_GROUP,
                worker_id,
                min_idle_time=REDIS_STREAM_CLAIM_IDLE_MILLISECONDS,
                start_id="0-0",
                count=1,
            )
        except Exception:
            return []
        if not result:
            return []
        if len(result) >= 2:
            messages = result[1] or []
            if messages:
                return [(REDIS_STREAM_KEY, messages)]
        return []

    @staticmethod
    def _retry_delay_seconds(attempt: int) -> float:
        exponent = max(0, attempt - 1)
        return RETRY_BACKOFF_SECONDS * (2 ** exponent)

    @staticmethod
    def _webhook_retry_delay_seconds(attempt: int) -> float:
        exponent = max(0, attempt - 1)
        return WEBHOOK_RETRY_BACKOFF_SECONDS * (2 ** exponent)

    @staticmethod
    def _should_retry_webhook_status(status_code: int, attempt: int) -> bool:
        if attempt >= WEBHOOK_MAX_ATTEMPTS:
            return False
        if status_code >= 500:
            return True
        return status_code in {408, 409, 425, 429}

    def replay_dead_letter_job(self, job_id: str) -> dict[str, Any] | None:
        job = self.job_store.get_job(job_id)
        if not job or job.get("status") != "dead_letter":
            return None
        replayed = self.job_store.replay_job(job_id)
        if replayed:
            self.upsert_job(replayed)
            self.enqueue_job(job_id)
        return replayed

    def worker_status(self) -> dict[str, Any]:
        counts = self.job_store.status_counts()
        return {
            "backend": WORKER_BACKEND,
            "queued": self.job_store.queued_count(),
            "status_counts": counts,
            "inprocess_workers": len(self.worker_threads),
            "max_attempts": MAX_JOB_ATTEMPTS,
            "retry_backoff_seconds": RETRY_BACKOFF_SECONDS,
            "lease_seconds": JOB_LEASE_SECONDS,
            "heartbeat_interval_seconds": WORKER_HEARTBEAT_INTERVAL_SECONDS,
            "webhook_max_attempts": WEBHOOK_MAX_ATTEMPTS,
            "webhook_retry_backoff_seconds": WEBHOOK_RETRY_BACKOFF_SECONDS,
            "webhook_signing_enabled": bool(WEBHOOK_SIGNING_SECRET),
            "redis_enabled": bool(self.redis_client),
            "redis_url": REDIS_URL if WORKER_BACKEND == "redis" else None,
            "redis_queue_key": REDIS_QUEUE_KEY if WORKER_BACKEND == "redis" else None,
            "redis_stream_key": REDIS_STREAM_KEY if WORKER_BACKEND == "redis" else None,
            "redis_consumer_group": REDIS_CONSUMER_GROUP if WORKER_BACKEND == "redis" else None,
            "redis_pending": self._redis_pending_summary() if WORKER_BACKEND == "redis" else None,
            "redis_healthy": self._redis_ping() if WORKER_BACKEND == "redis" else None,
        }
