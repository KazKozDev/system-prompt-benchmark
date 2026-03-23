"""SQLite-backed API job store."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class JobStore:
    def __init__(self, path: str = "results/api.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    owner TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    output_path TEXT,
                    result_id TEXT,
                    summary_json TEXT,
                    error TEXT,
                    config_json TEXT NOT NULL,
                    webhook_url TEXT,
                    webhook_status TEXT,
                    webhook_attempted_at TEXT,
                    webhook_attempts INTEGER DEFAULT 0,
                    webhook_last_error TEXT,
                    webhook_last_status_code INTEGER,
                    webhook_delivery_id TEXT,
                    webhook_delivered_at TEXT,
                    worker_backend TEXT,
                    locked_by TEXT,
                    lease_until TEXT,
                    started_at TEXT,
                    attempts INTEGER DEFAULT 0,
                    next_retry_at TEXT,
                    last_error_at TEXT
                )
                """
            )
            existing_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(jobs)"
                ).fetchall()
            }
            migrations = {
                "worker_backend": (
                    "ALTER TABLE jobs ADD COLUMN worker_backend TEXT"
                ),
                "locked_by": "ALTER TABLE jobs ADD COLUMN locked_by TEXT",
                "lease_until": "ALTER TABLE jobs ADD COLUMN lease_until TEXT",
                "started_at": "ALTER TABLE jobs ADD COLUMN started_at TEXT",
                "attempts": (
                    "ALTER TABLE jobs ADD COLUMN attempts INTEGER DEFAULT 0"
                ),
                "next_retry_at": (
                    "ALTER TABLE jobs ADD COLUMN next_retry_at TEXT"
                ),
                "last_error_at": (
                    "ALTER TABLE jobs ADD COLUMN last_error_at TEXT"
                ),
                "webhook_attempts": (
                    "ALTER TABLE jobs ADD COLUMN webhook_attempts "
                    "INTEGER DEFAULT 0"
                ),
                "webhook_last_error": (
                    "ALTER TABLE jobs ADD COLUMN webhook_last_error TEXT"
                ),
                "webhook_last_status_code": (
                    "ALTER TABLE jobs ADD COLUMN "
                    "webhook_last_status_code INTEGER"
                ),
                "webhook_delivery_id": (
                    "ALTER TABLE jobs ADD COLUMN webhook_delivery_id TEXT"
                ),
                "webhook_delivered_at": (
                    "ALTER TABLE jobs ADD COLUMN webhook_delivered_at TEXT"
                ),
            }
            for column, statement in migrations.items():
                if column not in existing_columns:
                    connection.execute(statement)
            connection.commit()

    def upsert_job(self, job: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, status, owner, created_at, completed_at,
                    output_path, result_id, summary_json, error,
                    config_json, webhook_url, webhook_status,
                    webhook_attempted_at, webhook_attempts,
                    webhook_last_error, webhook_last_status_code,
                    webhook_delivery_id, webhook_delivered_at,
                    worker_backend, locked_by, lease_until,
                    started_at, attempts, next_retry_at, last_error_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    owner=excluded.owner,
                    created_at=excluded.created_at,
                    completed_at=excluded.completed_at,
                    output_path=excluded.output_path,
                    result_id=excluded.result_id,
                    summary_json=excluded.summary_json,
                    error=excluded.error,
                    config_json=excluded.config_json,
                    webhook_url=excluded.webhook_url,
                    webhook_status=excluded.webhook_status,
                    webhook_attempted_at=excluded.webhook_attempted_at,
                    webhook_attempts=excluded.webhook_attempts,
                    webhook_last_error=excluded.webhook_last_error,
                    webhook_last_status_code=excluded.webhook_last_status_code,
                    webhook_delivery_id=excluded.webhook_delivery_id,
                    webhook_delivered_at=excluded.webhook_delivered_at,
                    worker_backend=excluded.worker_backend,
                    locked_by=excluded.locked_by,
                    lease_until=excluded.lease_until,
                    started_at=excluded.started_at,
                    attempts=excluded.attempts,
                    next_retry_at=excluded.next_retry_at,
                    last_error_at=excluded.last_error_at
                """,
                (
                    job["job_id"],
                    job["status"],
                    job.get("owner"),
                    job["created_at"],
                    job.get("completed_at"),
                    job.get("output_path"),
                    job.get("result_id"),
                    (
                        json.dumps(job.get("summary"), ensure_ascii=False)
                        if job.get("summary") is not None
                        else None
                    ),
                    job.get("error"),
                    json.dumps(job.get("config"), ensure_ascii=False),
                    job.get("webhook_url"),
                    job.get("webhook_status"),
                    job.get("webhook_attempted_at"),
                    int(job.get("webhook_attempts", 0)),
                    job.get("webhook_last_error"),
                    job.get("webhook_last_status_code"),
                    job.get("webhook_delivery_id"),
                    job.get("webhook_delivered_at"),
                    job.get("worker_backend"),
                    job.get("locked_by"),
                    job.get("lease_until"),
                    job.get("started_at"),
                    int(job.get("attempts", 0)),
                    job.get("next_retry_at"),
                    job.get("last_error_at"),
                ),
            )
            connection.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        statuses: list[str] | None = None,
        owner: str | None = None,
        has_results: bool | None = None,
        search_term: str | None = None,
        provider: str | None = None,
        dataset: str | None = None,
        webhook_statuses: list[str] | None = None,
        order_by: str = "created_at_desc",
    ) -> list[dict[str, Any]]:
        query, params = self._build_jobs_query(
            select_clause="SELECT *",
            statuses=statuses,
            owner=owner,
            has_results=has_results,
            search_term=search_term,
            provider=provider,
            dataset=dataset,
            webhook_statuses=webhook_statuses,
        )
        params.extend([int(limit), int(offset)])
        order_clause = self._build_order_clause(order_by)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"{query} {order_clause} LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def count_jobs(
        self,
        *,
        statuses: list[str] | None = None,
        owner: str | None = None,
        has_results: bool | None = None,
        search_term: str | None = None,
        provider: str | None = None,
        dataset: str | None = None,
        webhook_statuses: list[str] | None = None,
    ) -> int:
        query, params = self._build_jobs_query(
            select_clause="SELECT COUNT(*) AS count",
            statuses=statuses,
            owner=owner,
            has_results=has_results,
            search_term=search_term,
            provider=provider,
            dataset=dataset,
            webhook_statuses=webhook_statuses,
        )
        with self._lock, self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return int(row["count"]) if row else 0

    def list_pending_jobs(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') "
                "ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def queued_count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE status = 'queued'"
            ).fetchone()
        return int(row["count"]) if row else 0

    def claim_next_job(
        self,
        worker_id: str,
        lease_seconds: float = 7200.0,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=max(1.0, float(lease_seconds)))
        now_text = now.isoformat()
        lease_text = lease_until.isoformat()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE (
                    status = 'queued'
                    AND (next_retry_at IS NULL OR next_retry_at <= ?)
                )
                   OR (
                    status = 'running'
                    AND lease_until IS NOT NULL
                    AND lease_until < ?
                )
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (now_text, now_text),
            ).fetchone()
            if not row:
                return None
            job_id = str(row["job_id"])
            attempts = int(row["attempts"] or 0) + 1
            started_at = row["started_at"] or now_text
            connection.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    locked_by = ?,
                    lease_until = ?,
                    started_at = ?,
                    attempts = ?,
                    next_retry_at = NULL
                WHERE job_id = ?
                """,
                (worker_id, lease_text, started_at, attempts, job_id),
            )
            connection.commit()
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(claimed) if claimed else None

    def claim_job(
        self,
        job_id: str,
        worker_id: str,
        lease_seconds: float = 7200.0,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=max(1.0, float(lease_seconds)))
        now_text = now.isoformat()
        lease_text = lease_until.isoformat()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE job_id = ?
                  AND (
                    (
                        status = 'queued'
                        AND (
                            next_retry_at IS NULL
                            OR next_retry_at <= ?
                        )
                    )
                    OR (
                        status = 'running'
                        AND lease_until IS NOT NULL
                        AND lease_until < ?
                    )
                  )
                """,
                (job_id, now_text, now_text),
            ).fetchone()
            if not row:
                return None
            attempts = int(row["attempts"] or 0) + 1
            started_at = row["started_at"] or now_text
            connection.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    locked_by = ?,
                    lease_until = ?,
                    started_at = ?,
                    attempts = ?,
                    next_retry_at = NULL
                WHERE job_id = ?
                """,
                (worker_id, lease_text, started_at, attempts, job_id),
            )
            connection.commit()
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(claimed) if claimed else None

    def renew_lease(
        self,
        job_id: str,
        worker_id: str,
        lease_seconds: float = 7200.0,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        lease_until = now + timedelta(seconds=max(1.0, float(lease_seconds)))
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ? AND status = 'running' "
                "AND locked_by = ?",
                (job_id, worker_id),
            ).fetchone()
            if not row:
                return None
            connection.execute(
                "UPDATE jobs SET lease_until = ? WHERE job_id = ?",
                (lease_until.isoformat(), job_id),
            )
            connection.commit()
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(updated) if updated else None

    def release_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        status: str,
        error: str | None = None,
        completed_at: str | None = None,
        retryable: bool = False,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.0,
    ) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                return None
            attempts = int(row["attempts"] or 0)
            next_status = status
            lease_until = None
            locked_by = None
            next_retry_at = None
            last_error_at = None
            if retryable and attempts < max_attempts:
                next_status = "queued"
                next_retry_at = (
                    datetime.now(UTC)
                    + timedelta(
                        seconds=max(0.0, float(retry_delay_seconds))
                    )
                ).isoformat()
            elif retryable and attempts >= max_attempts:
                next_status = "dead_letter"
            if error:
                last_error_at = datetime.now(UTC).isoformat()
            connection.execute(
                """
                UPDATE jobs
                SET status = ?,
                    error = ?,
                    completed_at = ?,
                    locked_by = ?,
                    lease_until = ?,
                    next_retry_at = ?,
                    last_error_at = ?
                WHERE job_id = ?
                """,
                (
                    next_status,
                    error,
                    completed_at,
                    locked_by,
                    lease_until,
                    next_retry_at,
                    last_error_at,
                    job_id,
                ),
            )
            connection.commit()
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(updated) if updated else None

    def replay_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                return None
            connection.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    error = NULL,
                    completed_at = NULL,
                    output_path = NULL,
                    result_id = NULL,
                    summary_json = NULL,
                    locked_by = NULL,
                    lease_until = NULL,
                    next_retry_at = NULL
                WHERE job_id = ?
                """,
                (job_id,),
            )
            connection.commit()
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._row_to_job(updated) if updated else None

    def status_counts(self) -> dict[str, int]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def _build_jobs_query(
        self,
        *,
        select_clause: str,
        statuses: list[str] | None,
        owner: str | None,
        has_results: bool | None,
        search_term: str | None,
        provider: str | None,
        dataset: str | None,
        webhook_statuses: list[str] | None,
    ) -> tuple[str, list[Any]]:
        query = f"{select_clause} FROM jobs"
        clauses: list[str] = []
        params: list[Any] = []

        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        if owner:
            clauses.append("LOWER(COALESCE(owner, '')) LIKE ?")
            params.append(f"%{owner.lower()}%")
        if has_results is True:
            clauses.append("COALESCE(output_path, '') != ''")
        elif has_results is False:
            clauses.append("COALESCE(output_path, '') = ''")
        if provider:
            clauses.append("LOWER(COALESCE(config_json, '')) LIKE ?")
            params.append(f"%{provider.lower()}%")
        if dataset:
            clauses.append("LOWER(COALESCE(config_json, '')) LIKE ?")
            params.append(f"%{dataset.lower()}%")
        if webhook_statuses:
            placeholders = ", ".join("?" for _ in webhook_statuses)
            clauses.append(f"COALESCE(webhook_status, '') IN ({placeholders})")
            params.extend(webhook_statuses)
        if search_term:
            wildcard = f"%{search_term.lower()}%"
            search_clauses = [
                "LOWER(COALESCE(job_id, '')) LIKE ?",
                "LOWER(COALESCE(owner, '')) LIKE ?",
                "LOWER(COALESCE(result_id, '')) LIKE ?",
                "LOWER(COALESCE(status, '')) LIKE ?",
                "LOWER(COALESCE(error, '')) LIKE ?",
                "LOWER(COALESCE(output_path, '')) LIKE ?",
                "LOWER(COALESCE(config_json, '')) LIKE ?",
            ]
            clauses.append("(" + " OR ".join(search_clauses) + ")")
            params.extend([wildcard] * len(search_clauses))

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        return query, params

    @staticmethod
    def _build_order_clause(order_by: str) -> str:
        clauses = {
            "created_at_desc": "ORDER BY created_at DESC, job_id DESC",
            "created_at_asc": "ORDER BY created_at ASC, job_id ASC",
            "completed_at_desc": (
                "ORDER BY CASE WHEN completed_at IS NULL THEN 1 ELSE 0 END, "
                "completed_at DESC, created_at DESC"
            ),
            "completed_at_asc": (
                "ORDER BY CASE WHEN completed_at IS NULL THEN 1 ELSE 0 END, "
                "completed_at ASC, created_at ASC"
            ),
            "owner_asc": (
                "ORDER BY LOWER(COALESCE(owner, '')) ASC, created_at DESC"
            ),
            "owner_desc": (
                "ORDER BY LOWER(COALESCE(owner, '')) DESC, created_at DESC"
            ),
            "status_asc": (
                "ORDER BY LOWER(COALESCE(status, '')) ASC, created_at DESC"
            ),
            "status_desc": (
                "ORDER BY LOWER(COALESCE(status, '')) DESC, created_at DESC"
            ),
        }
        return clauses.get(order_by, clauses["created_at_desc"])

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "owner": row["owner"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "output_path": row["output_path"],
            "result_id": row["result_id"],
            "summary": (
                json.loads(row["summary_json"])
                if row["summary_json"]
                else None
            ),
            "error": row["error"],
            "config": (
                json.loads(row["config_json"])
                if row["config_json"]
                else {}
            ),
            "webhook_url": row["webhook_url"],
            "webhook_status": row["webhook_status"],
            "webhook_attempted_at": row["webhook_attempted_at"],
            "webhook_attempts": int(row["webhook_attempts"] or 0),
            "webhook_last_error": row["webhook_last_error"],
            "webhook_last_status_code": row["webhook_last_status_code"],
            "webhook_delivery_id": row["webhook_delivery_id"],
            "webhook_delivered_at": row["webhook_delivered_at"],
            "worker_backend": row["worker_backend"],
            "locked_by": row["locked_by"],
            "lease_until": row["lease_until"],
            "started_at": row["started_at"],
            "attempts": int(row["attempts"] or 0),
            "next_retry_at": row["next_retry_at"],
            "last_error_at": row["last_error_at"],
        }
