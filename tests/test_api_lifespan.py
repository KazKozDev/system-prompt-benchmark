"""Tests for FastAPI lifecycle startup behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_app_lifespan_starts_workers(monkeypatch) -> None:
    from src import api

    calls: list[str] = []

    monkeypatch.setattr(api, "_start_workers", lambda: calls.append("started"))

    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert calls == ["started"]
