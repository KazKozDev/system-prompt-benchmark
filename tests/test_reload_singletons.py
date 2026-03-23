"""Regression tests for module reload safety of global singletons."""

from __future__ import annotations

import importlib


def test_api_module_reuses_job_store_and_worker_backend_on_reload() -> None:
    from src import api

    job_store_id = id(api.JOB_STORE)
    worker_backend_id = id(api.WORKERS)

    reloaded = importlib.reload(api)

    assert id(reloaded.JOB_STORE) == job_store_id
    assert id(reloaded.WORKERS) == worker_backend_id


def test_plugin_manager_singleton_survives_module_reload() -> None:
    from src.plugins import manager as plugin_manager_module

    manager_id = id(plugin_manager_module.get_plugin_manager())

    reloaded = importlib.reload(plugin_manager_module)

    assert id(reloaded.get_plugin_manager()) == manager_id
