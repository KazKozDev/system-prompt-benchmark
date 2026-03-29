"""Tests for benchmark preset persistence helpers."""

from __future__ import annotations

from pathlib import Path

from src.config import JudgeConfig, ProviderConfig


def test_benchmark_preset_round_trip_and_delete(monkeypatch, tmp_path: Path):
    from src.ui import benchmark_presets

    monkeypatch.setattr(benchmark_presets, "PRESET_DIR", tmp_path)

    payload = {
        "version": 1,
        "prompt": {"system_prompt": "You are safe.", "source": "Paste Text"},
        "provider_preset": {
            "auth_mode": "Use Environment Variable",
            "provider": {"name": "ollama", "model": "qwen3.5:9b"},
        },
        "benchmark": {"mode": "Quick (10 tests)", "num_tests": 10},
        "dataset_snapshot": {
            "source": "Built-in Benchmark",
            "label": "Saved Snapshot",
            "tests": [{"id": 1, "category": "demo", "input": "hi"}],
            "issues": [],
            "metadata": {"name": "Saved Snapshot", "version": "1.0"},
        },
    }

    saved_name = benchmark_presets.save_benchmark_preset("My Preset", payload)
    assert saved_name == "my-preset"
    assert benchmark_presets.list_benchmark_presets() == ["my-preset"]

    loaded = benchmark_presets.load_benchmark_preset(saved_name)
    assert loaded == payload
    assert (
        benchmark_presets.parse_benchmark_preset_json(
            benchmark_presets.serialize_benchmark_preset(payload)
        )
        == payload
    )

    benchmark_presets.delete_benchmark_preset(saved_name)
    assert benchmark_presets.list_benchmark_presets() == []


def test_benchmark_preset_round_trip_restores_run_ready_session(
    monkeypatch,
    tmp_path: Path,
):
    from src.ui import benchmark_presets, provider_views
    from src.ui.evaluation_views import JUDGE_PROVIDER_FIELD_KEYS

    fake_session: dict[str, object] = {}
    monkeypatch.setattr(benchmark_presets, "PRESET_DIR", tmp_path)
    monkeypatch.setattr(
        benchmark_presets.st,
        "session_state",
        fake_session,
        raising=False,
    )
    monkeypatch.setattr(
        provider_views.st,
        "session_state",
        fake_session,
        raising=False,
    )

    fake_session[provider_views.PROVIDER_FIELD_KEYS["auth_mode"]] = (
        "Use Environment Variable"
    )

    payload = benchmark_presets.build_benchmark_preset_payload(
        system_prompt="You are safe.",
        prompt_source="Paste Text",
        provider_config=ProviderConfig(name="ollama", model="qwen3.5:9b"),
        judge_config=JudgeConfig(
            provider=ProviderConfig(
                name="anthropic",
                model="claude-3-5-sonnet-20241022",
                api_key="secret",
                api_key_env="ANTHROPIC_API_KEY",
            )
        ),
        dataset_state={
            "dataset_source": "Built-in Benchmark",
            "dataset_label": "Saved Snapshot",
            "dataset_path": "tests/safeprompt-benchmark-v2.json",
            "dataset_tests": [
                {
                    "id": 1,
                    "category": "demo",
                    "input": "hello",
                }
            ],
            "dataset_issues": [],
            "dataset_metadata": {
                "name": "Saved Snapshot",
                "version": "1.0",
            },
        },
        mode="Quick (10 tests)",
        num_tests=10,
        custom_num_tests=10,
        use_semantic=True,
        use_degradation=True,
        auto_analyze=True,
        generated_pack={
            "label": "Generated Pack",
            "metadata": {"name": "Generated Pack"},
            "tests": [{"id": 101, "category": "generated"}],
        },
        provider_test_result={"ok": True},
    )

    saved_name = benchmark_presets.save_benchmark_preset(
        "Integrated Preset",
        payload,
    )
    loaded = benchmark_presets.load_benchmark_preset(saved_name)

    assert payload["judge"]["provider"]["api_key"] is None

    is_ready, readiness_issues = benchmark_presets.benchmark_preset_readiness(
        loaded
    )
    assert is_ready is True
    assert readiness_issues == []

    fake_session.clear()
    benchmark_presets.apply_benchmark_preset_to_session(loaded)

    assert (
        fake_session[benchmark_presets.PENDING_WORKSPACE_MODE_KEY]
        == "Benchmark"
    )
    assert fake_session["prompt_text_area"] == "You are safe."
    assert fake_session["system_prompt"] == "You are safe."
    assert (
        fake_session[benchmark_presets.DATASET_SOURCE_KEY]
        == benchmark_presets.BENCHMARK_DATASET_SNAPSHOT_SOURCE
    )
    assert fake_session[benchmark_presets.BENCHMARK_DATASET_SNAPSHOT_KEY][
        "tests"
    ] == [{"id": 1, "category": "demo", "input": "hello"}]
    assert fake_session["generated_pack"]["label"] == "Generated Pack"
    assert (
        fake_session[provider_views.PROVIDER_FIELD_KEYS["provider_name"]]
        == "ollama"
    )
    assert (
        fake_session[provider_views.PROVIDER_TEST_SESSION_KEY]
        == {"ok": True}
    )
    assert (
        fake_session[JUDGE_PROVIDER_FIELD_KEYS["provider_name"]]
        == "anthropic"
    )
    assert (
        fake_session[JUDGE_PROVIDER_FIELD_KEYS["model"]]
        == "claude-3-5-sonnet-20241022"
    )
