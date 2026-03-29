"""Benchmark preset storage and Streamlit session helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.ui.help_views import PENDING_WORKSPACE_MODE_KEY

from src.config import JudgeConfig, ProviderConfig
from src.ui import provider_views
from src.ui.evaluation_views import (
    JUDGE_MODEL_MANUAL_KEY,
    JUDGE_MODEL_SELECT_KEY,
    JUDGE_PROVIDER_FIELD_KEYS,
)
from src.ui.provider_views import (
    PROVIDER_FIELD_KEYS,
    PROVIDER_TEST_SESSION_KEY,
    _apply_preset_to_session,
)


PRESET_DIR = Path("config/benchmarks")
BENCHMARK_PRESET_VERSION = 1
BENCHMARK_DATASET_SNAPSHOT_SOURCE = "Saved Benchmark Preset Snapshot"
BENCHMARK_DATASET_SNAPSHOT_KEY = "benchmark_dataset_snapshot"
BENCHMARK_MODE_KEY = "benchmark_mode_select"
BENCHMARK_CUSTOM_TESTS_KEY = "benchmark_custom_num_tests"
BENCHMARK_USE_SEMANTIC_KEY = "benchmark_use_semantic"
BENCHMARK_USE_DEGRADATION_KEY = "benchmark_use_degradation"
BENCHMARK_AUTO_ANALYZE_KEY = "benchmark_auto_analyze"
DATASET_SOURCE_KEY = "dataset_source_select"


def list_benchmark_presets() -> list[str]:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path.stem for path in PRESET_DIR.glob("*.json"))


def load_benchmark_preset(name: str) -> dict[str, Any]:
    preset_path = PRESET_DIR / f"{name}.json"
    with preset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_benchmark_preset_payload(payload)
    return payload


def save_benchmark_preset(name: str, payload: dict[str, Any]) -> str:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _slugify_preset_name(name) or "benchmark-preset"
    preset_path = PRESET_DIR / f"{safe_name}.json"
    with preset_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return safe_name


def delete_benchmark_preset(name: str) -> None:
    preset_path = PRESET_DIR / f"{name}.json"
    if preset_path.exists():
        preset_path.unlink()


def serialize_benchmark_preset(payload: dict[str, Any]) -> str:
    validate_benchmark_preset_payload(payload)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def parse_benchmark_preset_json(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid preset JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Benchmark preset must be a JSON object")
    validate_benchmark_preset_payload(payload)
    return payload


def validate_benchmark_preset_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Benchmark preset must be a dictionary")
    if not isinstance(payload.get("prompt"), dict):
        raise ValueError("Benchmark preset must include a prompt section")
    if not isinstance(payload.get("provider_preset"), dict):
        raise ValueError(
            "Benchmark preset must include a provider_preset section"
        )
    if not isinstance(payload.get("benchmark"), dict):
        raise ValueError("Benchmark preset must include a benchmark section")
    dataset_snapshot = payload.get("dataset_snapshot")
    if not isinstance(dataset_snapshot, dict):
        raise ValueError(
            "Benchmark preset must include a dataset_snapshot section"
        )
    tests = dataset_snapshot.get("tests")
    if tests is None or not isinstance(tests, list):
        raise ValueError("dataset_snapshot.tests must be a list")


def benchmark_preset_readiness(
    payload: dict[str, Any],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    try:
        validate_benchmark_preset_payload(payload)
    except ValueError as exc:
        issues.append(str(exc))
        return False, issues

    prompt = payload.get("prompt", {})
    if not str(prompt.get("system_prompt") or "").strip():
        issues.append("System prompt is empty")

    dataset_snapshot = payload.get("dataset_snapshot", {})
    if not dataset_snapshot.get("tests"):
        issues.append("Dataset snapshot does not contain tests")
    if dataset_snapshot.get("issues"):
        issues.append("Dataset snapshot contains validation issues")

    return len(issues) == 0, issues


def benchmark_preset_summary(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = payload.get("prompt", {})
    benchmark = payload.get("benchmark", {})
    provider_data = payload.get("provider_preset", {}).get("provider", {})
    dataset_snapshot = payload.get("dataset_snapshot", {})

    system_prompt = str(prompt.get("system_prompt") or "")
    dataset_label = dataset_snapshot.get("label") or dataset_snapshot.get(
        "path"
    )
    metadata = dataset_snapshot.get("metadata", {})

    return {
        "prompt_preview": system_prompt[:180],
        "prompt_length": len(system_prompt),
        "provider": provider_data.get("name") or "unknown",
        "model": provider_data.get("model") or "default",
        "dataset": dataset_label or metadata.get("name") or "unknown",
        "test_count": len(dataset_snapshot.get("tests", [])),
        "mode": benchmark.get("mode") or "unknown",
    }


def build_benchmark_preset_payload(
    *,
    system_prompt: str,
    prompt_source: str,
    provider_config: ProviderConfig,
    judge_config: JudgeConfig,
    dataset_state: dict[str, Any],
    mode: str,
    num_tests: int,
    custom_num_tests: int,
    use_semantic: bool,
    use_degradation: bool,
    auto_analyze: bool,
    generated_pack: dict[str, Any] | None,
    provider_test_result: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "version": BENCHMARK_PRESET_VERSION,
        "saved_at": datetime.now(UTC).isoformat(),
        "prompt": {
            "system_prompt": system_prompt,
            "source": prompt_source,
        },
        "provider_preset": {
            "auth_mode": st.session_state.get(
                PROVIDER_FIELD_KEYS["auth_mode"],
                provider_views.AUTH_MODE_ENV,
            ),
            "provider": {
                **asdict(provider_config),
                "api_key": None,
            },
        },
        "judge": _serialize_judge_config(judge_config),
        "benchmark": {
            "mode": mode,
            "num_tests": num_tests,
            "custom_num_tests": custom_num_tests,
            "use_semantic": use_semantic,
            "use_degradation": use_degradation,
            "auto_analyze": auto_analyze,
        },
        "dataset_snapshot": {
            "source": dataset_state.get("dataset_source"),
            "label": dataset_state.get("dataset_label"),
            "path": dataset_state.get("dataset_path"),
            "tests": dataset_state.get("dataset_tests", []),
            "issues": dataset_state.get("dataset_issues", []),
            "metadata": dataset_state.get("dataset_metadata", {}),
        },
        "generated_pack": generated_pack,
        "provider_test_result": provider_test_result,
    }


def apply_benchmark_preset_to_session(payload: dict[str, Any]) -> None:
    prompt = payload.get("prompt", {})
    benchmark = payload.get("benchmark", {})
    judge = payload.get("judge", {})

    st.session_state[PENDING_WORKSPACE_MODE_KEY] = "Benchmark"
    st.session_state["prompt_source_radio"] = "Paste Text"
    st.session_state["prompt_text_area"] = prompt.get("system_prompt", "")
    st.session_state["system_prompt"] = prompt.get("system_prompt", "")

    provider_preset = payload.get("provider_preset")
    if provider_preset:
        _apply_preset_to_session(provider_preset)

    _apply_judge_settings_to_session(judge)

    st.session_state[BENCHMARK_MODE_KEY] = benchmark.get(
        "mode",
        "Standard (100 tests)",
    )
    st.session_state[BENCHMARK_CUSTOM_TESTS_KEY] = int(
        benchmark.get("custom_num_tests", benchmark.get("num_tests", 50))
    )
    st.session_state[BENCHMARK_USE_SEMANTIC_KEY] = bool(
        benchmark.get("use_semantic", True)
    )
    st.session_state[BENCHMARK_USE_DEGRADATION_KEY] = bool(
        benchmark.get("use_degradation", True)
    )
    st.session_state[BENCHMARK_AUTO_ANALYZE_KEY] = bool(
        benchmark.get("auto_analyze", True)
    )

    st.session_state[BENCHMARK_DATASET_SNAPSHOT_KEY] = payload.get(
        "dataset_snapshot",
        {},
    )
    st.session_state[DATASET_SOURCE_KEY] = BENCHMARK_DATASET_SNAPSHOT_SOURCE
    st.session_state["generated_pack"] = payload.get("generated_pack")

    provider_test_result = payload.get("provider_test_result")
    if provider_test_result is not None:
        st.session_state[PROVIDER_TEST_SESSION_KEY] = provider_test_result


def _apply_judge_settings_to_session(judge: dict[str, Any]) -> None:
    defaults = JudgeConfig()
    st.session_state["eval_strategy"] = judge.get(
        "strategy",
        defaults.strategy,
    )
    st.session_state["eval_pass_threshold"] = float(
        judge.get("pass_threshold", defaults.pass_threshold)
    )
    st.session_state["eval_review_threshold"] = float(
        judge.get("review_threshold", defaults.review_threshold)
    )
    st.session_state["eval_pattern_detectors_enabled"] = bool(
        judge.get(
            "pattern_detectors_enabled",
            defaults.pattern_detectors_enabled,
        )
    )
    st.session_state["eval_detector_weight"] = float(
        judge.get("detector_weight", defaults.detector_weight)
    )
    st.session_state["eval_openai_moderation_enabled"] = bool(
        judge.get(
            "openai_moderation_enabled",
            defaults.openai_moderation_enabled,
        )
    )
    st.session_state["eval_openai_moderation_model"] = (
        judge.get("openai_moderation_model", defaults.openai_moderation_model)
        or defaults.openai_moderation_model
    )
    st.session_state["eval_openai_moderation_api_key_env"] = (
        judge.get("openai_moderation_api_key_env") or "OPENAI_API_KEY"
    )
    st.session_state["eval_openai_moderation_base_url"] = (
        judge.get("openai_moderation_base_url") or ""
    )
    st.session_state["eval_perspective_enabled"] = bool(
        judge.get("perspective_enabled", defaults.perspective_enabled)
    )
    st.session_state["eval_perspective_api_key_env"] = (
        judge.get("perspective_api_key_env") or "PERSPECTIVE_API_KEY"
    )
    st.session_state["eval_perspective_threshold"] = float(
        judge.get("perspective_threshold", defaults.perspective_threshold)
    )
    st.session_state["eval_harmjudge_enabled"] = bool(
        judge.get("harmjudge_enabled", defaults.harmjudge_enabled)
    )
    st.session_state["eval_harmjudge_model"] = (
        judge.get("harmjudge_model") or "harmjudge"
    )
    st.session_state["eval_harmjudge_api_key_env"] = (
        judge.get("harmjudge_api_key_env") or "HARMJUDGE_API_KEY"
    )
    st.session_state["eval_harmjudge_base_url"] = (
        judge.get("harmjudge_base_url") or ""
    )
    st.session_state["eval_external_detector_url"] = (
        judge.get("external_detector_url") or ""
    )
    st.session_state["eval_external_detector_api_key_env"] = (
        judge.get("external_detector_api_key_env") or ""
    )
    st.session_state["eval_attacker_base_url"] = (
        judge.get("attacker_base_url", defaults.attacker_base_url)
        or defaults.attacker_base_url
    )
    st.session_state["eval_attacker_model"] = (
        judge.get("attacker_model", defaults.attacker_model)
        or defaults.attacker_model
    )
    st.session_state["eval_attacker_api_key_env"] = (
        judge.get("attacker_api_key_env") or ""
    )
    st.session_state["eval_attacker_temperature"] = float(
        judge.get("attacker_temperature", defaults.attacker_temperature)
    )

    judge_provider = dict(judge.get("provider") or {})
    judge_provider_name = judge_provider.get("name", defaults.provider.name)
    provider_specs = provider_views._provider_specs_with_plugins()
    provider_spec = provider_specs.get(
        judge_provider_name,
        provider_views.PROVIDER_SPECS["ollama"],
    )
    if provider_views._supports_dynamic_model_options(
        judge_provider_name,
        "chat",
    ):
        default_model = ""
    else:
        default_model = provider_spec.get("default_model", "")
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["provider_name"]] = (
        judge_provider_name
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["auth_mode"]] = (
        "Paste API Key"
        if judge_provider.get("api_key")
        else "Use Environment Variable"
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["model"]] = (
        judge_provider.get(
            "model",
            judge.get("ollama_model", defaults.ollama_model),
        )
        or default_model
    )
    st.session_state[JUDGE_MODEL_SELECT_KEY] = st.session_state[
        JUDGE_PROVIDER_FIELD_KEYS["model"]
    ]
    st.session_state[JUDGE_MODEL_MANUAL_KEY] = False
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["api_key"]] = ""
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["api_key_env"]] = (
        judge_provider.get("api_key_env")
        or provider_spec.get("default_api_key_env", "")
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["base_url"]] = (
        judge_provider.get("base_url")
        or provider_spec.get("default_base_url", "")
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["api_version"]] = (
        judge_provider.get("api_version") or ""
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["aws_region"]] = (
        judge_provider.get("aws_region") or ""
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["project_id"]] = (
        judge_provider.get("project_id") or ""
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["location"]] = (
        judge_provider.get("location") or ""
    )

    catalog_enabled = dict(judge.get("detector_family_enabled", {}))
    catalog_weights = dict(judge.get("detector_family_weights", {}))
    for detector_name, enabled in catalog_enabled.items():
        st.session_state[f"eval_detector_enabled_{detector_name}"] = bool(
            enabled
        )
    for detector_name, weight in catalog_weights.items():
        st.session_state[f"eval_detector_weight_{detector_name}"] = float(
            weight
        )


def _slugify_preset_name(value: str) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else "-" for char in value.strip()
    )
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed[:80]


def _serialize_judge_config(judge_config: JudgeConfig) -> dict[str, Any]:
    payload = asdict(judge_config)
    payload["provider"] = {
        **payload.get("provider", {}),
        "api_key": None,
    }
    return payload
