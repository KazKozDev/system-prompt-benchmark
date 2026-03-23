"""Streamlit evaluation settings views."""

from __future__ import annotations

import streamlit as st

from src.config import JudgeConfig, ProviderConfig
from src.core.detectors import builtin_detector_catalog
from src.providers.run_benchmark import create_provider
from src.ui import provider_views


JUDGE_PROVIDER_FIELD_KEYS = {
    "provider_name": "eval_judge_provider_name",
    "auth_mode": "eval_judge_auth_mode",
    "model": "eval_judge_model",
    "api_key": "eval_judge_api_key",
    "api_key_env": "eval_judge_api_key_env",
    "base_url": "eval_judge_base_url",
    "api_version": "eval_judge_api_version",
    "aws_region": "eval_judge_aws_region",
    "project_id": "eval_judge_project_id",
    "location": "eval_judge_location",
}
JUDGE_MODEL_SELECT_KEY = "eval_judge_model_select"
JUDGE_MODEL_MANUAL_KEY = "eval_judge_model_manual"
JUDGE_TEST_RESULT_KEY = "eval_judge_test_result"
JUDGE_TEST_SYSTEM_PROMPT_KEY = "eval_judge_test_system_prompt"
JUDGE_TEST_USER_MESSAGE_KEY = "eval_judge_test_user_message"


def render_evaluation_settings(
    default_pass_threshold: float, default_review_threshold: float
) -> JudgeConfig:
    _initialize_judge_provider_state()
    st.subheader("Evaluation")
    strategy = st.selectbox(
        "Judge Strategy",
        ["auto", "llm", "heuristic", "ensemble"],
        index=0,
        key="eval_strategy",
    )
    threshold_cols = st.columns(2)
    with threshold_cols[0]:
        pass_threshold = st.slider(
            "PASS Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(default_pass_threshold),
            step=0.05,
            key="eval_pass_threshold",
        )
    with threshold_cols[1]:
        review_threshold = st.slider(
            "REVIEW Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(default_review_threshold),
            step=0.05,
            key="eval_review_threshold",
        )

    with st.expander("Judge Model", expanded=False):
        st.caption(
            "Judge provider and model are independent from the main benchmark "
            "provider. Dynamic model lists load from the selected judge "
            "provider."
        )
        judge_provider = _render_judge_provider_selector()

    with st.expander("Detector Configuration", expanded=False):
        pattern_detectors_enabled = st.checkbox(
            "Enable built-in pattern detectors",
            value=True,
            key="eval_pattern_detectors_enabled",
        )
        detector_weight = st.slider(
            "Detector Weight",
            min_value=0.0,
            max_value=0.9,
            value=0.25,
            step=0.05,
            key="eval_detector_weight",
        )
        detector_family_enabled: dict[str, bool] = {}
        detector_family_weights: dict[str, float] = {}
        catalog = builtin_detector_catalog()
        st.caption(f"Built-in families: {len(catalog)}")
        if pattern_detectors_enabled:
            st.markdown("**Built-in Detector Families**")
            for detector_name, metadata in catalog.items():
                cols = st.columns([2.3, 1.1])
                with cols[0]:
                    detector_family_enabled[detector_name] = st.checkbox(
                        detector_name.replace("_", " "),
                        value=True,
                        key=f"eval_detector_enabled_{detector_name}",
                        help=metadata["description"],
                    )
                with cols[1]:
                    detector_family_weights[detector_name] = st.slider(
                        "Weight",
                        min_value=0.0,
                        max_value=3.0,
                        value=1.0,
                        step=0.25,
                        key=f"eval_detector_weight_{detector_name}",
                        label_visibility="collapsed",
                    )
        else:
            detector_family_enabled = {name: False for name in catalog}
            detector_family_weights = {name: 1.0 for name in catalog}

        openai_moderation_enabled = st.checkbox(
            "Enable OpenAI Moderation",
            value=False,
            key="eval_openai_moderation_enabled",
        )
        openai_moderation_api_key_env = ""
        openai_moderation_model = "omni-moderation-latest"
        openai_moderation_base_url = ""
        if openai_moderation_enabled:
            openai_moderation_model = st.text_input(
                "OpenAI Moderation Model",
                value="omni-moderation-latest",
                key="eval_openai_moderation_model",
            )
            openai_moderation_api_key_env = st.text_input(
                "OpenAI Moderation API Key Env Var",
                value="OPENAI_API_KEY",
                key="eval_openai_moderation_api_key_env",
            )
            openai_moderation_base_url = st.text_input(
                "OpenAI Moderation Base URL",
                value="",
                key="eval_openai_moderation_base_url",
                placeholder="optional",
            )

        perspective_enabled = st.checkbox(
            "Enable Perspective API",
            value=False,
            key="eval_perspective_enabled",
        )
        perspective_api_key_env = ""
        perspective_threshold = 0.7
        if perspective_enabled:
            perspective_api_key_env = st.text_input(
                "Perspective API Key Env Var",
                value="PERSPECTIVE_API_KEY",
                key="eval_perspective_api_key_env",
            )
            perspective_threshold = st.slider(
                "Perspective Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="eval_perspective_threshold",
            )

        harmjudge_enabled = st.checkbox(
            "Enable HarmJudge-Compatible Endpoint",
            value=False,
            key="eval_harmjudge_enabled",
        )
        harmjudge_model = ""
        harmjudge_api_key_env = ""
        harmjudge_base_url = ""
        if harmjudge_enabled:
            harmjudge_model = st.text_input(
                "HarmJudge Model",
                value="harmjudge",
                key="eval_harmjudge_model",
            )
            harmjudge_api_key_env = st.text_input(
                "HarmJudge API Key Env Var",
                value="HARMJUDGE_API_KEY",
                key="eval_harmjudge_api_key_env",
            )
            harmjudge_base_url = st.text_input(
                "HarmJudge Base URL",
                value="",
                key="eval_harmjudge_base_url",
                placeholder="https://example.internal/v1",
            )

        external_detector_url = st.text_input(
            "External Detector URL",
            value="",
            key="eval_external_detector_url",
            placeholder="optional generic detector hook",
        )
        external_detector_api_key_env = ""
        if external_detector_url:
            external_detector_api_key_env = st.text_input(
                "External Detector API Key Env Var",
                value="",
                key="eval_external_detector_api_key_env",
            )

    with st.expander("Adaptive Attacker", expanded=False):
        st.caption(
            "The adaptive attacker drives multi-turn attack loops. "
            "Rule-based strategies (`prompt_leak_escalation`, `authority_escalation`, "
            "`tool_hijack_escalation`) work without any extra model. "
            "The `llm_adaptive` strategy requires a **separate** LLM that generates "
            "each attack message — configure it below."
        )

        attacker_base_url = st.text_input(
            "Attacker Base URL",
            value="http://localhost:11434",
            key="eval_attacker_base_url",
            help=(
                "URLs containing :11434 are treated as Ollama (uses /api/generate). "
                "Any other URL is called as an OpenAI-compatible endpoint "
                "(/v1/chat/completions). Leave the default to use a local Ollama instance."
            ),
        )

        _is_ollama = ":11434" in attacker_base_url or attacker_base_url.rstrip(
            "/"
        ).endswith("/api")
        if _is_ollama:
            st.info(
                "**Ollama mode** — the attacker will POST to "
                f"`{attacker_base_url.rstrip('/')}/api/generate`. "
                "Make sure `ollama serve` is running and the model is pulled.",
            )
        else:
            st.info(
                "**OpenAI-compatible mode** — the attacker will POST to "
                f"`{attacker_base_url.rstrip('/')}/v1/chat/completions`. "
                "Set the API key env var below if authentication is required.",
            )

        attacker_model = st.text_input(
            "Attacker Model",
            value="qwen3.5:9b",
            key="eval_attacker_model",
            help="Model name passed to the attacker endpoint (e.g. qwen3.5:9b for Ollama, gpt-4o-mini for OpenAI).",
        )

        attacker_api_key_env = ""
        if not _is_ollama:
            attacker_api_key_env = st.text_input(
                "Attacker API Key Env Var",
                value="",
                key="eval_attacker_api_key_env",
                placeholder="e.g. OPENAI_API_KEY",
                help="Name of the environment variable holding the API key for the attacker endpoint. Leave blank if no auth is required.",
            )

        attacker_temperature = st.slider(
            "Attacker Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            key="eval_attacker_temperature",
        )

    return JudgeConfig(
        strategy=strategy,
        ollama_model=judge_provider.model or "qwen3.5:9b",
        provider=judge_provider,
        pass_threshold=pass_threshold,
        review_threshold=review_threshold,
        attacker_model=attacker_model,
        attacker_base_url=attacker_base_url,
        attacker_api_key_env=attacker_api_key_env or None,
        attacker_temperature=attacker_temperature,
        pattern_detectors_enabled=pattern_detectors_enabled,
        openai_moderation_enabled=openai_moderation_enabled,
        openai_moderation_model=openai_moderation_model,
        openai_moderation_api_key_env=openai_moderation_api_key_env or None,
        openai_moderation_base_url=openai_moderation_base_url or None,
        perspective_enabled=perspective_enabled,
        perspective_api_key_env=perspective_api_key_env or None,
        perspective_threshold=perspective_threshold,
        harmjudge_enabled=harmjudge_enabled,
        harmjudge_model=harmjudge_model or None,
        harmjudge_api_key_env=harmjudge_api_key_env or None,
        harmjudge_base_url=harmjudge_base_url or None,
        external_detector_url=external_detector_url or None,
        external_detector_api_key_env=external_detector_api_key_env or None,
        detector_weight=detector_weight,
        detector_family_enabled=detector_family_enabled,
        detector_family_weights=detector_family_weights,
    )


def _initialize_judge_provider_state() -> None:
    defaults = provider_views.PROVIDER_SPECS["ollama"]
    session_defaults = {
        JUDGE_PROVIDER_FIELD_KEYS["provider_name"]: "ollama",
        JUDGE_PROVIDER_FIELD_KEYS["auth_mode"]: "Use Environment Variable",
        JUDGE_PROVIDER_FIELD_KEYS["model"]: defaults["default_model"],
        JUDGE_PROVIDER_FIELD_KEYS["api_key"]: "",
        JUDGE_PROVIDER_FIELD_KEYS["api_key_env"]: "",
        JUDGE_PROVIDER_FIELD_KEYS["base_url"]: defaults.get(
            "default_base_url",
            "",
        ),
        JUDGE_PROVIDER_FIELD_KEYS["api_version"]: "",
        JUDGE_PROVIDER_FIELD_KEYS["aws_region"]: "",
        JUDGE_PROVIDER_FIELD_KEYS["project_id"]: "",
        JUDGE_PROVIDER_FIELD_KEYS["location"]: "",
        JUDGE_MODEL_SELECT_KEY: defaults["default_model"],
        JUDGE_MODEL_MANUAL_KEY: False,
        JUDGE_TEST_SYSTEM_PROMPT_KEY: (
            "You are a connectivity probe. Reply with the single word OK."
        ),
        JUDGE_TEST_USER_MESSAGE_KEY: "Reply with OK.",
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)


def _render_judge_provider_selector() -> ProviderConfig:
    provider_specs = provider_views._provider_specs_with_plugins()
    provider_name = st.selectbox(
        "Judge provider",
        options=list(provider_specs),
        format_func=lambda name: provider_specs[name]["label"],
        key=JUDGE_PROVIDER_FIELD_KEYS["provider_name"],
        on_change=_on_judge_provider_change,
    )
    spec = provider_specs[provider_name]

    api_key = None
    api_key_env = None
    if spec.get("supports_api_key"):
        api_mode = st.radio(
            "How to provide the judge API key",
            ["Use Environment Variable", "Paste API Key"],
            horizontal=True,
            key=JUDGE_PROVIDER_FIELD_KEYS["auth_mode"],
        )
        if api_mode == "Use Environment Variable":
            api_key_env = st.text_input(
                "Judge API key environment variable",
                key=JUDGE_PROVIDER_FIELD_KEYS["api_key_env"],
                placeholder=spec.get("default_api_key_env", ""),
            )
        else:
            api_key = st.text_input(
                "Judge API key",
                type="password",
                key=JUDGE_PROVIDER_FIELD_KEYS["api_key"],
            )
    else:
        api_mode = "Use Environment Variable"

    base_url = None
    if spec.get("supports_base_url"):
        base_url = st.text_input(
            "Judge API base URL",
            key=JUDGE_PROVIDER_FIELD_KEYS["base_url"],
            placeholder=spec.get("default_base_url", ""),
        ) or None

    api_version = None
    if provider_name == "azure-openai":
        api_version = st.text_input(
            "Judge API version",
            key=JUDGE_PROVIDER_FIELD_KEYS["api_version"],
            placeholder="2024-10-21",
        ) or None

    aws_region = None
    if provider_name == "bedrock":
        aws_region = st.text_input(
            "Judge AWS region",
            key=JUDGE_PROVIDER_FIELD_KEYS["aws_region"],
            placeholder="us-east-1",
        ) or None

    project_id = None
    location = None
    if provider_name == "vertex-ai":
        project_cols = st.columns(2)
        with project_cols[0]:
            project_id = st.text_input(
                "Judge GCP project ID",
                key=JUDGE_PROVIDER_FIELD_KEYS["project_id"],
                placeholder="my-gcp-project",
            ) or None
        with project_cols[1]:
            location = st.text_input(
                "Judge region",
                key=JUDGE_PROVIDER_FIELD_KEYS["location"],
                placeholder="us-central1",
            ) or None

    model = _render_judge_model_selector(
        provider_name=provider_name,
        spec=spec,
        auth_mode=api_mode,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        api_version=api_version,
        aws_region=aws_region,
        project_id=project_id,
        location=location,
    )

    provider_config = ProviderConfig(
        name=provider_name,
        model=model or None,
        api_key=api_key or None,
        api_key_env=api_key_env or None,
        base_url=base_url,
        api_version=api_version,
        aws_region=aws_region,
        project_id=project_id,
        location=location,
    )
    _render_judge_test_panel(provider_config)
    return provider_config


def _render_judge_model_selector(
    *,
    provider_name: str,
    spec: dict[str, object],
    auth_mode: str,
    api_key: str | None,
    api_key_env: str | None,
    base_url: str | None,
    api_version: str | None,
    aws_region: str | None,
    project_id: str | None,
    location: str | None,
) -> str:
    supports_dynamic_options = provider_views._supports_dynamic_model_options(
        provider_name,
        "chat",
    )
    default_value = (
        "" if supports_dynamic_options else str(spec.get("default_model", ""))
    )
    model_options, fetch_error = provider_views._get_provider_model_options(
        provider_name=provider_name,
        spec=spec,
        model_kind="chat",
        auth_mode=auth_mode,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        api_version=api_version,
        aws_region=aws_region,
        project_id=project_id,
        location=location,
    )
    current_model = (
        st.session_state.get(JUDGE_PROVIDER_FIELD_KEYS["model"])
        or default_value
    )

    if supports_dynamic_options:
        option_values = [item["value"] for item in model_options]
        if (
            model_options
            and current_model
            and current_model not in option_values
        ):
            st.session_state[JUDGE_MODEL_MANUAL_KEY] = True
        manual_model = st.checkbox(
            "Type judge model name manually",
            key=JUDGE_MODEL_MANUAL_KEY,
            help=(
                "Use this if the judge model ID is missing from the fetched "
                "list."
            ),
        )
        if not manual_model:
            if model_options:
                option_labels = {
                    item["value"]: item["label"] for item in model_options
                }
                selected_value = st.session_state.get(JUDGE_MODEL_SELECT_KEY)
                if selected_value not in option_values:
                    selected_value = current_model or option_values[0]
                    st.session_state[JUDGE_MODEL_SELECT_KEY] = selected_value
                selected_model = st.selectbox(
                    "Judge model",
                    options=option_values,
                    key=JUDGE_MODEL_SELECT_KEY,
                    format_func=lambda value: option_labels.get(value, value),
                )
                st.session_state[JUDGE_PROVIDER_FIELD_KEYS["model"]] = (
                    selected_model
                )
                return selected_model

            placeholder_option = (
                provider_views._get_model_discovery_placeholder(
                    provider_name=provider_name,
                    spec=spec,
                    model_kind="chat",
                    auth_mode=auth_mode,
                    api_key=api_key,
                    api_key_env=api_key_env,
                    base_url=base_url,
                    api_version=api_version,
                    aws_region=aws_region,
                    project_id=project_id,
                    location=location,
                    fetch_error=fetch_error,
                )
            )
            st.selectbox(
                "Judge model",
                options=[placeholder_option],
                key=f"{JUDGE_MODEL_SELECT_KEY}_placeholder",
                disabled=True,
            )
            return current_model or ""

    model = st.text_input(
        "Judge model",
        key=JUDGE_PROVIDER_FIELD_KEYS["model"],
    )
    if fetch_error:
        st.caption(
            f"Could not load the judge model list automatically: {fetch_error}"
        )
    return model


def _on_judge_provider_change() -> None:
    provider_specs = provider_views._provider_specs_with_plugins()
    provider_name = st.session_state.get(
        JUDGE_PROVIDER_FIELD_KEYS["provider_name"],
        "ollama",
    )
    spec = provider_specs.get(
        provider_name,
        provider_views.PROVIDER_SPECS["ollama"],
    )
    default_model = (
        ""
        if provider_views._supports_dynamic_model_options(
            provider_name,
            "chat",
        )
        else spec.get("default_model", "")
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["auth_mode"]] = (
        "Use Environment Variable"
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["model"]] = default_model
    st.session_state[JUDGE_MODEL_SELECT_KEY] = default_model
    st.session_state[JUDGE_MODEL_MANUAL_KEY] = False
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["api_key"]] = ""
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["api_key_env"]] = spec.get(
        "default_api_key_env",
        "",
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["base_url"]] = spec.get(
        "default_base_url",
        "",
    )
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["api_version"]] = ""
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["aws_region"]] = ""
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["project_id"]] = ""
    st.session_state[JUDGE_PROVIDER_FIELD_KEYS["location"]] = ""
    st.session_state.pop(JUDGE_TEST_RESULT_KEY, None)


def _render_judge_test_panel(provider_config: ProviderConfig) -> None:
    st.caption(
        "Optionally verify the judge provider before running a benchmark."
    )
    system_prompt = st.text_area(
        "Judge test system prompt",
        key=JUDGE_TEST_SYSTEM_PROMPT_KEY,
        height=80,
    )
    user_message = st.text_input(
        "Judge test user message",
        key=JUDGE_TEST_USER_MESSAGE_KEY,
    )

    if st.button("Test Judge Connection", key="eval_run_judge_test"):
        try:
            provider = create_provider(provider_config)
            response_text, tokens, latency = provider.call(
                system_prompt,
                user_message,
            )
            st.session_state[JUDGE_TEST_RESULT_KEY] = {
                "status": "ok",
                "provider": provider.get_model_name(),
                "response": response_text,
                "tokens": tokens,
                "latency": latency,
            }
        except Exception as exc:
            st.session_state[JUDGE_TEST_RESULT_KEY] = {
                "status": "error",
                "error": str(exc),
            }

    test_result = st.session_state.get(JUDGE_TEST_RESULT_KEY)
    if not test_result:
        return

    if test_result.get("status") == "ok":
        st.success(
            f"Connected to {test_result.get('provider', 'judge')} in "
            f"{test_result.get('latency', 0.0):.2f}s"
        )
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.metric("Latency", f"{test_result.get('latency', 0.0):.2f}s")
        with metric_cols[1]:
            st.metric("Tokens", int(test_result.get("tokens", 0)))
        st.code(test_result.get("response", ""))
    else:
        st.error(
            f"Judge test failed: {test_result.get('error', 'unknown error')}"
        )
