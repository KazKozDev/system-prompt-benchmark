"""Streamlit evaluation settings views."""

from __future__ import annotations

import streamlit as st

from src.config import JudgeConfig
from src.core.detectors import builtin_detector_catalog


def render_evaluation_settings(
    default_pass_threshold: float, default_review_threshold: float
) -> JudgeConfig:
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
