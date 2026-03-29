"""Streamlit provider configuration views."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from src.config import ProviderConfig
from src.plugins.manager import get_plugin_manager


PROVIDER_SPECS = {
    "ollama": {
        "label": "Ollama (Local)",
        "default_model": "qwen3.5:9b",
        "default_embedding_model": "nomic-embed-text",
        "default_base_url": "http://localhost:11434",
        "supports_api_key": False,
        "supports_base_url": True,
        "supports_embeddings": True,
        "transport": "ollama-chat",
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-5-1",
        "default_embedding_model": "text-embedding-3-small",
        "default_api_key_env": "OPENAI_API_KEY",
        "default_base_url": "https://api.openai.com/v1",
        "supports_api_key": True,
        "supports_base_url": True,
        "supports_embeddings": True,
        "transport": "openai-chat-completions",
    },
    "openai-compatible": {
        "label": "OpenAI-Compatible",
        "default_model": "gpt-4o-mini",
        "default_api_key_env": "OPENAI_API_KEY",
        "default_base_url": "https://api.openai.com/v1",
        "supports_api_key": True,
        "supports_base_url": True,
        "supports_embeddings": True,
        "transport": "openai-chat-completions",
    },
    "azure-openai": {
        "label": "Azure OpenAI",
        "default_model": "gpt-4o",
        "default_api_key_env": "AZURE_OPENAI_API_KEY",
        "default_base_url": "https://your-resource.openai.azure.com/",
        "supports_api_key": True,
        "supports_base_url": True,
        "supports_embeddings": True,
        "transport": "azure-openai-chat-completions",
    },
    "anthropic": {
        "label": "Anthropic",
        "default_model": "claude-3-5-sonnet-20241022",
        "default_api_key_env": "ANTHROPIC_API_KEY",
        "supports_api_key": True,
        "supports_base_url": False,
        "transport": "anthropic-messages",
    },
    "grok": {
        "label": "Grok",
        "default_model": "grok-4",
        "default_api_key_env": "XAI_API_KEY",
        "default_base_url": "https://api.x.ai/v1",
        "supports_api_key": True,
        "supports_base_url": True,
        "transport": "openai-chat-completions",
    },
    "groq": {
        "label": "Groq",
        "default_model": "llama-3.3-70b-versatile",
        "default_api_key_env": "GROQ_API_KEY",
        "default_base_url": "https://api.groq.com/openai/v1",
        "supports_api_key": True,
        "supports_base_url": True,
        "supports_embeddings": True,
        "transport": "openai-chat-completions",
    },
    "gemini": {
        "label": "Gemini",
        "default_model": "gemini-1.5-pro",
        "default_api_key_env": "GOOGLE_API_KEY",
        "supports_api_key": True,
        "supports_base_url": False,
        "transport": "gemini-generate-content",
    },
    "cohere": {
        "label": "Cohere",
        "default_model": "command-r-plus",
        "default_embedding_model": "embed-english-v3.0",
        "default_rerank_model": "rerank-v3.5",
        "default_api_key_env": "COHERE_API_KEY",
        "supports_api_key": True,
        "supports_base_url": False,
        "supports_embeddings": True,
        "supports_rerank": True,
        "transport": "cohere-chat",
    },
    "together": {
        "label": "Together",
        "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "default_api_key_env": "TOGETHER_API_KEY",
        "default_base_url": "https://api.together.xyz/v1",
        "supports_api_key": True,
        "supports_base_url": True,
        "supports_embeddings": True,
        "transport": "openai-chat-completions",
    },
    "mistral": {
        "label": "Mistral",
        "default_model": "mistral-large-latest",
        "default_api_key_env": "MISTRAL_API_KEY",
        "default_base_url": "https://api.mistral.ai/v1",
        "supports_api_key": True,
        "supports_base_url": True,
        "supports_embeddings": True,
        "transport": "openai-chat-completions",
    },
    "vertex-ai": {
        "label": "Vertex AI",
        "default_model": "gemini-1.5-pro",
        "supports_api_key": False,
        "supports_base_url": False,
        "transport": "vertex-ai-gemini",
    },
    "openrouter": {
        "label": "OpenRouter",
        "default_model": "openai/gpt-4o-mini",
        "default_api_key_env": "OPENROUTER_API_KEY",
        "default_base_url": "https://openrouter.ai/api/v1",
        "supports_api_key": True,
        "supports_base_url": True,
        "supports_embeddings": True,
        "transport": "openai-chat-completions",
    },
    "fireworks": {
        "label": "Fireworks",
        "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "default_api_key_env": "FIREWORKS_API_KEY",
        "default_base_url": "https://api.fireworks.ai/inference/v1",
        "supports_api_key": True,
        "supports_base_url": True,
        "supports_embeddings": True,
        "transport": "openai-chat-completions",
    },
    "bedrock": {
        "label": "AWS Bedrock",
        "default_model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "supports_api_key": False,
        "supports_base_url": False,
        "transport": "aws-bedrock-invoke-model",
    },
    "custom-http": {
        "label": "Custom HTTP Endpoint",
        "default_model": "custom-http",
        "supports_api_key": True,
        "supports_base_url": True,
        "transport": "custom-http-json",
    },
}

PRESET_DIR = Path("config/providers")
PROVIDER_TEST_SESSION_KEY = "provider_test_result"
AUTH_MODE_ENV = "Use Environment Variable"
AUTH_MODE_PASTE = "Paste API Key"
LEGACY_AUTH_MODE_ENV = "Environment Variable"
PROVIDER_MODEL_SELECT_KEY = "provider_model_select"
PROVIDER_MODEL_MANUAL_KEY = "provider_model_manual"
PROVIDER_EMBEDDING_MODEL_SELECT_KEY = "provider_embedding_model_select"
PROVIDER_EMBEDDING_MODEL_MANUAL_KEY = "provider_embedding_model_manual"
PROVIDER_RERANK_MODEL_SELECT_KEY = "provider_rerank_model_select"
PROVIDER_RERANK_MODEL_MANUAL_KEY = "provider_rerank_model_manual"
PROVIDER_MODEL_CACHE_KEY = "provider_model_cache"
MODEL_FIELD_CONFIG = {
    "chat": {
        "field_key": "model",
        "select_key": PROVIDER_MODEL_SELECT_KEY,
        "manual_key": PROVIDER_MODEL_MANUAL_KEY,
        "label": "Model",
        "manual_label": "Type model name manually",
        "default_spec_key": "default_model",
    },
    "embedding": {
        "field_key": "embedding_model",
        "select_key": PROVIDER_EMBEDDING_MODEL_SELECT_KEY,
        "manual_key": PROVIDER_EMBEDDING_MODEL_MANUAL_KEY,
        "label": "Embedding model",
        "manual_label": "Type embedding model manually",
        "default_spec_key": "default_embedding_model",
    },
    "rerank": {
        "field_key": "rerank_model",
        "select_key": PROVIDER_RERANK_MODEL_SELECT_KEY,
        "manual_key": PROVIDER_RERANK_MODEL_MANUAL_KEY,
        "label": "Rerank model",
        "manual_label": "Type rerank model manually",
        "default_spec_key": "default_rerank_model",
    },
}
PROVIDER_FIELD_KEYS = {
    "provider_name": "provider_name_select",
    "auth_mode": "provider_auth_mode",
    "model": "provider_model",
    "embedding_model": "provider_embedding_model",
    "rerank_model": "provider_rerank_model",
    "api_key": "provider_api_key",
    "api_key_env": "provider_api_key_env",
    "base_url": "provider_base_url",
    "api_version": "provider_api_version",
    "aws_region": "provider_aws_region",
    "project_id": "provider_project_id",
    "location": "provider_location",
    "custom_headers": "provider_custom_headers",
    "request_template": "provider_request_template",
    "response_text_path": "provider_response_text_path",
    "response_tokens_path": "provider_response_tokens_path",
    "test_system_prompt": "provider_test_system_prompt",
    "test_user_message": "provider_test_user_message",
}


def render_provider_selector() -> tuple[ProviderConfig, dict]:
    _initialize_provider_state()
    provider_specs = _provider_specs_with_plugins()
    st.session_state[PROVIDER_FIELD_KEYS["auth_mode"]] = _normalize_auth_mode(
        st.session_state.get(PROVIDER_FIELD_KEYS["auth_mode"])
    )

    provider_name = st.selectbox(
        "Provider",
        options=list(provider_specs),
        format_func=lambda name: provider_specs[name]["label"],
        key=PROVIDER_FIELD_KEYS["provider_name"],
        on_change=_on_provider_change,
    )
    spec = provider_specs[provider_name]

    api_key = None
    api_key_env = None
    base_url = None
    headers = {}
    request_template = {}
    response_text_path = None
    response_tokens_path = None
    validation_errors = []
    api_version = None
    aws_region = None
    project_id = None
    location = None

    if spec.get("supports_api_key"):
        api_mode = st.radio(
            "How to provide the API key",
            [AUTH_MODE_ENV, AUTH_MODE_PASTE],
            horizontal=True,
            key=PROVIDER_FIELD_KEYS["auth_mode"],
        )
        if _normalize_auth_mode(api_mode) == AUTH_MODE_ENV:
            api_key_env = st.text_input(
                "API key environment variable",
                key=PROVIDER_FIELD_KEYS["api_key_env"],
                placeholder=spec.get("default_api_key_env", ""),
            )
        else:
            api_key = st.text_input(
                "API key",
                type="password",
                key=PROVIDER_FIELD_KEYS["api_key"],
            )

    if spec.get("supports_base_url"):
        base_url = st.text_input(
            "API base URL",
            key=PROVIDER_FIELD_KEYS["base_url"],
            placeholder=spec.get("default_base_url", ""),
        ) or None

    if provider_name == "azure-openai":
        api_version = st.text_input(
            "API version",
            key=PROVIDER_FIELD_KEYS["api_version"],
            placeholder="2024-10-21",
        ) or None

    if provider_name == "bedrock":
        aws_region = st.text_input(
            "AWS region",
            key=PROVIDER_FIELD_KEYS["aws_region"],
            placeholder="us-east-1",
        ) or None

    if provider_name == "vertex-ai":
        project_cols = st.columns(2)
        with project_cols[0]:
            project_id = st.text_input(
                "GCP project ID",
                key=PROVIDER_FIELD_KEYS["project_id"],
                placeholder="my-gcp-project",
            ) or None
        with project_cols[1]:
            location = st.text_input(
                "Region",
                key=PROVIDER_FIELD_KEYS["location"],
                placeholder="us-central1",
            ) or None

    if provider_name == "custom-http":
        st.caption(
            "Configure a custom JSON endpoint with request and response mappings."
        )
        header_lines = st.text_area(
            "HTTP headers",
            height=100,
            help=(
                "One KEY=VALUE header per line. You can use {{api_key}} "
                "and {{model}} placeholders."
            ),
            key=PROVIDER_FIELD_KEYS["custom_headers"],
        )
        try:
            headers = _parse_header_lines(header_lines)
        except ValueError as exc:
            validation_errors.append(str(exc))
        request_template_text = st.text_area(
            "Request JSON template",
            height=180,
            key=PROVIDER_FIELD_KEYS["request_template"],
        )
        try:
            request_template = _safe_json_load(request_template_text, "request template")
        except ValueError as exc:
            validation_errors.append(str(exc))
        response_cols = st.columns(2)
        with response_cols[0]:
            response_text_path = st.text_input(
                "Response text field",
                help="Dot path to the generated text in the JSON response.",
                key=PROVIDER_FIELD_KEYS["response_text_path"],
            )
        with response_cols[1]:
            response_tokens_path = st.text_input(
                "Response tokens field",
                help="Optional dot path to total tokens in the JSON response.",
                key=PROVIDER_FIELD_KEYS["response_tokens_path"],
            ) or None

    model = _render_model_selector(
        provider_name=provider_name,
        spec=spec,
        model_kind="chat",
        auth_mode=st.session_state.get(
            PROVIDER_FIELD_KEYS["auth_mode"],
            AUTH_MODE_ENV,
        ),
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        api_version=api_version,
        aws_region=aws_region,
        project_id=project_id,
        location=location,
    )
    embedding_model = None
    rerank_model = None
    if spec.get("supports_embeddings"):
        embedding_model = _render_model_selector(
            provider_name=provider_name,
            spec=spec,
            model_kind="embedding",
            auth_mode=st.session_state.get(
                PROVIDER_FIELD_KEYS["auth_mode"],
                AUTH_MODE_ENV,
            ),
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            api_version=api_version,
            aws_region=aws_region,
            project_id=project_id,
            location=location,
        )
    if spec.get("supports_rerank"):
        rerank_model = _render_model_selector(
            provider_name=provider_name,
            spec=spec,
            model_kind="rerank",
            auth_mode=st.session_state.get(
                PROVIDER_FIELD_KEYS["auth_mode"],
                AUTH_MODE_ENV,
            ),
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
        embedding_model=embedding_model or None,
        rerank_model=rerank_model or None,
        api_key=api_key or None,
        api_key_env=api_key_env or None,
        base_url=base_url,
        api_version=api_version,
        aws_region=aws_region,
        project_id=project_id,
        location=location,
        headers=headers,
        request_template=request_template,
        response_text_path=response_text_path or None,
        response_tokens_path=response_tokens_path,
    )
    capabilities = {
        "label": spec["label"],
        "transport": spec["transport"],
        "supports_api_key": spec.get("supports_api_key", False),
        "supports_base_url": spec.get("supports_base_url", False),
        "validation_errors": validation_errors,
    }
    for error in validation_errors:
        st.error(error)

    _sync_provider_test_result(provider_config)
    _render_provider_test_panel(provider_config, capabilities)
    return provider_config, capabilities


def list_provider_presets() -> list[str]:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path.stem for path in PRESET_DIR.glob("*.json"))


def load_provider_preset(name: str) -> dict:
    preset_path = PRESET_DIR / f"{name}.json"
    with preset_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_provider_preset(name: str, provider_config: ProviderConfig, auth_mode: str) -> str:
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _slugify_preset_name(name) or "provider-preset"
    preset_path = PRESET_DIR / f"{safe_name}.json"
    payload = {
        "version": 1,
        "auth_mode": auth_mode,
        "provider": {
            **asdict(provider_config),
            "api_key": None,
        },
    }
    with preset_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return safe_name


def delete_provider_preset(name: str) -> None:
    preset_path = PRESET_DIR / f"{name}.json"
    if preset_path.exists():
        preset_path.unlink()


def _initialize_provider_state() -> None:
    defaults = {
        PROVIDER_FIELD_KEYS["provider_name"]: "ollama",
        PROVIDER_FIELD_KEYS["auth_mode"]: AUTH_MODE_ENV,
        PROVIDER_FIELD_KEYS["model"]: PROVIDER_SPECS["ollama"]["default_model"],
        PROVIDER_FIELD_KEYS["embedding_model"]: PROVIDER_SPECS["ollama"].get(
            "default_embedding_model",
            "",
        ),
        PROVIDER_FIELD_KEYS["rerank_model"]: PROVIDER_SPECS["ollama"].get(
            "default_rerank_model",
            "",
        ),
        PROVIDER_FIELD_KEYS["api_key"]: "",
        PROVIDER_FIELD_KEYS["api_key_env"]: "",
        PROVIDER_FIELD_KEYS["base_url"]: PROVIDER_SPECS["ollama"]["default_base_url"],
        PROVIDER_FIELD_KEYS["api_version"]: "",
        PROVIDER_FIELD_KEYS["aws_region"]: "",
        PROVIDER_FIELD_KEYS["project_id"]: "",
        PROVIDER_FIELD_KEYS["location"]: "",
        PROVIDER_FIELD_KEYS["custom_headers"]: "",
        PROVIDER_FIELD_KEYS["request_template"]: json.dumps(
            {
                "model": "{{model}}",
                "system_prompt": "{{system_prompt}}",
                "input": "{{user_message}}",
            },
            indent=2,
        ),
        PROVIDER_FIELD_KEYS["response_text_path"]: "text",
        PROVIDER_FIELD_KEYS["response_tokens_path"]: "",
        PROVIDER_FIELD_KEYS["test_system_prompt"]: "You are a connectivity probe. Reply with the single word OK.",
        PROVIDER_FIELD_KEYS["test_user_message"]: "Reply with OK.",
        PROVIDER_MODEL_SELECT_KEY: PROVIDER_SPECS["ollama"]["default_model"],
        PROVIDER_MODEL_MANUAL_KEY: False,
        PROVIDER_EMBEDDING_MODEL_SELECT_KEY: PROVIDER_SPECS["ollama"].get(
            "default_embedding_model",
            "",
        ),
        PROVIDER_EMBEDDING_MODEL_MANUAL_KEY: False,
        PROVIDER_RERANK_MODEL_SELECT_KEY: PROVIDER_SPECS["ollama"].get(
            "default_rerank_model",
            "",
        ),
        PROVIDER_RERANK_MODEL_MANUAL_KEY: False,
        PROVIDER_MODEL_CACHE_KEY: {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state[PROVIDER_FIELD_KEYS["auth_mode"]] = _normalize_auth_mode(
        st.session_state.get(PROVIDER_FIELD_KEYS["auth_mode"])
    )


def _apply_preset_to_session(preset: dict) -> None:
    provider_data = preset.get("provider", {})
    provider_name = provider_data.get("name", "ollama")
    provider_specs = _provider_specs_with_plugins()
    spec = provider_specs.get(provider_name, provider_specs["ollama"])
    default_chat_model = (
        "" if _supports_dynamic_model_options(provider_name, "chat")
        else spec.get("default_model", "")
    )
    default_embedding_model = (
        "" if _supports_dynamic_model_options(provider_name, "embedding")
        else spec.get("default_embedding_model", "")
    )
    default_rerank_model = (
        "" if _supports_dynamic_model_options(provider_name, "rerank")
        else spec.get("default_rerank_model", "")
    )
    st.session_state[PROVIDER_FIELD_KEYS["provider_name"]] = provider_name
    st.session_state[PROVIDER_FIELD_KEYS["auth_mode"]] = _normalize_auth_mode(
        preset.get("auth_mode", AUTH_MODE_ENV)
    )
    st.session_state[PROVIDER_FIELD_KEYS["model"]] = (
        provider_data.get("model") or default_chat_model
    )
    st.session_state[PROVIDER_FIELD_KEYS["embedding_model"]] = (
        provider_data.get("embedding_model")
        or default_embedding_model
    )
    st.session_state[PROVIDER_FIELD_KEYS["rerank_model"]] = (
        provider_data.get("rerank_model")
        or default_rerank_model
    )
    st.session_state[PROVIDER_MODEL_SELECT_KEY] = st.session_state[PROVIDER_FIELD_KEYS["model"]]
    st.session_state[PROVIDER_MODEL_MANUAL_KEY] = False
    st.session_state[PROVIDER_EMBEDDING_MODEL_SELECT_KEY] = st.session_state[
        PROVIDER_FIELD_KEYS["embedding_model"]
    ]
    st.session_state[PROVIDER_EMBEDDING_MODEL_MANUAL_KEY] = False
    st.session_state[PROVIDER_RERANK_MODEL_SELECT_KEY] = st.session_state[
        PROVIDER_FIELD_KEYS["rerank_model"]
    ]
    st.session_state[PROVIDER_RERANK_MODEL_MANUAL_KEY] = False
    st.session_state[PROVIDER_FIELD_KEYS["api_key"]] = ""
    st.session_state[PROVIDER_FIELD_KEYS["api_key_env"]] = provider_data.get("api_key_env") or spec.get("default_api_key_env", "")
    st.session_state[PROVIDER_FIELD_KEYS["base_url"]] = provider_data.get("base_url") or spec.get("default_base_url", "")
    st.session_state[PROVIDER_FIELD_KEYS["api_version"]] = provider_data.get("api_version") or ""
    st.session_state[PROVIDER_FIELD_KEYS["aws_region"]] = provider_data.get("aws_region") or ""
    st.session_state[PROVIDER_FIELD_KEYS["project_id"]] = provider_data.get("project_id") or ""
    st.session_state[PROVIDER_FIELD_KEYS["location"]] = provider_data.get("location") or ""
    st.session_state[PROVIDER_FIELD_KEYS["custom_headers"]] = "\n".join(
        f"{key}={value}" for key, value in (provider_data.get("headers") or {}).items()
    )
    st.session_state[PROVIDER_FIELD_KEYS["request_template"]] = json.dumps(
        provider_data.get("request_template") or {
            "model": "{{model}}",
            "system_prompt": "{{system_prompt}}",
            "input": "{{user_message}}",
        },
        indent=2,
    )
    st.session_state[PROVIDER_FIELD_KEYS["response_text_path"]] = provider_data.get("response_text_path") or "text"
    st.session_state[PROVIDER_FIELD_KEYS["response_tokens_path"]] = provider_data.get("response_tokens_path") or ""


def _slugify_preset_name(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed[:80]


def _on_provider_change() -> None:
    provider_specs = _provider_specs_with_plugins()
    provider_name = st.session_state.get(PROVIDER_FIELD_KEYS["provider_name"], "ollama")
    spec = provider_specs.get(provider_name, PROVIDER_SPECS["ollama"])
    default_model = (
        "" if _supports_dynamic_model_options(provider_name, "chat")
        else spec.get("default_model", "")
    )
    default_embedding_model = (
        "" if _supports_dynamic_model_options(provider_name, "embedding")
        else spec.get("default_embedding_model", "")
    )
    default_rerank_model = (
        "" if _supports_dynamic_model_options(provider_name, "rerank")
        else spec.get("default_rerank_model", "")
    )
    st.session_state[PROVIDER_FIELD_KEYS["model"]] = default_model
    st.session_state[PROVIDER_MODEL_SELECT_KEY] = default_model
    st.session_state[PROVIDER_MODEL_MANUAL_KEY] = False
    st.session_state[PROVIDER_FIELD_KEYS["embedding_model"]] = (
        default_embedding_model
    )
    st.session_state[PROVIDER_EMBEDDING_MODEL_SELECT_KEY] = (
        default_embedding_model
    )
    st.session_state[PROVIDER_EMBEDDING_MODEL_MANUAL_KEY] = False
    st.session_state[PROVIDER_FIELD_KEYS["rerank_model"]] = (
        default_rerank_model
    )
    st.session_state[PROVIDER_RERANK_MODEL_SELECT_KEY] = default_rerank_model
    st.session_state[PROVIDER_RERANK_MODEL_MANUAL_KEY] = False
    st.session_state[PROVIDER_FIELD_KEYS["api_key"]] = ""
    st.session_state[PROVIDER_FIELD_KEYS["api_key_env"]] = spec.get(
        "default_api_key_env",
        "",
    )
    st.session_state[PROVIDER_FIELD_KEYS["base_url"]] = spec.get(
        "default_base_url",
        "",
    )
    st.session_state[PROVIDER_FIELD_KEYS["api_version"]] = ""
    st.session_state[PROVIDER_FIELD_KEYS["aws_region"]] = ""
    st.session_state[PROVIDER_FIELD_KEYS["project_id"]] = ""
    st.session_state[PROVIDER_FIELD_KEYS["location"]] = ""
    st.session_state.pop(PROVIDER_TEST_SESSION_KEY, None)


def _render_model_selector(
    *,
    provider_name: str,
    spec: dict[str, Any],
    model_kind: str,
    auth_mode: str,
    api_key: str | None,
    api_key_env: str | None,
    base_url: str | None,
    api_version: str | None,
    aws_region: str | None,
    project_id: str | None,
    location: str | None,
) -> str:
    model_config = MODEL_FIELD_CONFIG[model_kind]
    field_key = PROVIDER_FIELD_KEYS[model_config["field_key"]]
    select_key = model_config["select_key"]
    manual_key = model_config["manual_key"]
    supports_dynamic_options = _supports_dynamic_model_options(provider_name, model_kind)
    default_value = (
        "" if supports_dynamic_options
        else spec.get(model_config["default_spec_key"], "")
    )
    model_options, fetch_error = _get_provider_model_options(
        provider_name=provider_name,
        spec=spec,
        model_kind=model_kind,
        auth_mode=auth_mode,
        api_key=api_key,
        api_key_env=api_key_env,
        base_url=base_url,
        api_version=api_version,
        aws_region=aws_region,
        project_id=project_id,
        location=location,
    )
    current_model = st.session_state.get(field_key) or default_value

    if supports_dynamic_options:
        if model_options:
            option_values = [item["value"] for item in model_options]
            if current_model and current_model not in option_values:
                st.session_state[manual_key] = True
        else:
            option_values = []
        manual_model = st.checkbox(
            model_config["manual_label"],
            key=manual_key,
            help=(
                "Use this if you need a custom model ID or deployment name "
                "that is not in the fetched list."
            ),
        )
        if not manual_model:
            if model_options:
                option_labels = {
                    item["value"]: item["label"] for item in model_options
                }
                selected_value = st.session_state.get(select_key)
                if selected_value not in option_values:
                    selected_value = current_model or option_values[0]
                    st.session_state[select_key] = selected_value
                selected_model = st.selectbox(
                    model_config["label"],
                    options=option_values,
                    key=select_key,
                    format_func=lambda value: option_labels.get(value, value),
                )
                st.session_state[field_key] = selected_model
                return selected_model

            placeholder_key = f"{select_key}_placeholder"
            placeholder_option = _get_model_discovery_placeholder(
                provider_name=provider_name,
                spec=spec,
                model_kind=model_kind,
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
            st.session_state[placeholder_key] = placeholder_option
            st.selectbox(
                model_config["label"],
                options=[placeholder_option],
                key=placeholder_key,
                disabled=True,
                label_visibility="visible",
            )
            return current_model or ""

    model = st.text_input(
        model_config["label"],
        key=field_key,
        help=(
            "Leave empty to use the provider default for this capability."
            if model_kind != "chat"
            else None
        ),
    )
    if fetch_error:
        st.caption(
            f"Could not load the {model_kind} model list automatically: "
            f"{fetch_error}"
        )
    return model


def _get_model_discovery_placeholder(
    *,
    provider_name: str,
    spec: dict[str, Any],
    model_kind: str,
    auth_mode: str,
    api_key: str | None,
    api_key_env: str | None,
    base_url: str | None,
    api_version: str | None,
    aws_region: str | None,
    project_id: str | None,
    location: str | None,
    fetch_error: str | None,
) -> str:
    if fetch_error:
        return f"Could not load {model_kind} models"

    if spec.get("supports_api_key"):
        resolved_api_key = _resolve_provider_api_key(spec, auth_mode, api_key, api_key_env)
        if not resolved_api_key:
            return "Add API key to load models"

    if provider_name == "azure-openai":
        if not base_url:
            return "Add Azure endpoint to load models"
        if not api_version:
            return "Add API version to load models"
    if provider_name == "vertex-ai":
        if not project_id:
            return "Add GCP project ID to load models"
        if not location:
            return "Add region to load models"
    if provider_name == "bedrock" and not aws_region:
        return "Add AWS region to load models"

    return f"No {model_kind} models available"


def _supports_dynamic_model_options(
    provider_name: str,
    model_kind: str,
) -> bool:
    if provider_name == "custom-http":
        return False
    if model_kind == "chat":
        return provider_name in {
            "ollama",
            "openai",
            "openai-compatible",
            "azure-openai",
            "anthropic",
            "grok",
            "groq",
            "gemini",
            "cohere",
            "together",
            "mistral",
            "vertex-ai",
            "openrouter",
            "fireworks",
            "bedrock",
        }
    if model_kind == "embedding":
        return provider_name in {
            "ollama",
            "openai",
            "openai-compatible",
            "azure-openai",
            "cohere",
            "groq",
            "together",
            "mistral",
            "openrouter",
            "fireworks",
        }
    if model_kind == "rerank":
        return provider_name in {"cohere"}
    return False


def _get_provider_model_options(
    *,
    provider_name: str,
    spec: dict[str, Any],
    model_kind: str,
    auth_mode: str,
    api_key: str | None,
    api_key_env: str | None,
    base_url: str | None,
    api_version: str | None,
    aws_region: str | None,
    project_id: str | None,
    location: str | None,
) -> tuple[list[dict[str, str]], str | None]:
    cache: dict[tuple[Any, ...], tuple[list[dict[str, str]], str | None]] = st.session_state.setdefault(
        PROVIDER_MODEL_CACHE_KEY,
        {},
    )
    resolved_api_key = _resolve_provider_api_key(spec, auth_mode, api_key, api_key_env)
    cache_key = (
        provider_name,
        model_kind,
        auth_mode,
        resolved_api_key or "",
        base_url or "",
        api_version or "",
        aws_region or "",
        project_id or "",
        location or "",
    )
    if cache_key in cache:
        return cache[cache_key]

    try:
        options = _fetch_provider_model_options(
            provider_name=provider_name,
            spec=spec,
            model_kind=model_kind,
            api_key=resolved_api_key,
            base_url=base_url,
            api_version=api_version,
            aws_region=aws_region,
            project_id=project_id,
            location=location,
        )
        result = (options, None)
    except Exception as exc:
        result = ([], str(exc))

    cache[cache_key] = result
    return result


def _resolve_provider_api_key(
    spec: dict[str, Any],
    auth_mode: str,
    api_key: str | None,
    api_key_env: str | None,
) -> str | None:
    if _normalize_auth_mode(auth_mode) == AUTH_MODE_PASTE:
        return api_key or None
    env_name = api_key_env or spec.get("default_api_key_env")
    if not env_name:
        return None
    return os.getenv(env_name)


def _normalize_auth_mode(auth_mode: str | None) -> str:
    if auth_mode == LEGACY_AUTH_MODE_ENV:
        return AUTH_MODE_ENV
    if auth_mode == AUTH_MODE_PASTE:
        return AUTH_MODE_PASTE
    return AUTH_MODE_ENV


def _fetch_provider_model_options(
    *,
    provider_name: str,
    spec: dict[str, Any],
    model_kind: str,
    api_key: str | None,
    base_url: str | None,
    api_version: str | None,
    aws_region: str | None,
    project_id: str | None,
    location: str | None,
) -> list[dict[str, str]]:
    if provider_name == "ollama":
        if model_kind == "rerank":
            return []
        return _fetch_ollama_models(
            base_url
            or spec.get("default_base_url", "http://localhost:11434")
        )
    if provider_name in {
        "openai",
        "openai-compatible",
        "grok",
        "groq",
        "together",
        "mistral",
        "openrouter",
        "fireworks",
    }:
        return _fetch_openai_compatible_models(
            base_url or spec.get("default_base_url", "https://api.openai.com/v1"),
            api_key,
            model_kind=model_kind,
        )
    if provider_name == "azure-openai":
        if model_kind == "rerank":
            return []
        return _fetch_azure_openai_deployments(
            base_url,
            api_key,
            api_version or "2024-10-21",
            model_kind=model_kind,
        )
    if provider_name == "anthropic":
        if model_kind != "chat":
            return []
        return _fetch_anthropic_models(api_key)
    if provider_name == "gemini":
        if model_kind != "chat":
            return []
        return _fetch_gemini_models(api_key)
    if provider_name == "cohere":
        return _fetch_cohere_models(api_key, model_kind=model_kind)
    if provider_name == "vertex-ai":
        if model_kind != "chat":
            return []
        return _fetch_vertex_models(project_id, location)
    if provider_name == "bedrock":
        if model_kind != "chat":
            return []
        return _fetch_bedrock_models(aws_region)
    return []


def _fetch_ollama_models(base_url: str) -> list[dict[str, str]]:
    payload = _request_json(f"{base_url.rstrip('/')}/api/tags")
    models = [item.get("name") for item in payload.get("models", []) if item.get("name")]
    return _model_options_from_values(models)


def _fetch_openai_compatible_models(
    base_url: str,
    api_key: str | None,
    *,
    model_kind: str = "chat",
) -> list[dict[str, str]]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = _request_json(f"{base_url.rstrip('/')}/models", headers=headers)
    options = []
    for item in payload.get("data", []):
        model_id = item.get("id")
        if not model_id:
            continue
        if _matches_model_kind(item, str(model_id), model_kind):
            options.append({"value": str(model_id), "label": str(model_id)})
    return _sort_model_options(options)


def _fetch_azure_openai_deployments(
    base_url: str | None,
    api_key: str | None,
    api_version: str,
    *,
    model_kind: str = "chat",
) -> list[dict[str, str]]:
    if not base_url:
        return []
    if not api_key:
        return []
    payload = _request_json(
        f"{base_url.rstrip('/')}/openai/deployments?api-version={api_version}",
        headers={"api-key": api_key},
    )
    options = []
    fallback_options = []
    for item in payload.get("data", []):
        deployment_name = (
            item.get("id")
            or item.get("deployment_name")
            or item.get("name")
        )
        model_name = item.get("model") or item.get("model_name")
        if deployment_name:
            label = deployment_name
            if model_name:
                label = f"{deployment_name} ({model_name})"
            option = {"value": deployment_name, "label": label}
            fallback_options.append(option)
            if _matches_model_kind(
                item,
                f"{deployment_name} {model_name or ''}",
                model_kind,
            ):
                options.append(option)
    if model_kind == "embedding" and not options:
        return _sort_model_options(fallback_options)
    return _sort_model_options(options)


def _fetch_anthropic_models(api_key: str | None) -> list[dict[str, str]]:
    if not api_key:
        return []
    payload = _request_json(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    models = [item.get("id") for item in payload.get("data", []) if item.get("id")]
    return _model_options_from_values(models)


def _fetch_gemini_models(api_key: str | None) -> list[dict[str, str]]:
    if not api_key:
        return []
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    models = []
    for model in genai.list_models():
        methods = set(getattr(model, "supported_generation_methods", []) or [])
        if "generateContent" not in methods:
            continue
        model_name = getattr(model, "name", "")
        if model_name.startswith("models/"):
            model_name = model_name.split("/", 1)[1]
        if model_name:
            models.append(model_name)
    return _model_options_from_values(models)


def _fetch_cohere_models(
    api_key: str | None,
    *,
    model_kind: str = "chat",
) -> list[dict[str, str]]:
    if not api_key:
        return []
    payload = _request_json(
        "https://api.cohere.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    raw_models = payload.get("models") or payload.get("data") or []
    options = []
    for item in raw_models:
        model_name = item.get("name") or item.get("id")
        if not model_name:
            continue
        if _matches_model_kind(item, str(model_name), model_kind):
            options.append(
                {"value": str(model_name), "label": str(model_name)}
            )
    return _sort_model_options(options)


def _matches_model_kind(
    metadata: dict[str, Any],
    model_name: str,
    model_kind: str,
) -> bool:
    if model_kind == "chat":
        return True

    normalized_name = model_name.lower()
    tokens = set()

    def _collect_tokens(value: Any) -> None:
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                tokens.add(str(nested_key).lower())
                _collect_tokens(nested_value)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _collect_tokens(item)
            return
        if value is not None:
            tokens.add(str(value).lower())

    for key in (
        "type",
        "types",
        "capability",
        "capabilities",
        "endpoint",
        "endpoints",
        "supported_endpoints",
        "modalities",
        "input_modalities",
        "output_modalities",
        "default_endpoints",
    ):
        if key in metadata:
            _collect_tokens(metadata[key])

    haystack = " ".join(sorted(tokens | {normalized_name}))
    embedding_markers = ("embed", "embedding")
    rerank_markers = ("rerank", "re-rank")

    if model_kind == "embedding":
        return any(marker in haystack for marker in embedding_markers)
    if model_kind == "rerank":
        return any(marker in haystack for marker in rerank_markers)
    return True


def _fetch_vertex_models(
    project_id: str | None,
    location: str | None,
) -> list[dict[str, str]]:
    if not project_id or not location:
        return []
    from google.auth import default
    from google.auth.transport.requests import Request

    credentials, _ = default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    payload = _request_json(
        (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/"
            f"{project_id}/locations/{location}/publishers/google/models"
        ),
        headers={"Authorization": f"Bearer {credentials.token}"},
    )
    models = []
    for item in payload.get("publisherModels", []):
        name = item.get("name", "")
        if name:
            models.append(name.split("/")[-1])
    return _model_options_from_values(models)


def _fetch_bedrock_models(aws_region: str | None) -> list[dict[str, str]]:
    import boto3

    client = boto3.client("bedrock", region_name=aws_region or None)
    payload = client.list_foundation_models(byOutputModality="TEXT")
    models = [
        item.get("modelId")
        for item in payload.get("modelSummaries", [])
        if item.get("modelId")
    ]
    return _model_options_from_values(models)


def _request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()


def _model_options_from_values(values: list[str | None]) -> list[dict[str, str]]:
    options = [
        {"value": value, "label": value}
        for value in values
        if value
    ]
    return _sort_model_options(options)


def _sort_model_options(
    options: list[dict[str, str]],
) -> list[dict[str, str]]:
    deduped: dict[str, dict[str, str]] = {}
    for item in options:
        deduped.setdefault(item["value"], item)
    return sorted(
        deduped.values(),
        key=lambda item: item["label"].lower(),
    )


def _render_provider_test_panel(provider_config: ProviderConfig, capabilities: dict) -> None:
    from src.providers.run_benchmark import create_provider

    st.divider()
    with st.expander("Test Connection", expanded=False):
        st.caption("Run a quick test request before starting the benchmark.")
        system_prompt = st.text_area(
            "Test system prompt",
            key=PROVIDER_FIELD_KEYS["test_system_prompt"],
            height=80,
        )
        user_message = st.text_input(
            "Test user message",
            key=PROVIDER_FIELD_KEYS["test_user_message"],
        )

        if st.button("Run test", key="provider_run_test"):
            if capabilities.get("validation_errors"):
                st.error(
                    "Fix provider configuration errors before running the test."
                )
                return
            try:
                provider = create_provider(provider_config)
                response_text, tokens, latency = provider.call(system_prompt, user_message)
                st.session_state[PROVIDER_TEST_SESSION_KEY] = {
                    "status": "ok",
                    "provider": provider.get_model_name(),
                    "capabilities": provider.get_capabilities(),
                    "provider_config": {
                        **asdict(provider_config),
                        "api_key": None,
                    },
                    "auth_mode": _normalize_auth_mode(
                        st.session_state.get(
                            PROVIDER_FIELD_KEYS["auth_mode"],
                            AUTH_MODE_ENV,
                        )
                    ),
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "response": response_text,
                    "tokens": tokens,
                    "latency": latency,
                }
            except Exception as exc:
                st.session_state[PROVIDER_TEST_SESSION_KEY] = {
                    "status": "error",
                    "error": str(exc),
                }

        test_result = st.session_state.get(PROVIDER_TEST_SESSION_KEY)
        if not test_result:
            return

        if test_result.get("status") == "ok":
            st.success(
                f"Connected to {test_result.get('provider', 'provider')} "
                f"in {test_result.get('latency', 0.0):.2f}s"
            )
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Latency", f"{test_result.get('latency', 0.0):.2f}s")
            with metric_cols[1]:
                st.metric("Tokens", int(test_result.get("tokens", 0)))
            with metric_cols[2]:
                st.metric("Transport", test_result.get("capabilities", {}).get("transport", "unknown"))
            st.code(test_result.get("response", ""))
        else:
            st.error(f"Provider test failed: {test_result.get('error', 'unknown error')}")


def _sync_provider_test_result(provider_config: ProviderConfig) -> None:
    test_result = st.session_state.get(PROVIDER_TEST_SESSION_KEY)
    if not test_result:
        return

    saved_config = test_result.get("provider_config") or {}
    current_config = _serialize_provider_config(provider_config)
    saved_auth_mode = test_result.get("auth_mode")
    current_auth_mode = _normalize_auth_mode(
        st.session_state.get(PROVIDER_FIELD_KEYS["auth_mode"], AUTH_MODE_ENV)
    )
    if saved_config != current_config or _normalize_auth_mode(saved_auth_mode) != current_auth_mode:
        st.session_state.pop(PROVIDER_TEST_SESSION_KEY, None)


def _serialize_provider_config(provider_config: ProviderConfig) -> dict[str, Any]:
    return {
        **asdict(provider_config),
        "api_key": None,
    }


def _parse_header_lines(value: str) -> dict[str, str]:
    headers = {}
    for line in value.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Invalid header line: {line}")
        key, header_value = line.split("=", 1)
        headers[key.strip()] = header_value.strip()
    return headers


def _safe_json_load(value: str, label: str) -> dict:
    try:
        data = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label.capitalize()} must be a JSON object")
    return data


def _provider_specs_with_plugins() -> dict[str, dict]:
    specs = dict(PROVIDER_SPECS)
    manager = get_plugin_manager()
    for provider_name in manager.provider_names():
        if provider_name in specs:
            continue
        specs[provider_name] = {
            "label": f"Plugin: {provider_name}",
            "default_model": provider_name,
            "supports_api_key": True,
            "supports_base_url": True,
            "transport": "plugin-provider",
        }
    return specs
