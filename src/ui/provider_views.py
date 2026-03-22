"""Streamlit provider configuration views."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from src.config import ProviderConfig
from src.plugins.manager import get_plugin_manager


PROVIDER_SPECS = {
    "ollama": {
        "label": "Ollama (Local)",
        "default_model": "qwen3.5:9b",
        "default_base_url": "http://localhost:11434",
        "supports_api_key": False,
        "supports_base_url": True,
        "transport": "ollama-chat",
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-5-1",
        "default_api_key_env": "OPENAI_API_KEY",
        "supports_api_key": True,
        "supports_base_url": True,
        "transport": "openai-chat-completions",
    },
    "openai-compatible": {
        "label": "OpenAI-Compatible",
        "default_model": "gpt-4o-mini",
        "default_api_key_env": "OPENAI_API_KEY",
        "default_base_url": "https://api.openai.com/v1",
        "supports_api_key": True,
        "supports_base_url": True,
        "transport": "openai-chat-completions",
    },
    "azure-openai": {
        "label": "Azure OpenAI",
        "default_model": "gpt-4o",
        "default_api_key_env": "AZURE_OPENAI_API_KEY",
        "default_base_url": "https://your-resource.openai.azure.com/",
        "supports_api_key": True,
        "supports_base_url": True,
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
        "default_model": "grok-beta",
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
        "default_api_key_env": "COHERE_API_KEY",
        "supports_api_key": True,
        "supports_base_url": False,
        "transport": "cohere-chat",
    },
    "together": {
        "label": "Together",
        "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "default_api_key_env": "TOGETHER_API_KEY",
        "default_base_url": "https://api.together.xyz/v1",
        "supports_api_key": True,
        "supports_base_url": True,
        "transport": "openai-chat-completions",
    },
    "mistral": {
        "label": "Mistral",
        "default_model": "mistral-large-latest",
        "default_api_key_env": "MISTRAL_API_KEY",
        "default_base_url": "https://api.mistral.ai/v1",
        "supports_api_key": True,
        "supports_base_url": True,
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
        "transport": "openai-chat-completions",
    },
    "fireworks": {
        "label": "Fireworks",
        "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "default_api_key_env": "FIREWORKS_API_KEY",
        "default_base_url": "https://api.fireworks.ai/inference/v1",
        "supports_api_key": True,
        "supports_base_url": True,
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
PROVIDER_FIELD_KEYS = {
    "provider_name": "provider_name_select",
    "auth_mode": "provider_auth_mode",
    "model": "provider_model",
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

    presets = list_provider_presets()
    preset_options = ["None"] + presets

    def _on_preset_change() -> None:
        """Auto-load provider preset when selection changes."""
        chosen = st.session_state.get("provider_preset_select", "None")
        if chosen and chosen != "None":
            preset = load_provider_preset(chosen)
            _apply_preset_to_session(preset)

    st.selectbox(
        "Preset",
        options=preset_options,
        key="provider_preset_select",
        help="Select a saved provider preset to apply automatically.",
        on_change=_on_preset_change,
    )

    provider_name = st.selectbox(
        "Provider",
        options=list(provider_specs),
        format_func=lambda name: provider_specs[name]["label"],
        key=PROVIDER_FIELD_KEYS["provider_name"],
    )
    spec = provider_specs[provider_name]

    if not st.session_state.get(PROVIDER_FIELD_KEYS["model"]):
        st.session_state[PROVIDER_FIELD_KEYS["model"]] = spec.get("default_model", "")
    model = st.text_input("Model", key=PROVIDER_FIELD_KEYS["model"])
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
            "Credential Source",
            ["Environment Variable", "Paste API Key"],
            horizontal=True,
            key=PROVIDER_FIELD_KEYS["auth_mode"],
        )
        if api_mode == "Environment Variable":
            api_key_env = st.text_input(
                "API Key Env Var",
                key=PROVIDER_FIELD_KEYS["api_key_env"],
                placeholder=spec.get("default_api_key_env", ""),
            )
            if not api_key_env and spec.get("default_api_key_env"):
                st.caption(f"Suggested: `{spec['default_api_key_env']}`")
        else:
            api_key = st.text_input("API Key", type="password", key=PROVIDER_FIELD_KEYS["api_key"])

    if spec.get("supports_base_url"):
        base_url = st.text_input(
            "Base URL",
            key=PROVIDER_FIELD_KEYS["base_url"],
            placeholder=spec.get("default_base_url", ""),
        ) or None

    if provider_name == "azure-openai":
        api_version = st.text_input(
            "API Version",
            key=PROVIDER_FIELD_KEYS["api_version"],
            placeholder="2024-10-21",
        ) or None

    if provider_name == "bedrock":
        aws_region = st.text_input(
            "AWS Region",
            key=PROVIDER_FIELD_KEYS["aws_region"],
            placeholder="us-east-1",
        ) or None

    if provider_name == "vertex-ai":
        project_cols = st.columns(2)
        with project_cols[0]:
            project_id = st.text_input(
                "Project ID",
                key=PROVIDER_FIELD_KEYS["project_id"],
                placeholder="my-gcp-project",
            ) or None
        with project_cols[1]:
            location = st.text_input(
                "Location",
                key=PROVIDER_FIELD_KEYS["location"],
                placeholder="us-central1",
            ) or None

    if provider_name == "custom-http":
        st.caption("Configure a JSON endpoint using request and response mappings.")
        header_lines = st.text_area(
            "Headers",
            height=100,
            help="One KEY=VALUE header per line. You can use {{api_key}} and {{model}} placeholders.",
            key=PROVIDER_FIELD_KEYS["custom_headers"],
        )
        try:
            headers = _parse_header_lines(header_lines)
        except ValueError as exc:
            validation_errors.append(str(exc))
        request_template_text = st.text_area(
            "Request Template (JSON)",
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
                "Response Text Path",
                help="Dot path to the generated text in the JSON response.",
                key=PROVIDER_FIELD_KEYS["response_text_path"],
            )
        with response_cols[1]:
            response_tokens_path = st.text_input(
                "Response Tokens Path",
                help="Optional dot path to total tokens in the JSON response.",
                key=PROVIDER_FIELD_KEYS["response_tokens_path"],
            ) or None

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

    st.divider()
    preset_name = st.text_input(
        "Save Current Config As",
        value=provider_name,
        key="provider_preset_name",
        help="Saves provider settings under config/providers/. Raw pasted API keys are not stored.",
    )
    if st.button("Save Preset", key="provider_save_preset"):
        saved_name = save_provider_preset(
            preset_name,
            provider_config,
            auth_mode=st.session_state.get(PROVIDER_FIELD_KEYS["auth_mode"], "Environment Variable"),
        )
        if provider_config.api_key:
            st.warning("Preset saved without the pasted API key. Re-enter it after loading, or switch to env-var mode.")
        st.success(f"Saved preset to config/providers/{saved_name}.json")

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
        PROVIDER_FIELD_KEYS["auth_mode"]: "Environment Variable",
        PROVIDER_FIELD_KEYS["model"]: PROVIDER_SPECS["ollama"]["default_model"],
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
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _apply_preset_to_session(preset: dict) -> None:
    provider_data = preset.get("provider", {})
    provider_name = provider_data.get("name", "ollama")
    provider_specs = _provider_specs_with_plugins()
    spec = provider_specs.get(provider_name, provider_specs["ollama"])
    st.session_state[PROVIDER_FIELD_KEYS["provider_name"]] = provider_name
    st.session_state[PROVIDER_FIELD_KEYS["auth_mode"]] = preset.get("auth_mode", "Environment Variable")
    st.session_state[PROVIDER_FIELD_KEYS["model"]] = provider_data.get("model") or spec.get("default_model", "")
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


def _render_provider_test_panel(provider_config: ProviderConfig, capabilities: dict) -> None:
    from src.providers.run_benchmark import create_provider

    st.divider()
    with st.expander("Test Connection", expanded=False):
        st.caption("Run a lightweight probe request before starting the full benchmark.")
        system_prompt = st.text_area(
            "Probe System Prompt",
            key=PROVIDER_FIELD_KEYS["test_system_prompt"],
            height=80,
        )
        user_message = st.text_input(
            "Probe User Message",
            key=PROVIDER_FIELD_KEYS["test_user_message"],
        )

        if st.button("Run Provider Test", key="provider_run_test"):
            if capabilities.get("validation_errors"):
                st.error("Fix provider configuration errors before running the provider test.")
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
                    "auth_mode": st.session_state.get(PROVIDER_FIELD_KEYS["auth_mode"], "Environment Variable"),
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
            if st.button("Save As Preset After Successful Test", key="provider_save_after_test"):
                preset_base_name = st.session_state.get("provider_preset_name") or provider_config.name
                saved_name = save_provider_preset(
                    preset_base_name,
                    ProviderConfig(**test_result.get("provider_config", {})),
                    auth_mode=test_result.get("auth_mode", "Environment Variable"),
                )
                st.success(f"Saved preset to config/providers/{saved_name}.json")
        else:
            st.error(f"Provider test failed: {test_result.get('error', 'unknown error')}")


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
