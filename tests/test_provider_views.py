"""Tests for provider model discovery helpers."""

from __future__ import annotations

from types import SimpleNamespace


def test_fetch_openai_compatible_models_uses_models_endpoint(
    monkeypatch,
) -> None:
    from src.ui import provider_views

    calls: list[tuple[str, dict[str, str] | None, int]] = []

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"id": "gpt-4o"},
                    {"id": "gpt-4o-mini"},
                ]
            }

    def _fake_get(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 0,
    ):
        calls.append((url, headers, timeout))
        return _FakeResponse()

    monkeypatch.setattr(provider_views.requests, "get", _fake_get)

    options = provider_views._fetch_openai_compatible_models(
        "https://api.openai.com/v1",
        "secret-key",
    )

    assert calls == [
        (
            "https://api.openai.com/v1/models",
            {"Authorization": "Bearer secret-key"},
            20,
        )
    ]
    assert options == [
        {"value": "gpt-4o", "label": "gpt-4o"},
        {"value": "gpt-4o-mini", "label": "gpt-4o-mini"},
    ]


def test_fetch_openai_compatible_models_filters_embedding_models(
    monkeypatch,
) -> None:
    from src.ui import provider_views

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"id": "gpt-4o"},
                    {"id": "text-embedding-3-small"},
                    {"id": "text-embedding-3-large"},
                ]
            }

    monkeypatch.setattr(
        provider_views.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(),
    )

    options = provider_views._fetch_openai_compatible_models(
        "https://api.openai.com/v1",
        "secret-key",
        model_kind="embedding",
    )

    assert options == [
        {"value": "text-embedding-3-large", "label": "text-embedding-3-large"},
        {"value": "text-embedding-3-small", "label": "text-embedding-3-small"},
    ]


def test_fetch_openai_compatible_models_keeps_full_chat_list(
    monkeypatch,
) -> None:
    from src.ui import provider_views

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"id": "gpt-4o"},
                    {"id": "text-embedding-3-small"},
                    {"id": "omni-moderation-latest"},
                ]
            }

    monkeypatch.setattr(
        provider_views.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(),
    )

    options = provider_views._fetch_openai_compatible_models(
        "https://api.openai.com/v1",
        "secret-key",
        model_kind="chat",
    )

    assert options == [
        {"value": "gpt-4o", "label": "gpt-4o"},
        {"value": "omni-moderation-latest", "label": "omni-moderation-latest"},
        {"value": "text-embedding-3-small", "label": "text-embedding-3-small"},
    ]


def test_fetch_azure_openai_deployments_formats_labels(monkeypatch) -> None:
    from src.ui import provider_views

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"id": "prod-gpt4o", "model": "gpt-4o"},
                    {"id": "prod-mini", "model": "gpt-4o-mini"},
                ]
            }

    def _fake_get(
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 0,
    ):
        assert url == (
            "https://example.openai.azure.com/openai/deployments"
            "?api-version=2024-10-21"
        )
        assert headers == {"api-key": "azure-secret"}
        assert timeout == 20
        return _FakeResponse()

    monkeypatch.setattr(provider_views.requests, "get", _fake_get)

    options = provider_views._fetch_azure_openai_deployments(
        "https://example.openai.azure.com/",
        "azure-secret",
        "2024-10-21",
    )

    assert options == [
        {"value": "prod-gpt4o", "label": "prod-gpt4o (gpt-4o)"},
        {"value": "prod-mini", "label": "prod-mini (gpt-4o-mini)"},
    ]


def test_on_provider_change_resets_provider_specific_defaults(
    monkeypatch,
) -> None:
    from src.ui import provider_views

    fake_st = SimpleNamespace(session_state={
        provider_views.PROVIDER_FIELD_KEYS["provider_name"]: "openai",
        provider_views.PROVIDER_FIELD_KEYS["model"]: "old-model",
        provider_views.PROVIDER_FIELD_KEYS["embedding_model"]: "old-embedding",
        provider_views.PROVIDER_FIELD_KEYS["rerank_model"]: "old-rerank",
        provider_views.PROVIDER_FIELD_KEYS["api_key"]: "pasted-key",
        provider_views.PROVIDER_FIELD_KEYS["api_key_env"]: "OLD_ENV",
        provider_views.PROVIDER_FIELD_KEYS[
            "base_url"
        ]: "https://old.example.com",
        provider_views.PROVIDER_FIELD_KEYS["api_version"]: "old-version",
        provider_views.PROVIDER_FIELD_KEYS["aws_region"]: "us-west-2",
        provider_views.PROVIDER_FIELD_KEYS["project_id"]: "old-project",
        provider_views.PROVIDER_FIELD_KEYS["location"]: "old-location",
        provider_views.PROVIDER_MODEL_SELECT_KEY: "old-model",
        provider_views.PROVIDER_MODEL_MANUAL_KEY: True,
        provider_views.PROVIDER_EMBEDDING_MODEL_SELECT_KEY: "old-embedding",
        provider_views.PROVIDER_EMBEDDING_MODEL_MANUAL_KEY: True,
        provider_views.PROVIDER_RERANK_MODEL_SELECT_KEY: "old-rerank",
        provider_views.PROVIDER_RERANK_MODEL_MANUAL_KEY: True,
    })
    monkeypatch.setattr(provider_views, "st", fake_st)

    provider_views._on_provider_change()

    assert fake_st.session_state[provider_views.PROVIDER_FIELD_KEYS["model"]] == ""
    assert fake_st.session_state[provider_views.PROVIDER_MODEL_SELECT_KEY] == ""
    assert fake_st.session_state[provider_views.PROVIDER_MODEL_MANUAL_KEY] is False
    assert (
        fake_st.session_state[provider_views.PROVIDER_FIELD_KEYS["embedding_model"]]
        == ""
    )
    assert fake_st.session_state[provider_views.PROVIDER_EMBEDDING_MODEL_SELECT_KEY] == ""
    assert (
        fake_st.session_state[provider_views.PROVIDER_EMBEDDING_MODEL_MANUAL_KEY]
        is False
    )
    assert fake_st.session_state[provider_views.PROVIDER_FIELD_KEYS["rerank_model"]] == ""
    assert fake_st.session_state[provider_views.PROVIDER_RERANK_MODEL_SELECT_KEY] == ""
    assert fake_st.session_state[provider_views.PROVIDER_RERANK_MODEL_MANUAL_KEY] is False
    assert fake_st.session_state[provider_views.PROVIDER_FIELD_KEYS["api_key"]] == ""
    assert fake_st.session_state[provider_views.PROVIDER_FIELD_KEYS["api_key_env"]] == "OPENAI_API_KEY"
    assert fake_st.session_state[provider_views.PROVIDER_FIELD_KEYS["base_url"]] == "https://api.openai.com/v1"


def test_get_provider_model_options_uses_cache(monkeypatch) -> None:
    from src.ui import provider_views

    fake_st = SimpleNamespace(session_state={provider_views.PROVIDER_MODEL_CACHE_KEY: {}})
    monkeypatch.setattr(provider_views, "st", fake_st)

    calls = {"count": 0}

    def _fake_fetch_provider_model_options(**kwargs):
        del kwargs
        calls["count"] += 1
        return [{"value": "gpt-4o", "label": "gpt-4o"}]

    monkeypatch.setattr(
        provider_views,
        "_fetch_provider_model_options",
        _fake_fetch_provider_model_options,
    )

    params = {
        "provider_name": "openai",
        "spec": provider_views.PROVIDER_SPECS["openai"],
        "model_kind": "chat",
        "auth_mode": "Paste API Key",
        "api_key": "secret",
        "api_key_env": None,
        "base_url": "https://api.openai.com/v1",
        "api_version": None,
        "aws_region": None,
        "project_id": None,
        "location": None,
    }

    first = provider_views._get_provider_model_options(**params)
    second = provider_views._get_provider_model_options(**params)

    assert first == ([{"value": "gpt-4o", "label": "gpt-4o"}], None)
    assert second == first
    assert calls["count"] == 1


def test_fetch_cohere_models_filters_embedding_and_rerank(monkeypatch) -> None:
    from src.ui import provider_views

    def _fake_request_json(url: str, *, headers: dict[str, str] | None = None):
        assert url == "https://api.cohere.com/v1/models"
        assert headers == {"Authorization": "Bearer cohere-secret"}
        return {
            "models": [
                {"name": "command-r-plus", "endpoints": ["chat"]},
                {"name": "embed-english-v3.0", "endpoints": ["embed"]},
                {"name": "rerank-v3.5", "endpoints": ["rerank"]},
            ]
        }

    monkeypatch.setattr(provider_views, "_request_json", _fake_request_json)

    embedding_options = provider_views._fetch_cohere_models(
        "cohere-secret",
        model_kind="embedding",
    )
    rerank_options = provider_views._fetch_cohere_models(
        "cohere-secret",
        model_kind="rerank",
    )

    assert embedding_options == [
        {"value": "embed-english-v3.0", "label": "embed-english-v3.0"}
    ]
    assert rerank_options == [
        {"value": "rerank-v3.5", "label": "rerank-v3.5"}
    ]


def test_fetch_provider_model_options_uses_grok_models_endpoint(monkeypatch) -> None:
    from src.ui import provider_views

    calls: list[tuple[str, str | None, str]] = []

    def _fake_fetch_openai_compatible_models(
        base_url: str,
        api_key: str | None,
        *,
        model_kind: str = "chat",
    ) -> list[dict[str, str]]:
        calls.append((base_url, api_key, model_kind))
        return [{"value": "grok-2", "label": "grok-2"}]

    monkeypatch.setattr(
        provider_views,
        "_fetch_openai_compatible_models",
        _fake_fetch_openai_compatible_models,
    )

    options = provider_views._fetch_provider_model_options(
        provider_name="grok",
        spec=provider_views.PROVIDER_SPECS["grok"],
        model_kind="chat",
        api_key="xai-secret",
        base_url="https://api.x.ai/v1",
        api_version=None,
        aws_region=None,
        project_id=None,
        location=None,
    )

    assert options == [{"value": "grok-2", "label": "grok-2"}]
    assert calls == [
        ("https://api.x.ai/v1", "xai-secret", "chat")
    ]


def test_grok_default_model_is_current_alias() -> None:
    from src.ui import provider_views

    assert provider_views.PROVIDER_SPECS["grok"]["default_model"] == "grok-4"


def test_supports_dynamic_model_options_for_builtin_providers() -> None:
    from src.ui import provider_views

    assert provider_views._supports_dynamic_model_options("openai", "chat") is True
    assert provider_views._supports_dynamic_model_options("openai", "embedding") is True
    assert provider_views._supports_dynamic_model_options("cohere", "rerank") is True
    assert provider_views._supports_dynamic_model_options("custom-http", "chat") is False


def test_get_model_discovery_placeholder_requires_api_key() -> None:
    from src.ui import provider_views

    placeholder = provider_views._get_model_discovery_placeholder(
        provider_name="openai",
        spec=provider_views.PROVIDER_SPECS["openai"],
        model_kind="chat",
        auth_mode="Paste API Key",
        api_key=None,
        api_key_env=None,
        base_url="https://api.openai.com/v1",
        api_version=None,
        aws_region=None,
        project_id=None,
        location=None,
        fetch_error=None,
    )

    assert placeholder == "Add API key to load models"


def test_sync_provider_test_result_clears_stale_provider_result(monkeypatch) -> None:
    from src.config import ProviderConfig
    from src.ui import provider_views

    fake_st = SimpleNamespace(session_state={
        provider_views.PROVIDER_FIELD_KEYS["auth_mode"]: "Paste API Key",
        provider_views.PROVIDER_TEST_SESSION_KEY: {
            "status": "ok",
            "auth_mode": "Paste API Key",
            "provider_config": {
                "name": "openai",
                "model": "gpt-4o",
                "embedding_model": None,
                "rerank_model": None,
                "api_key": None,
                "api_key_env": None,
                "base_url": "https://api.openai.com/v1",
                "api_version": None,
                "aws_region": None,
                "project_id": None,
                "location": None,
                "headers": {},
                "request_template": {},
                "response_text_path": None,
                "response_tokens_path": None,
                "timeout_seconds": 60.0,
                "max_retries": 2,
                "retry_backoff_seconds": 1.5,
                "temperature": 0.2,
                "max_tokens": 2048,
            },
        },
    })
    monkeypatch.setattr(provider_views, "st", fake_st)

    provider_views._sync_provider_test_result(
        ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key=None,
            api_key_env="ANTHROPIC_API_KEY",
        )
    )

    assert provider_views.PROVIDER_TEST_SESSION_KEY not in fake_st.session_state


def test_sync_provider_test_result_keeps_matching_provider_result(monkeypatch) -> None:
    from src.config import ProviderConfig
    from src.ui import provider_views

    provider_config = ProviderConfig(
        name="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
    )
    serialized = provider_views._serialize_provider_config(provider_config)
    fake_st = SimpleNamespace(session_state={
        provider_views.PROVIDER_FIELD_KEYS["auth_mode"]: "Paste API Key",
        provider_views.PROVIDER_TEST_SESSION_KEY: {
            "status": "ok",
            "auth_mode": "Paste API Key",
            "provider_config": serialized,
        },
    })
    monkeypatch.setattr(provider_views, "st", fake_st)

    provider_views._sync_provider_test_result(provider_config)

    assert fake_st.session_state[provider_views.PROVIDER_TEST_SESSION_KEY]["provider_config"] == serialized