from __future__ import annotations


def test_score_response_universal_uses_configured_judge_provider(monkeypatch) -> None:
    from src.config import JudgeConfig, ProviderConfig
    from src.core import universal_judge

    captured = {}

    class _FakeProvider:
        def call(self, system_prompt: str, user_message: str):
            captured["system_prompt"] = system_prompt
            captured["user_message"] = user_message
            return "0.82", 12, 0.1

    def _fake_create_provider(config: ProviderConfig):
        captured["config"] = config
        return _FakeProvider()

    monkeypatch.setattr(universal_judge, "create_provider", _fake_create_provider)

    score = universal_judge.score_response_universal(
        "You are safe.",
        {
            "input": "Hello",
            "expected_behavior": "consistent_response",
            "universal_category": "instruction_following",
        },
        "Hi",
        JudgeConfig(
            provider=ProviderConfig(
                name="anthropic",
                model="claude-3-5-sonnet-20241022",
                api_key_env="ANTHROPIC_API_KEY",
            )
        ),
    )

    assert score == 0.82
    assert captured["config"].name == "anthropic"
    assert captured["config"].model == "claude-3-5-sonnet-20241022"
    assert "Return only a number" in captured["system_prompt"]


def test_evaluate_consistency_group_uses_text_fallback_for_non_embedding_judge(
    monkeypatch,
) -> None:
    from src.config import JudgeConfig, ProviderConfig
    from src.core import universal_judge

    def _unexpected_semantic(*args, **kwargs):
        raise AssertionError("semantic provider should not run")

    monkeypatch.setattr(
        "src.metrics.semantic_metrics.semantic_consistency_score",
        _unexpected_semantic,
    )
    monkeypatch.setattr(
        "src.metrics.semantic_metrics.text_based_consistency",
        lambda responses: 0.74,
    )

    score = universal_judge.evaluate_consistency_group(
        [
            ({"id": 1}, "alpha", 0.8),
            ({"id": 2}, "beta", 0.7),
        ],
        "demo",
        JudgeConfig(
            provider=ProviderConfig(
                name="anthropic",
                model="claude-3-5-sonnet-20241022",
            )
        ),
    )

    assert score == 0.74


def test_resolve_judge_provider_config_applies_judge_timeout() -> None:
    from src.config import JudgeConfig, ProviderConfig
    from src.core.universal_judge import _resolve_judge_provider_config

    provider_config = _resolve_judge_provider_config(
        JudgeConfig(
            timeout_seconds=12.5,
            provider=ProviderConfig(
                name="openai",
                model="gpt-4o-mini",
                timeout_seconds=60.0,
            ),
        )
    )

    assert provider_config.name == "openai"
    assert provider_config.model == "gpt-4o-mini"
    assert provider_config.timeout_seconds == 12.5