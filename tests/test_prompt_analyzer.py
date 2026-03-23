from __future__ import annotations

from src.config import ProviderConfig
from src.utils import prompt_analyzer


def test_analyze_system_prompt_uses_selected_provider(monkeypatch) -> None:
    class _FakeProvider:
        def call(self, system_prompt: str, user_message: str):
            assert "strict prompt analysis engine" in system_prompt.lower()
            assert "SYSTEM PROMPT:" in user_message
            assert "customer support" in user_message.lower()
            return (
                """
                {
                  "role": "customer support agent",
                  "domain": "customer_support",
                  "capabilities": ["answer support questions"],
                  "boundaries": ["share secrets"],
                  "constraints": {
                    "format": "Markdown",
                    "length": "concise",
                    "tone": "professional",
                    "language": "English"
                  },
                  "core_topics": ["orders", "refunds"]
                }
                """,
                123,
                0.2,
            )

        def get_model_name(self) -> str:
            return "Anthropic/claude-3-5-sonnet-20241022"

    monkeypatch.setattr(
        prompt_analyzer,
        "create_provider",
        lambda config: _FakeProvider(),
    )

    analysis = prompt_analyzer.analyze_system_prompt(
        "You are a customer support assistant helping with refunds.",
        provider_config=ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
        ),
        use_llm=True,
    )

    assert analysis["analysis_method"] == "llm"
    assert analysis["analysis_provider"] == "Anthropic/claude-3-5-sonnet-20241022"
    assert analysis["role"] == "customer support agent"
    assert analysis["constraints"]["tone"] == "professional"


def test_analyze_system_prompt_falls_back_when_provider_fails(monkeypatch) -> None:
    def _raise_provider_error(config):
        del config
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        prompt_analyzer,
        "create_provider",
        _raise_provider_error,
    )

    analysis = prompt_analyzer.analyze_system_prompt(
        "You are a finance assistant. Never provide investment advice.",
        provider_config=ProviderConfig(name="openai", model="gpt-4o"),
        use_llm=True,
    )

    assert analysis["analysis_method"] == "heuristic"
    assert analysis["analysis_fallback_reason"] == "provider unavailable"
    assert analysis["role"]
    assert isinstance(analysis["boundaries"], list)