from __future__ import annotations


def test_merge_run_args_supports_independent_judge_provider() -> None:
    from src.cli import _merge_run_args, build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--judge-provider",
            "anthropic",
            "--judge-model",
            "claude-3-5-sonnet-20241022",
            "--judge-api-key-env",
            "ANTHROPIC_API_KEY",
            "--judge-base-url",
            "https://api.anthropic.com",
        ]
    )

    config = _merge_run_args(args)

    assert config.judge.provider.name == "anthropic"
    assert config.judge.provider.model == "claude-3-5-sonnet-20241022"
    assert config.judge.provider.api_key_env == "ANTHROPIC_API_KEY"
    assert config.judge.provider.base_url == "https://api.anthropic.com"
    assert config.judge.ollama_model == "claude-3-5-sonnet-20241022"


def test_benchmark_config_from_dict_loads_judge_provider() -> None:
    from src.config import benchmark_config_from_dict

    config = benchmark_config_from_dict(
        {
            "judge": {
                "strategy": "llm",
                "provider": {
                    "name": "grok",
                    "model": "grok-4",
                    "api_key_env": "XAI_API_KEY",
                    "base_url": "https://api.x.ai/v1",
                },
            }
        }
    )

    assert config.judge.provider.name == "grok"
    assert config.judge.provider.model == "grok-4"
    assert config.judge.provider.api_key_env == "XAI_API_KEY"
    assert config.judge.provider.base_url == "https://api.x.ai/v1"