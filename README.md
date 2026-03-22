<p align="center">
  <img src="logo.png" width="320" alt="System Prompt Benchmark">
</p>


Automated red-team evaluation of LLM system prompts across 12 security and behavioral categories.

<p align="center">
  <img src="https://img.shields.io/badge/Attack_Vectors-287-1565C0?style=flat" alt="Vectors">
  <img src="https://img.shields.io/badge/Prompt_Injection-Tested-1976D2?style=flat" alt="Injection">
  <img src="https://img.shields.io/badge/Jailbreak-Detection-2196F3?style=flat" alt="Jailbreak">
  <img src="https://img.shields.io/badge/Data_Leaks-Prevention-42A5F5?style=flat" alt="Leaks">
</p>

## Highlights

- 300-test benchmark covering jailbreaks, prompt leakage, authority bypass, and role drift
- Multi-turn adaptive attacks driven by a separate red-team LLM
- Weighted scoring across 12 universal categories with per-category breakdowns
- Three interfaces: Streamlit UI, CLI, and REST API
- Plugin SDK for custom providers, transforms, judges, and exporters

## Demo

![Demo](docs/demo.gif)

> No GIF recorded yet. Run `bash start.sh` to see the UI at `http://localhost:8501`.

## Overview

System Prompt Benchmark evaluates how well a system prompt resists adversarial inputs. Paste or upload any system prompt, pick a model, and run up to 300 attack tests. Each response is scored by a configurable judge stack and mapped to 12 universal categories — from jailbreak resistance and prompt leakage to multi-turn consistency. Results are exportable as JSON or PDF.

Built for AI engineers and product teams who ship LLM-powered features and need to verify that system prompts hold up under pressure before deployment.

## Motivation

Existing tools either test generic safety (not prompt-specific security) or require a fixed dataset format tied to one vendor. Prompts that pass informal testing in the playground routinely fail under systematic adversarial pressure — authority escalation, encoding tricks, multi-turn erosion. This tool treats every system prompt as the unit under test, produces reproducible scores, and compares results across prompt versions.

## Features

- Quick / Standard / Full benchmark modes (10 / 100 / 300 tests)
- 12 weighted evaluation categories: role adherence, instruction following, security, jailbreak resistance, consistency, scope boundaries, graceful degradation, ethics & compliance, constraint following, robustness, multi-turn behavior, edge cases
- 5 adaptive multi-turn attack strategies: prompt leak escalation, authority escalation, tool hijack escalation, jailbreak escalation, social engineering escalation
- LLM-adaptive attacker mode using a separate red-team model
- 15+ supported providers: OpenAI, Anthropic, Gemini, Vertex AI, Azure OpenAI, Bedrock, Cohere, Together, Mistral, OpenRouter, Fireworks, Groq, Grok, Ollama, custom HTTP
- Pluggable detector stack: pattern detectors, OpenAI Moderation, Perspective API, HarmJudge, external webhook
- Dataset management: JSON / JSONL / CSV formats, remote catalog sync, preset profiles
- Version comparison: `spb compare baseline.json candidate.json`
- Plugin SDK for custom providers, transforms, judges, and exporters
- REST API with SQLite job store, Redis queue backend, Prometheus metrics
- PDF report export

## Architecture

Components:

- `app.py` — Streamlit UI; sidebar collects system prompt, provider, test mode, dataset, judge config
- `src/core/run_universal_benchmark.py` — benchmark runner with parallel execution, rate limiting, consistency group scoring
- `src/core/detectors.py` — pattern detector families (leakage, jailbreak, refusal, toxicity)
- `src/core/universal_judge.py` — multi-strategy judge: heuristic, LLM, ensemble, consistency
- `src/providers/run_benchmark.py` — unified LLM provider abstraction (15+ backends)
- `src/datasets.py` — dataset loading, validation, transforms, remote catalog sync
- `src/api.py` — FastAPI REST API with job queue, worker backend, webhook delivery
- `src/cli.py` — full-featured CLI (`run`, `compare`, `summarize`, `validate-dataset`, `convert-dataset`, `sync-packs`, `plugins`, …)
- `src/metrics/` — formal metrics, semantic similarity, degradation scoring
- `src/plugins/` — plugin manager and SDK (providers, transforms, judges, exporters)

Flow:

```
System Prompt + Attack Dataset
        ↓
  Provider (LLM call)
        ↓
  Detector Stack (patterns + optional external)
        ↓
  Judge (heuristic / LLM / ensemble)
        ↓
  Category Scoring (12 weighted categories)
        ↓
  Report (JSON / PDF / UI)
```

## Tech Stack

- Python 3.9+
- Streamlit — web UI
- FastAPI + Uvicorn — REST API
- SQLite — job store (API mode)
- Redis — optional queue backend
- Plotly / ReportLab — charts and PDF export
- Ollama — default local judge and attacker model

## Quick Start

```bash
git clone https://github.com/KazKozDev/system-prompt-benchmark
cd system-prompt-benchmark
pip install -r requirements.txt

# Streamlit UI
bash start.sh

# Or directly
streamlit run app.py
```

The UI opens at `http://localhost:8501`. Select a system prompt, configure a provider, and click **Start Benchmark**.

More details → [docs/](docs/)

## Usage

**CLI — run a benchmark:**

```bash
python -m spb run \
  --prompt prompts/customer-support-bot.txt \
  --provider openai \
  --model gpt-4o \
  --api-key-env OPENAI_API_KEY \
  --max-tests 100 \
  --output results/run.json
```

**CLI — compare two runs:**

```bash
python -m spb compare results/baseline.json results/candidate.json
```

**CLI — local Ollama (no API key needed):**

```bash
python -m spb run \
  --prompt prompts/customer-support-bot.txt \
  --provider ollama \
  --model qwen3:14b \
  --max-tests 25
```

**Config file:**

```bash
python -m spb run -c benchmark.example.yaml
```

**REST API:**

```bash
uvicorn src.api:app --port 8080

curl -X POST http://localhost:8080/runs \
  -H "Authorization: Bearer $SPB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"config": {"prompt_file": "prompts/customer-support-bot.txt", "provider": {"name": "ollama", "model": "qwen3:14b"}}}'
```

## Project Structure

```
src/
  core/           benchmark runner, judge, detectors, categories
  providers/      unified LLM provider layer (15+ backends)
  datasets.py     dataset loading, validation, remote sync
  api.py          FastAPI REST API
  cli.py          command-line interface
  metrics/        formal, semantic, and degradation metrics
  plugins/        plugin manager and SDK
  ui/             Streamlit view components
  utils/          prompt analyzer, PDF report
prompts/          9 example system prompts
tests/            benchmark datasets
datasets/         custom dataset examples
config/           YAML configs
results/          benchmark outputs (gitignored)
docs/             plugin SDK reference, roadmap
```

## Status

- Stage: Beta
- Planned:
  - HuggingFace dataset registry integration
  - Automated prompt improvement suggestions
  - CI/CD benchmark gate (`spb run` returns non-zero on score below threshold)
  - Extended multimodal attack coverage

## Testing

```bash
pytest tests/ -v
```

## Contributing

- Fork the repo
- Create a feature branch
- Open a pull request

Plugin development: see [docs/plugin-sdk.md](docs/plugin-sdk.md)

## License

MIT

## Contact

- GitHub: [@KazKozDev](https://github.com/KazKozDev)
