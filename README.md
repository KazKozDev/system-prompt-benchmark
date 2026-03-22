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

- Automated 300-test benchmark for any system prompt
- Multi-turn adaptive attacks via dedicated red-team LLM
- 15+ supported providers and customizable detector stack
- Developer-friendly interfaces: Streamlit UI, CLI, and REST API

## Demo

![Demo](docs/demo.gif)

## Overview

System Prompt Benchmark evaluates your system prompt's resistance against adversarial inputs.
It automatically runs up to 300 attack vectors across 12 security categories and scores the responses.
Provides engineering teams with reproducible robustness metrics before shipping LLM features to production.

## Motivation

- Playgrounds and ad-hoc testing miss complex, multi-turn jailbreaks.
- Most tools test generic model safety, not prompt-specific security.
- This project provides a systematic, automated red-teaming pipeline for your exact system instructions.

## Features

- 3 scaling modes: Quick, Standard, and Full benchmark
- 12 evaluation categories including jailbreak resistance and instruction following
- 5 adaptive attack strategies like authority escalation and tool hijacking
- Pluggable judge and detector stack
- Dataset management with remote catalog sync

## Architecture

Components:

- Runner — handles parallel test execution and rate limiting
- Providers — unifies 15+ LLM APIs behind a single interface
- Judge — scores responses using heuristic, LLM, and consistency strategies

Flow:

System Prompt + Tests → Provider Call → Detector Stack → Universal Judge → Final Report

## Tech Stack

- Python
- Streamlit
- FastAPI
- SQlite

## Quick Start

1. `git clone https://github.com/KazKozDev/system-prompt-benchmark.git`
2. `pip install -r requirements.txt`
3. `bash start.sh`

More details → docs/setup.md

## Usage

Example:

```bash
python -m spb run --prompt custom.txt --provider ollama --model qwen3:14b
```

## Project Structure

```
src/
  core/
  providers/
  metrics/
  plugins/
```

## Status

- Stage: Beta
- Planned:
  - HuggingFace dataset registry integration
  - CI/CD benchmark gate

## Testing

```bash
pytest tests/ -v
```

## Contributing

- Fork
- Branch
- PR

## License

MIT

## Contact

- GitHub
- LinkedIn
- Email
