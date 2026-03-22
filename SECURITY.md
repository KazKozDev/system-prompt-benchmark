# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in system-prompt-benchmark, **do not open a public GitHub issue**.

Report it privately to: [kazkozdev@gmail.com](mailto:kazkozdev@gmail.com)

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within 72 hours. Once confirmed, we will work on a fix and coordinate disclosure.

## Scope

This policy covers:
- The benchmark engine and evaluation logic (`src/core/`)
- The REST API and authentication (`src/api.py`, `src/platform/`)
- Dataset loading and remote sync (`src/datasets.py`)

## Note on Intended Behavior

This tool is designed to test LLM system prompts against adversarial attacks. The attack datasets in `tests/` contain prompt injection and jailbreak examples — this is expected and intentional. Vulnerabilities in the benchmark tool itself (not the LLMs being tested) are in scope for this policy.
