# Contributing

Thank you for your interest in contributing to system-prompt-benchmark.

## How to contribute

1. **Fork** the repository
2. **Create a branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** — keep them focused on a single concern
4. **Test** your changes
   ```bash
   pytest tests/
   ```
5. **Submit a Pull Request** against `main`

## What we welcome

- New benchmark attack packs (add to `tests/`)
- New LLM provider integrations (add to `src/providers/`)
- New judge strategies or detectors (add to `src/core/`)
- Bug fixes and performance improvements
- Documentation improvements

## Guidelines

- Follow the existing code style
- Keep PRs small and focused
- Write tests for new functionality
- Update `benchmark.example.yaml` if you add new config options
- Add example system prompts to `prompts/` if relevant

## Reporting issues

Use the GitHub issue templates for bug reports and feature requests.

## Questions

[LinkedIn](https://www.linkedin.com/in/kazkozdev/) · [Email](mailto:kazkozdev@gmail.com)
