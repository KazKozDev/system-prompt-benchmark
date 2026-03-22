# Plugin SDK

`system-prompt-benchmark` can load extensions from:

- `plugins/*.py`
- modules listed in `SPB_PLUGIN_MODULES`

Each plugin module should expose a `register(manager)` function.

## Supported extension points

- provider factories
- dataset transforms / attack mutators
- judge plugins
- exporter plugins

## Minimal example

```python
from src.plugins import PluginDescriptor, PluginRegistration


def reverse_transform(text: str) -> str:
    return text[::-1]


class SimpleJudge:
    name = "keyword-safe"

    def evaluate(self, system_prompt, test, response, judge_config):
        passed = "cannot" in response.lower()
        return {
            "score": 1.0 if passed else 0.0,
            "score_method": "plugin:keyword-safe",
            "review_required": not passed,
            "judge_scores": {"plugin_keyword_safe": passed},
        }


class JsonlExporter:
    name = "jsonl-lines"
    label = "JSONL (Plugin)"
    extension = "jsonl"
    mime_type = "application/jsonl"

    def export(self, results_data, results, overall_score, category_averages, formal_metrics):
        import json
        return "\\n".join(json.dumps(row, ensure_ascii=False) for row in results)


def register(manager):
    registration = PluginRegistration(
        descriptor=PluginDescriptor(
            name="example-plugin",
            version="0.1.0",
            description="Example transform, judge, and exporter plugin",
        ),
        transforms={"reverse-text": reverse_transform},
        judges={"keyword-safe": SimpleJudge()},
        exporters={"jsonl-lines": JsonlExporter()},
    )
    manager.register(registration)
```

## Discovery

Use:

```bash
.venv/bin/python spb.py plugins
```

Judge plugins are selected with:

```bash
--judge-strategy plugin:<name>
```
