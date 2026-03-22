"""Stable plugin SDK for system-prompt-benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderFactory(Protocol):
    name: str

    def create(self, config: Any) -> Any: ...


class TransformFunc(Protocol):
    def __call__(self, text: str) -> str: ...


class JudgePlugin(Protocol):
    name: str

    def evaluate(self, system_prompt: str, test: dict, response: str, judge_config: Any) -> Any: ...


class ExporterPlugin(Protocol):
    name: str
    label: str
    extension: str
    mime_type: str

    def export(self, results_data: dict, results: list[dict], overall_score: float, category_averages: dict, formal_metrics: dict) -> str | bytes: ...


@dataclass
class PluginDescriptor:
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class PluginRegistration:
    descriptor: PluginDescriptor
    providers: dict[str, ProviderFactory] = field(default_factory=dict)
    transforms: dict[str, TransformFunc] = field(default_factory=dict)
    judges: dict[str, JudgePlugin] = field(default_factory=dict)
    exporters: dict[str, ExporterPlugin] = field(default_factory=dict)
