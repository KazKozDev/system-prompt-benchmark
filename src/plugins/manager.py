"""Plugin registry and loader."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any

from src.plugins.sdk import ExporterPlugin, JudgePlugin, PluginDescriptor, PluginRegistration, ProviderFactory, TransformFunc


PLUGIN_DIR = Path("plugins")


class PluginManager:
    def __init__(self) -> None:
        self._loaded = False
        self._descriptors: dict[str, PluginDescriptor] = {}
        self._providers: dict[str, ProviderFactory] = {}
        self._transforms: dict[str, TransformFunc] = {}
        self._judges: dict[str, JudgePlugin] = {}
        self._exporters: dict[str, ExporterPlugin] = {}

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load_from_plugin_dir()
        self._load_from_env()

    def register(self, registration: PluginRegistration) -> None:
        self._descriptors[registration.descriptor.name] = registration.descriptor
        self._providers.update(registration.providers)
        self._transforms.update(registration.transforms)
        self._judges.update(registration.judges)
        self._exporters.update(registration.exporters)

    def register_provider(self, name: str, factory: ProviderFactory) -> None:
        self._providers[name] = factory

    def register_transform(self, name: str, transform: TransformFunc) -> None:
        self._transforms[name] = transform

    def register_judge(self, name: str, judge: JudgePlugin) -> None:
        self._judges[name] = judge

    def register_exporter(self, name: str, exporter: ExporterPlugin) -> None:
        self._exporters[name] = exporter

    def provider_factory(self, name: str) -> ProviderFactory | None:
        self.ensure_loaded()
        return self._providers.get(name)

    def transform(self, name: str) -> TransformFunc | None:
        self.ensure_loaded()
        return self._transforms.get(name)

    def judge(self, name: str) -> JudgePlugin | None:
        self.ensure_loaded()
        return self._judges.get(name)

    def exporter(self, name: str) -> ExporterPlugin | None:
        self.ensure_loaded()
        return self._exporters.get(name)

    def provider_names(self) -> list[str]:
        self.ensure_loaded()
        return sorted(self._providers)

    def transform_names(self) -> list[str]:
        self.ensure_loaded()
        return sorted(self._transforms)

    def judge_names(self) -> list[str]:
        self.ensure_loaded()
        return sorted(self._judges)

    def exporters(self) -> list[ExporterPlugin]:
        self.ensure_loaded()
        return [self._exporters[name] for name in sorted(self._exporters)]

    def descriptors(self) -> list[PluginDescriptor]:
        self.ensure_loaded()
        return [self._descriptors[name] for name in sorted(self._descriptors)]

    def _load_from_plugin_dir(self) -> None:
        if not PLUGIN_DIR.exists():
            return
        for path in sorted(PLUGIN_DIR.glob("*.py")):
            if path.name.startswith("_"):
                continue
            module_name = f"spb_plugin_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self._register_module(module)

    def _load_from_env(self) -> None:
        raw = os.getenv("SPB_PLUGIN_MODULES", "")
        for module_name in [item.strip() for item in raw.split(",") if item.strip()]:
            module = importlib.import_module(module_name)
            self._register_module(module)

    def _register_module(self, module: Any) -> None:
        register = getattr(module, "register", None)
        if callable(register):
            register(self)


_PLUGIN_MANAGER = PluginManager()


def get_plugin_manager() -> PluginManager:
    _PLUGIN_MANAGER.ensure_loaded()
    return _PLUGIN_MANAGER
