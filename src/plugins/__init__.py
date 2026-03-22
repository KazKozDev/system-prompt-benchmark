"""Plugin SDK and loader."""

from src.plugins.manager import get_plugin_manager
from src.plugins.sdk import PluginDescriptor, PluginRegistration

__all__ = ["PluginDescriptor", "PluginRegistration", "get_plugin_manager"]
