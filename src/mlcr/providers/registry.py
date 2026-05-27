from __future__ import annotations

from typing import Callable

from mlcr.providers.base import Provider

_REGISTRY: dict[str, Callable[[], Provider]] = {}
_INSTANCES: dict[str, Provider] = {}


def register(name: str, factory: Callable[[], Provider]) -> None:
    _REGISTRY[name] = factory
    _INSTANCES.pop(name, None)


def get(name: str) -> Provider:
    if name in _INSTANCES:
        return _INSTANCES[name]
    if name not in _REGISTRY:
        _load_builtins()
    if name not in _REGISTRY:
        raise KeyError(f"unknown provider: {name}")
    _INSTANCES[name] = _REGISTRY[name]()
    return _INSTANCES[name]


def _load_builtins() -> None:
    # Lazy registration so optional SDK imports don't break basic usage.
    def _google():
        from mlcr.providers.google_provider import GoogleProvider
        return GoogleProvider()
    _REGISTRY.setdefault("google", _google)

    def _anthropic_gcp():
        from mlcr.providers.anthropic_gcp_provider import AnthropicGCPProvider
        return AnthropicGCPProvider()
    _REGISTRY.setdefault("anthropic_gcp", _anthropic_gcp)

    def _azure_openai():
        from mlcr.providers.azure_openai_provider import AzureOpenAIProvider
        return AzureOpenAIProvider()
    _REGISTRY.setdefault("azure_openai", _azure_openai)
