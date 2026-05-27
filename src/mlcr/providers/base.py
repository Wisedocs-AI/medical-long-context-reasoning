from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlcr.config import ModelConfig


class ProviderError(Exception):
    pass


class TransientProviderError(ProviderError):
    """Retryable error (rate limit, 5xx, network blip)."""


class PermanentProviderError(ProviderError):
    """Non-retryable error (bad request, auth, model doesn't support modality)."""


@dataclass
class ChatRequest:
    system: str | None
    user_text: str | None
    model_cfg: ModelConfig
    images: list[Path] = field(default_factory=list)


@dataclass
class ChatResponse:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    thinking: str | None = None


class Provider(ABC):
    name: str = ""

    @abstractmethod
    def call(self, req: ChatRequest) -> ChatResponse: ...


# Shared transient status codes recognised across all providers.
_TRANSIENT_STATUS_CODES: set[int] = {408, 409, 425, 429, 499, 500, 502, 503, 504, 529}

# Substrings matched against lowercased exception class names.
_TRANSIENT_CLASS_SUBSTRINGS: tuple[str, ...] = (
    "ratelimit", "resourceexhausted", "deadlineexceeded",
    "overloaded", "throttl", "timeout", "cancelled",
    "unavailable", "serviceunavailable",
    "apiconnection", "internalserver",
    "remoteprotocol", "connect",
)

# Substrings matched against lowercased exception messages.
_TRANSIENT_MSG_SUBSTRINGS: tuple[str, ...] = (
    "timed out", "timeout", "deadline exceeded",
    "operation was cancelled", "cancelled",
    "server disconnected", "disconnected",
    "connection reset", "connection aborted",
    "connection error", "peer closed",
    "remotedisconnected", "incompleteread",
)


def map_provider_error(e: Exception) -> ProviderError:
    """Classify an SDK/network exception as transient or permanent."""
    msg = str(e)
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if code in _TRANSIENT_STATUS_CODES:
        return TransientProviderError(msg)
    cls = type(e).__name__.lower()
    if any(s in cls for s in _TRANSIENT_CLASS_SUBSTRINGS):
        return TransientProviderError(msg)
    if any(s in msg.lower() for s in _TRANSIENT_MSG_SUBSTRINGS):
        return TransientProviderError(msg)
    return PermanentProviderError(msg)
