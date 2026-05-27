from __future__ import annotations

import base64
import json
import os
import tempfile
import time

from mlcr.providers.base import (
    ChatRequest,
    ChatResponse,
    PermanentProviderError,
    Provider,
    map_provider_error,
)

_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".webp": "image/webp", ".gif": "image/gif"}


def _ensure_credentials_file() -> None:
    """AnthropicVertex uses google.auth.default() which requires
    GOOGLE_APPLICATION_CREDENTIALS to be a file path, not raw JSON.
    When the env var contains inline JSON (as the Google provider supports),
    write it to a NamedTemporaryFile and update the env var to point at it.
    The file is intentionally not cleaned up — it lives for the process lifetime.
    """
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if creds.startswith("{"):
        try:
            json.loads(creds)  # validate before writing
        except json.JSONDecodeError as e:
            raise PermanentProviderError(
                f"GOOGLE_APPLICATION_CREDENTIALS looks like JSON but is invalid: {e}"
            )
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="mlcr_sa_"
        )
        tmp.write(creds)
        tmp.flush()
        tmp.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name


class AnthropicGCPProvider(Provider):
    """Claude via Vertex AI (GCP).

    Auth: reads ``GOOGLE_CLOUD_PROJECT`` and ``GOOGLE_CLOUD_LOCATION`` (default
    ``us-east5``) from the environment. Application Default Credentials (ADC) are
    used automatically — run ``gcloud auth application-default login`` locally or
    set ``GOOGLE_APPLICATION_CREDENTIALS`` to a service-account key file or inline
    JSON (the same inline-JSON format accepted by the Google provider).

    The ``model:`` field in a model config should be the Vertex AI model name, e.g.
    ``claude-sonnet-4-6@20250514``.
    """
    name = "anthropic_gcp"

    def __init__(self) -> None:
        try:
            from anthropic import AnthropicVertex  # type: ignore
        except ImportError as e:
            raise PermanentProviderError(
                "anthropic SDK with vertex extras not installed. "
                "`pip install 'mlcr[anthropic]'`"
            ) from e
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise PermanentProviderError("GOOGLE_CLOUD_PROJECT is not set")
        region = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east5")
        _ensure_credentials_file()
        self._client = AnthropicVertex(project_id=project, region=region)

    def call(self, req: ChatRequest) -> ChatResponse:
        content: list[dict] = []
        for p in req.images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _MEDIA.get(p.suffix.lower(), "image/png"),
                    "data": base64.b64encode(p.read_bytes()).decode(),
                },
            })
        if req.user_text:
            content.append({"type": "text", "text": req.user_text})

        # Mark the last content block as the prompt-cache breakpoint. The two
        # thinking variants in each cache group fire back-to-back, so the
        # default 5-minute TTL keeps the cache warm for both hits at minimal
        # write-premium cost (1.25x vs 2x for the 1-hour TTL).
        if content:
            content[-1]["cache_control"] = {"type": "ephemeral"}

        kwargs: dict = {
            "model": req.model_cfg.model,
            "max_tokens": req.model_cfg.max_output_tokens,
            "temperature": req.model_cfg.temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if req.system:
            kwargs["system"] = req.system
        if req.model_cfg.thinking:
            kwargs["thinking"] = req.model_cfg.thinking
        kwargs.update(req.model_cfg.extra or {})

        t0 = time.time()
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as e:
            raise map_provider_error(e)
        latency_ms = int((time.time() - t0) * 1000)

        text_parts: list[str] = []
        thinking_text: list[str] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "thinking":
                thinking_text.append(getattr(block, "thinking", ""))
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            }
            if thinking_text:
                otd = getattr(resp.usage, "output_tokens_details", None)
                native_tt = getattr(otd, "thinking_tokens", None) if otd else None
                if native_tt is not None:
                    usage["thinking_tokens"] = native_tt
            cache_write = getattr(resp.usage, "cache_creation_input_tokens", None)
            cache_read = getattr(resp.usage, "cache_read_input_tokens", None)
            if cache_write is not None:
                usage["cache_creation_input_tokens"] = cache_write
            if cache_read is not None:
                usage["cache_read_input_tokens"] = cache_read
        return ChatResponse(
            text="".join(text_parts),
            usage=usage,
            raw={"id": resp.id, "stop_reason": resp.stop_reason},
            latency_ms=latency_ms,
            thinking="\n\n".join(thinking_text) if thinking_text else None,
        )
