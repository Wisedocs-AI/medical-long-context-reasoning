from __future__ import annotations

import base64
import os
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


class AzureOpenAIProvider(Provider):
    """GPT models via Azure OpenAI Service.

    Auth: reads ``AZURE_OPENAI_API_KEY`` and ``AZURE_OPENAI_ENDPOINT`` from the
    environment.  Optionally set ``AZURE_OPENAI_API_VERSION`` (default
    ``2025-04-01-preview``).

    When thinking is enabled, uses the Responses API (``client.responses.create``)
    to get reasoning summaries. Otherwise falls back to Chat Completions.
    """
    name = "azure_openai"

    def __init__(self) -> None:
        try:
            from openai import AzureOpenAI  # type: ignore
        except ImportError as e:
            raise PermanentProviderError(
                "openai SDK not installed. `pip install 'mlcr[azure]'`"
            ) from e

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            raise PermanentProviderError("AZURE_OPENAI_API_KEY is not set")
        if not endpoint:
            raise PermanentProviderError("AZURE_OPENAI_ENDPOINT is not set")

        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    def call(self, req: ChatRequest) -> ChatResponse:
        if req.model_cfg.thinking:
            return self._call_responses(req)
        return self._call_chat(req)

    def _call_responses(self, req: ChatRequest) -> ChatResponse:
        """Use the Responses API to get reasoning summaries."""
        input_items: list[dict] = []
        if req.system:
            input_items.append({"role": "developer", "content": req.system})

        user_content: list[dict] = []
        for p in req.images:
            media_type = _MEDIA.get(p.suffix.lower(), "image/png")
            data = base64.b64encode(p.read_bytes()).decode()
            user_content.append({
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{data}",
            })
        if req.user_text:
            user_content.append({"type": "input_text", "text": req.user_text})
        input_items.append({"role": "user", "content": user_content})

        effort = req.model_cfg.thinking.get("effort", "medium")
        kwargs: dict = {
            "model": req.model_cfg.model,
            "max_output_tokens": req.model_cfg.max_output_tokens,
            "input": input_items,
            "reasoning": {"effort": effort, "summary": "detailed"},
        }
        kwargs.update(req.model_cfg.extra or {})

        t0 = time.time()
        try:
            resp = self._client.responses.create(**kwargs)
        except Exception as e:
            raise map_provider_error(e)
        latency_ms = int((time.time() - t0) * 1000)

        text = resp.output_text or ""
        reasoning_summaries: list[str] = []
        for item in resp.output:
            if getattr(item, "type", None) == "reasoning":
                for part in (item.summary or []):
                    if getattr(part, "type", None) == "summary_text":
                        reasoning_summaries.append(part.text)

        usage: dict = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            }
            otd = getattr(resp.usage, "output_tokens_details", None)
            reasoning_tokens = getattr(otd, "reasoning_tokens", None) if otd else None
            if reasoning_tokens is not None:
                usage["thinking_tokens"] = reasoning_tokens

        return ChatResponse(
            text=text,
            usage=usage,
            raw={"id": resp.id, "model": resp.model},
            latency_ms=latency_ms,
            thinking="\n\n".join(reasoning_summaries) if reasoning_summaries else None,
        )

    def _call_chat(self, req: ChatRequest) -> ChatResponse:
        """Use Chat Completions API (no thinking)."""
        user_content: list[dict] = []
        for p in req.images:
            media_type = _MEDIA.get(p.suffix.lower(), "image/png")
            data = base64.b64encode(p.read_bytes()).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{data}",
                },
            })
        if req.user_text:
            user_content.append({"type": "text", "text": req.user_text})

        messages: list[dict] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": user_content})

        kwargs: dict = {
            "model": req.model_cfg.model,
            "max_completion_tokens": req.model_cfg.max_output_tokens,
            "temperature": req.model_cfg.temperature,
            "messages": messages,
        }
        kwargs.update(req.model_cfg.extra or {})

        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise map_provider_error(e)
        latency_ms = int((time.time() - t0) * 1000)

        text = resp.choices[0].message.content or "" if resp.choices else ""

        usage: dict = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }

        return ChatResponse(
            text=text,
            usage=usage,
            raw={"id": resp.id, "model": resp.model, "finish_reason": resp.choices[0].finish_reason if resp.choices else None},
            latency_ms=latency_ms,
            thinking=None,
        )
