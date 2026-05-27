from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

from mlcr.providers.base import (
    ChatRequest,
    ChatResponse,
    PermanentProviderError,
    Provider,
    map_provider_error,
)

_log = logging.getLogger("mlcr")

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp", ".gif": "image/gif"}

# Vertex AI rejects a generateContent request whose total (base64-encoded) body
# exceeds ~20 MB with a 400 INVALID_ARGUMENT. High image-count rows (many real +
# filler pages, all sent inline as bytes) blow past that. To stay under the cap we
# downscale + re-encode each image to JPEG before sending. Tunable via env so the
# budget can be adjusted without code changes:
#   GOOGLE_IMAGE_MAX_EDGE       longest-edge px cap (0/negative disables resizing)
#   GOOGLE_IMAGE_JPEG_QUALITY   JPEG quality 1-95
#   GOOGLE_IMAGE_COMPRESS       set to 0/false to disable compression entirely
_IMAGE_MAX_EDGE_DEFAULT = 640
_IMAGE_JPEG_QUALITY_DEFAULT = 72

# Vertex AI request label (surfaces in Cloud Logging / billing). Label values must
# be lowercase and contain only letters, digits, dashes, and underscores, so the
# human-readable "Long Context Evaluation" is encoded as a slug.
_REQUEST_LABELS = {"feature": "long-context-evaluation"}


# --------------------------------------------------------------------------- #
# Explicit context-cache registry
# --------------------------------------------------------------------------- #
# Gemini's explicit caching API lets you pre-upload a stable context and
# reference it by name on subsequent generate_content calls, billing the
# cached tokens at ~4x cheaper read rate instead of full input rate.
#
# TTL is set via GOOGLE_CACHE_TTL_S (default 600 s / 10 min). Within each
# runner cache group the same context is sent back-to-back (once per thinking
# level), so even a short TTL guarantees a hit on the second call. Setting it
# longer than the experiment's expected group latency is sufficient.
#
# The registry is keyed on a SHA-256 of (model_id, encoded_content_bytes) so
# two groups with genuinely different contexts never collide. Cache entries
# are created lazily and deleted eagerly once they are no longer needed (or
# after TTL expiry on the Vertex side).

_CONTEXT_CACHE_TTL_S_DEFAULT = 600


class _ContextCacheRegistry:
    """Thread-safe, lazy registry of Gemini CachedContent resources."""

    def __init__(self, client, ttl_s: int) -> None:
        self._client = client
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        # key -> (cache_name, expire_epoch)
        self._entries: dict[str, tuple[str, float]] = {}

    @staticmethod
    def _cache_key(model: str, parts: list) -> str:
        """Stable hash of (model, serialised content sizes)."""
        from google.genai import types  # type: ignore
        h = hashlib.sha256()
        h.update(model.encode())
        for p in parts:
            if isinstance(p, types.Part):
                raw = getattr(p, "inline_data", None)
                if raw is not None:
                    h.update(getattr(raw, "data", b"") or b"")
                else:
                    h.update((getattr(p, "text", "") or "").encode())
            else:
                h.update(str(p).encode())
        return h.hexdigest()

    def get_or_create(self, model: str, parts: list, system: str | None) -> str | None:
        """Return a CachedContent resource name for the given context.

        Returns ``None`` if cache creation fails (caller should proceed
        without caching rather than raising).
        """
        from google.genai import types  # type: ignore

        key = self._cache_key(model, parts)
        now = time.time()

        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                cache_name, expire_epoch = entry
                if now < expire_epoch:
                    return cache_name
                # Sentinel ("", 0.0) means another thread is creating; skip.
                if cache_name == "" and expire_epoch == 0.0:
                    return None
                del self._entries[key]
            self._entries[key] = ("", 0.0)

        contents = [types.Content(role="user", parts=parts)]
        cfg_kwargs: dict = {"contents": contents, "ttl": f"{self._ttl_s}s"}
        if system:
            cfg_kwargs["system_instruction"] = system

        try:
            cache = self._client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(**cfg_kwargs),
            )
        except Exception as e:
            _log.warning("context cache creation failed (will send full context): %s", e)
            with self._lock:
                if self._entries.get(key) == ("", 0.0):
                    del self._entries[key]
            return None

        expire_epoch = now + self._ttl_s
        with self._lock:
            self._entries[key] = (cache.name, expire_epoch)
        _log.debug("created context cache %s (ttl=%ds)", cache.name, self._ttl_s)
        return cache.name

    def delete(self, model: str, parts: list) -> None:
        """Eagerly delete the cache entry after its last use."""
        key = self._cache_key(model, parts)
        with self._lock:
            entry = self._entries.pop(key, None)
        if entry is None:
            return
        cache_name, _ = entry
        try:
            self._client.caches.delete(name=cache_name)
            _log.debug("deleted context cache %s", cache_name)
        except Exception as e:
            _log.debug("context cache delete failed (will expire naturally): %s", e)


class GoogleProvider(Provider):
    """Gemini via Vertex AI.

    Auth: reads `GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS`.
    `GOOGLE_APPLICATION_CREDENTIALS` may be either:
      - a filesystem path to a service-account JSON file, OR
      - the service-account JSON content itself (a string starting with `{`).

    Optional `GOOGLE_CLOUD_LOCATION` (default `us-central1`).
    """
    name = "google"

    def __init__(self) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # noqa: F401
        except ImportError as e:
            raise PermanentProviderError(
                "google-genai SDK not installed. `pip install 'mlcr[google]'`"
            ) from e
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise PermanentProviderError("GOOGLE_CLOUD_PROJECT is not set")
        creds_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_env:
            raise PermanentProviderError("GOOGLE_APPLICATION_CREDENTIALS is not set")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        credentials = _load_credentials(creds_env)
        client_kwargs: dict = {"vertexai": True, "project": project, "location": location}
        if credentials is not None:
            client_kwargs["credentials"] = credentials
        # Service tier (default "flex"). Vertex AI does NOT accept `service_tier`
        # in the request body (it 400s); flex/priority must be selected via these
        # client HTTP headers instead. Standard/none -> no headers (server default).
        tier = _service_tier()
        http_opts: dict = {}
        if tier in ("flex", "priority"):
            http_opts["headers"] = {
                "X-Vertex-AI-LLM-Request-Type": "shared",
                "X-Vertex-AI-LLM-Shared-Request-Type": tier,
            }

        # Client-side request timeout. The SDK has no default, so a stalled or
        # throttled connection can block a worker thread forever; a timeout turns
        # that into a (retryable) DEADLINE_EXCEEDED instead. Override via
        # GOOGLE_REQUEST_TIMEOUT_S (seconds); 0/negative disables it.
        # Flex requests can sit in a queue for minutes (best-effort, off-peak
        # capacity), so when flex is active we default to a much longer timeout
        # (Google recommends >= 10 min) unless the user pins one explicitly.
        default_timeout = 600.0 if tier == "flex" else 120.0
        timeout_s = _env_float("GOOGLE_REQUEST_TIMEOUT_S", default_timeout)
        if timeout_s > 0:
            http_opts["timeout"] = int(timeout_s * 1000)
        if http_opts:
            client_kwargs["http_options"] = types.HttpOptions(**http_opts)
        self._client = genai.Client(**client_kwargs)

        # Build a fallback client without flex headers for requests that exceed
        # the flex payload limit (20 MB). Only needed when flex is active.
        self._fallback_client = None
        if tier in ("flex", "priority"):
            fb_kwargs = dict(client_kwargs)
            fb_http_opts: dict = {}
            if timeout_s > 0:
                fb_http_opts["timeout"] = int(timeout_s * 1000)
            if fb_http_opts:
                fb_kwargs["http_options"] = types.HttpOptions(**fb_http_opts)
            else:
                fb_kwargs.pop("http_options", None)
            self._fallback_client = genai.Client(**fb_kwargs)

        ttl_s = _env_int("GOOGLE_CACHE_TTL_S", _CONTEXT_CACHE_TTL_S_DEFAULT)
        self._cache_registry = _ContextCacheRegistry(self._client, ttl_s) if ttl_s > 0 else None

    def call(self, req: ChatRequest) -> ChatResponse:
        from google.genai import types  # type: ignore

        parts: list = []
        if req.user_text:
            parts.append(types.Part.from_text(text=req.user_text))
        for p in req.images:
            data, mime = _encode_image(p)
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))

        gen_cfg_kwargs: dict = {
            "temperature": req.model_cfg.temperature,
            "max_output_tokens": req.model_cfg.max_output_tokens,
        }
        if req.system:
            gen_cfg_kwargs["system_instruction"] = req.system
        if req.model_cfg.thinking:
            gen_cfg_kwargs["thinking_config"] = types.ThinkingConfig(**req.model_cfg.thinking)
        gen_cfg_kwargs["labels"] = dict(_REQUEST_LABELS)
        gen_cfg_kwargs.update(req.model_cfg.extra or {})

        # Context caching: cache all parts (text + images + system instruction).
        # The API requires non-empty contents even when cached_content is set,
        # so we send a blank text part as a placeholder.
        cache_name: str | None = None
        request_parts = parts
        if self._cache_registry is not None and parts:
            cache_name = self._cache_registry.get_or_create(
                req.model_cfg.model, parts, req.system
            )
            if cache_name is not None:
                gen_cfg_kwargs["cached_content"] = cache_name
                request_parts = [types.Part.from_text(text="")]
                gen_cfg_kwargs.pop("system_instruction", None)

        t0 = time.time()
        try:
            resp = self._client.models.generate_content(
                model=req.model_cfg.model,
                contents=request_parts,
                config=types.GenerateContentConfig(**gen_cfg_kwargs),
            )
        except Exception as e:
            if self._fallback_client is not None and _is_flex_fallback_error(e):
                _log.warning("flex not supported for this model/payload, retrying on standard tier")
                fb_cfg = dict(gen_cfg_kwargs)
                fb_cfg.pop("cached_content", None)
                try:
                    resp = self._fallback_client.models.generate_content(
                        model=req.model_cfg.model,
                        contents=parts,
                        config=types.GenerateContentConfig(**fb_cfg),
                    )
                except Exception as e2:
                    raise map_provider_error(e2)
            else:
                raise map_provider_error(e)
        latency_ms = int((time.time() - t0) * 1000)

        # Extract text and thinking parts from the response candidates.
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        candidates = getattr(resp, "candidates", None)
        if candidates:
            for part in (candidates[0].content.parts or []):
                if getattr(part, "thought", False):
                    if part.text:
                        thinking_parts.append(part.text)
                else:
                    if part.text:
                        text_parts.append(part.text)
        text = "".join(text_parts) if text_parts else (resp.text or "")

        usage = {}
        meta = getattr(resp, "usage_metadata", None)
        if meta:
            usage = {
                "prompt_tokens": getattr(meta, "prompt_token_count", None),
                "completion_tokens": getattr(meta, "candidates_token_count", None),
            }
            tt = getattr(meta, "thoughts_token_count", None)
            if tt is not None:
                usage["thinking_tokens"] = tt
            cached_tokens = getattr(meta, "cached_content_token_count", None)
            if cached_tokens:
                usage["cached_content_token_count"] = cached_tokens
        return ChatResponse(
            text=text,
            usage=usage,
            raw={},
            latency_ms=latency_ms,
            thinking="\n\n".join(thinking_parts) if thinking_parts else None,
        )


def _service_tier() -> str:
    """Gemini service tier for every request. Defaults to ``flex`` (50% cheaper,
    best-effort/queued capacity). Override via GOOGLE_SERVICE_TIER, e.g.
    ``standard`` / ``priority``; set to ``none``/``unspecified``/empty to omit the
    field entirely (server then uses its default)."""
    raw = (os.environ.get("GOOGLE_SERVICE_TIER") or "flex").strip().lower()
    if raw in ("", "none", "unspecified", "default"):
        return ""
    return raw


def _compression_enabled() -> bool:
    raw = os.environ.get("GOOGLE_IMAGE_COMPRESS")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _encode_image(path: Path) -> tuple[bytes, str]:
    """Return (bytes, mime_type) for an image, downscaled + re-encoded to JPEG.

    Vertex caps the total inline request at ~20 MB; shrinking each page keeps
    high-image-count rows under that limit. If compression is disabled or Pillow
    can't open the file, the original bytes are sent unchanged.
    """
    if not _compression_enabled():
        return path.read_bytes(), _MIME.get(path.suffix.lower(), "image/png")

    max_edge = _env_int("GOOGLE_IMAGE_MAX_EDGE", _IMAGE_MAX_EDGE_DEFAULT)
    quality = _env_int("GOOGLE_IMAGE_JPEG_QUALITY", _IMAGE_JPEG_QUALITY_DEFAULT)
    try:
        from PIL import Image  # type: ignore

        im = Image.open(path)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im).convert("RGB")
        else:
            im = im.convert("RGB")
        if max_edge > 0:
            w, h = im.size
            longest = max(w, h)
            if longest > max_edge:
                scale = max_edge / longest
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except OSError as e:
        _log.warning("image compression failed for %s, sending raw bytes: %s", path, e)
        return path.read_bytes(), _MIME.get(path.suffix.lower(), "image/png")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _load_credentials(value: str):
    """Parse GOOGLE_APPLICATION_CREDENTIALS as either inline JSON or a file path.

    Returns a `google.oauth2.service_account.Credentials` object when inline JSON
    is detected, or `None` to let the SDK auto-discover from the file path
    (preserving the standard behavior).
    """
    stripped = value.strip()
    if stripped.startswith("{"):
        try:
            info = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise PermanentProviderError(
                f"GOOGLE_APPLICATION_CREDENTIALS looks like JSON but is not parseable: {e}"
            )
        try:
            from google.oauth2 import service_account  # type: ignore
        except ImportError as e:
            raise PermanentProviderError(
                "google-auth not installed. `pip install 'mlcr[google]'`"
            ) from e
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        return service_account.Credentials.from_service_account_info(info, scopes=scopes)

    if not Path(stripped).is_file():
        raise PermanentProviderError(
            f"GOOGLE_APPLICATION_CREDENTIALS is neither inline JSON nor an existing file path: {stripped[:60]!r}"
        )
    return None  # let SDK pick up the file via the env var


def _is_flex_fallback_error(e: Exception) -> bool:
    """True if the error is a flex-specific rejection that should fall back to standard."""
    msg = str(e).lower()
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if code != 400:
        return False
    return "payload size exceeds" in msg or "flex api is not supported" in msg
