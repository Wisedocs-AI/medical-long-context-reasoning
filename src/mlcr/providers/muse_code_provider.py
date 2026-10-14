from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from mlcr.providers.base import (
    ChatRequest,
    ChatResponse,
    PermanentProviderError,
    Provider,
    TransientProviderError,
)

# Muse is an agent harness, not a raw model endpoint. These flags lock a
# run down to a single, tool-less model step in an empty workspace so the
# result is as close to a plain model completion as the CLI allows. Results
# from this provider measure Muse + model and must be labelled as such.
_LOCKDOWN_FLAGS = (
    "--preset", "empty",
    "--disable-shell",
    "--disable-write",
    "--disable-web-tools",
    "--disable-reminders",
    "--no-foreign-personal-context",
    "--no-session-log",
)

_TRANSIENT_MSG_SUBSTRINGS = (
    "rate limit", "rate_limit", "overloaded", "too many requests",
    "timed out", "timeout", "temporarily", "try again",
    "503", "502", "429",
)


def _classify_failure(message: str) -> Exception:
    low = message.lower()
    if any(s in low for s in _TRANSIENT_MSG_SUBSTRINGS):
        return TransientProviderError(message)
    return PermanentProviderError(message)


def _parse_events(stdout: str) -> tuple[str | None, str | None, dict]:
    """Return (final_text, failure_message, raw) from `muse exec --json` JSONL."""
    final_text: str | None = None
    failure: str | None = None
    raw: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") or {}
        ptype = event.get("payload_type")
        if ptype == "run.model.configured":
            raw["model_id"] = payload.get("model_id")
            raw["provider_id"] = payload.get("provider_id")
        elif ptype == "run.terminal.completed":
            final_text = payload.get("text") or ""
            raw["session_id"] = (event.get("stream") or {}).get("id")
        elif ptype == "run.terminal.failed":
            failure = payload.get("reason") or payload.get("text") or "muse run failed"
    return final_text, failure, raw


class MuseCodeProvider(Provider):
    """Meta models via the Muse CLI (`muse exec`), e.g. Muse Wonder Full.

    Muse Wonder Full is not available via any API; it is only reachable
    through a Muse subscription. This provider drives the CLI
    headlessly so that model can still be evaluated on the public dataset.

    Auth: whatever `muse` is logged in with (`muse login`), or `META_API_KEY`
    in the environment for the Meta provider. No API key is read by mlcr itself.
    Set `MUSE_BIN` to override the `muse` executable path.

    Limitations versus API providers: no temperature control (reasoning effort
    only), and the CLI does not report per-call token usage, so `usage` is
    empty and cost cannot be derived. `model:` is the Muse model id,
    e.g. `muse-wonder-full` or `muse-wonder-1.0`.

    Supported `extra:` keys in the model config: `reasoning_effort`
    (overrides the experiment thinking level), `max_model_steps` (default 1),
    `timeout_s` (default 3600), `contributor_mode` (default "off"),
    `workspace` (default: a fresh empty temp dir).
    """
    name = "muse_code"

    def __init__(self) -> None:
        self._bin = os.environ.get("MUSE_BIN") or shutil.which("muse")
        if not self._bin:
            raise PermanentProviderError(
                "Muse CLI (`muse`) not found on PATH. Install Muse "
                "and run `muse login`, or set MUSE_BIN to the executable."
            )
        self._workspace = Path(tempfile.mkdtemp(prefix="mlcr_muse_code_"))

    def _command(self, req: ChatRequest, prompt_file: Path) -> list[str]:
        extra = req.model_cfg.extra or {}
        effort = extra.get("reasoning_effort") or (req.model_cfg.thinking or {}).get("effort")
        cmd = [
            self._bin, "exec", "--json",
            *_LOCKDOWN_FLAGS,
            "--workspace", str(extra.get("workspace") or self._workspace),
            "--model", req.model_cfg.model,
            "--contributor-mode", str(extra.get("contributor_mode", "off")),
            "--max-model-steps", str(int(extra.get("max_model_steps", 1))),
        ]
        if effort:
            cmd += ["--reasoning-effort", str(effort)]
        for image in req.images:
            cmd += ["--image", str(image)]
        cmd += ["--prompt-file", str(prompt_file)]
        return cmd

    def call(self, req: ChatRequest) -> ChatResponse:
        parts = [p for p in (req.system, req.prefix_text, req.user_text) if p]
        prompt = "\n\n".join(parts)
        extra = req.model_cfg.extra or {}
        timeout_s = float(extra.get("timeout_s", 3600))

        prompt_path: Path | None = None
        t0 = time.time()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="mlcr_prompt_", delete=False
            ) as f:
                f.write(prompt)
                prompt_path = Path(f.name)
            env = {**os.environ, "MUSE_NO_AUTO_UPDATE": "1"}
            try:
                proc = subprocess.run(
                    self._command(req, prompt_path),
                    capture_output=True, text=True, timeout=timeout_s, env=env,
                )
            except subprocess.TimeoutExpired:
                raise TransientProviderError(
                    f"muse exec timed out after {timeout_s:.0f}s"
                )
            except OSError as e:
                raise PermanentProviderError(f"failed to launch muse: {e}")
        finally:
            if prompt_path is not None:
                prompt_path.unlink(missing_ok=True)
        latency_ms = int((time.time() - t0) * 1000)

        text, failure, raw = _parse_events(proc.stdout)
        if text is None:
            message = failure or (proc.stderr or "").strip().splitlines()
            if isinstance(message, list):
                message = message[-1] if message else f"muse exited {proc.returncode}"
            raise _classify_failure(f"muse exec failed: {message}")
        raw["exit_code"] = proc.returncode
        return ChatResponse(text=text, usage={}, raw=raw, latency_ms=latency_ms)
