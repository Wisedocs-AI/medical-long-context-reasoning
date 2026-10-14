from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mlcr.config import ModelConfig  # noqa: E402
from mlcr.providers.base import (  # noqa: E402
    ChatRequest,
    PermanentProviderError,
    TransientProviderError,
)
from mlcr.providers.muse_code_provider import MuseCodeProvider, _parse_events  # noqa: E402
from mlcr.thinking import apply_thinking  # noqa: E402

STUB = r'''
import json, os, sys
argv_file = os.environ.get("MLCR_STUB_ARGV")
if argv_file:
    with open(argv_file, "w") as f:
        json.dump(sys.argv[1:], f)
mode = os.environ.get("MLCR_STUB_MODE", "success")
def emit(ptype, payload):
    print(json.dumps({"payload_type": ptype,
                      "stream": {"kind": "session", "id": "sess-1"},
                      "payload": payload}), flush=True)
emit("run.model.configured", {"model_id": "muse-wonder-full", "provider_id": "meta"})
if mode == "success":
    emit("run.terminal.completed", {"kind": "run_terminal", "terminal": "completed",
                                    "text": "stub answer", "reason": None})
elif mode == "fail_access":
    emit("run.terminal.failed", {"kind": "run_terminal", "terminal": "failed",
                                 "reason": "model `muse-wonder-full` does not exist or you lack access"})
    sys.exit(1)
elif mode == "fail_transient":
    emit("run.terminal.failed", {"kind": "run_terminal", "terminal": "failed",
                                 "reason": "rate limit exceeded, try again later"})
    sys.exit(1)
'''


@pytest.fixture()
def stub_muse(tmp_path, monkeypatch):
    stub = tmp_path / "muse"
    stub.write_text("#!" + sys.executable + "\n" + STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("MUSE_BIN", str(stub))
    monkeypatch.setenv("MLCR_STUB_ARGV", str(argv_file))
    return argv_file


def _request(tmp_path, **cfg_kwargs):
    cfg = ModelConfig(id="muse-wonder-full", provider="muse_code",
                      model="muse-wonder-full", **cfg_kwargs)
    return ChatRequest(system=None, user_text="case text\n\nQuestion: what happened?",
                       model_cfg=cfg)


def test_success_parses_terminal_text_and_lockdown_flags(stub_muse, tmp_path):
    provider = MuseCodeProvider()
    resp = provider.call(_request(tmp_path))
    assert resp.text == "stub answer"
    assert resp.usage == {}
    assert resp.raw["model_id"] == "muse-wonder-full"
    assert resp.raw["session_id"] == "sess-1"

    argv = json.loads(Path(stub_muse).read_text())
    assert argv[:2] == ["exec", "--json"]
    for flag in ("--disable-shell", "--disable-write", "--disable-web-tools",
                 "--no-foreign-personal-context", "--no-session-log"):
        assert flag in argv
    assert argv[argv.index("--preset") + 1] == "empty"
    assert argv[argv.index("--model") + 1] == "muse-wonder-full"
    assert argv[argv.index("--max-model-steps") + 1] == "1"
    prompt_arg = argv[argv.index("--prompt-file") + 1]
    assert not os.path.exists(prompt_arg)  # temp prompt file cleaned up


def test_thinking_level_maps_to_reasoning_effort(stub_muse, tmp_path):
    cfg = ModelConfig(id="muse-wonder-full", provider="muse_code", model="muse-wonder-full")
    mapped = apply_thinking(cfg, "high")
    assert mapped.thinking == {"effort": "high"}
    provider = MuseCodeProvider()
    provider.call(ChatRequest(system=None, user_text="q", model_cfg=mapped))
    argv = json.loads(Path(stub_muse).read_text())
    assert argv[argv.index("--reasoning-effort") + 1] == "high"


def test_access_failure_is_permanent(stub_muse, tmp_path, monkeypatch):
    monkeypatch.setenv("MLCR_STUB_MODE", "fail_access")
    with pytest.raises(PermanentProviderError):
        MuseCodeProvider().call(_request(tmp_path))


def test_rate_limit_failure_is_transient(stub_muse, tmp_path, monkeypatch):
    monkeypatch.setenv("MLCR_STUB_MODE", "fail_transient")
    with pytest.raises(TransientProviderError):
        MuseCodeProvider().call(_request(tmp_path))


def test_missing_binary_is_permanent(monkeypatch):
    monkeypatch.delenv("MUSE_BIN", raising=False)
    monkeypatch.setattr("mlcr.providers.muse_code_provider.shutil.which", lambda _: None)
    with pytest.raises(PermanentProviderError):
        MuseCodeProvider()


def test_parse_events_ignores_non_json_lines():
    text, failure, _ = _parse_events('noise\n{"payload_type": "run.terminal.completed", "payload": {"text": "ok"}, "stream": {"id": "s"}}')
    assert text == "ok"
    assert failure is None
