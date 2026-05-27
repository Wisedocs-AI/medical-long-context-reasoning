from __future__ import annotations

import csv
import json
import logging
import shutil
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mlcr.cases import (
    filter_filler_by_forms,
    list_filler_images,
    list_filler_text,
    load_case,
)
from mlcr.config import ExperimentConfig
from mlcr.filler import BuiltContext, build_context, make_rng
from mlcr.logging_setup import setup_logging
from mlcr.matrix import Row, build_matrix, format_ratio
from mlcr.providers import registry
from mlcr.thinking import apply_thinking
from mlcr.providers.base import (
    ChatRequest,
    PermanentProviderError,
    Provider,
    ProviderError,
    TransientProviderError,
)

log = logging.getLogger("mlcr")


class Runner:
    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        force: bool = False,
        dry_run: bool = False,
        limit: int | None = None,
        confirm: bool = False,
        experiment_uuid: str | None = None,
        run_dir: Path | None = None,
        use_previous_runs: bool = True,
        config_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.force = force
        self.dry_run = dry_run
        self.limit = limit
        self.confirm = confirm
        self.use_previous_runs = use_previous_runs
        self.config_path = config_path
        self._prev_run_dirs: list[Path] = []
        self.experiment_uuid = experiment_uuid or str(uuid.uuid4())
        if run_dir is None:
            run_dir = (cfg.repo_root or Path.cwd()) / cfg.output_dir / self.experiment_uuid
        self.run_dir = run_dir
        self.results_dir = run_dir / "results"
        self.summary_jsonl = run_dir / "summary.jsonl"
        self.summary_csv = run_dir / "summary.csv"
        self._summary_lock = threading.Lock()
        self._provider_sems: dict[str, threading.Semaphore] = {}
        self._provider_cache: dict[str, Provider] = {}
        self._provider_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._cache_hits = 0
        self._api_calls = 0
        self._rows_total = 0

    # ---------- public ----------

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.run_dir / "logs")

        if self.config_path is not None and self.config_path.is_file():
            dest = self.run_dir / "config.yaml"
            if not dest.exists():
                shutil.copy2(self.config_path, dest)
                log.info("copied config %s -> %s", self.config_path, dest)

        if self.use_previous_runs:
            self._prev_run_dirs = self._discover_previous_run_dirs()
            log.info("use_previous_runs enabled: scanning %d prior run dir(s)",
                     len(self._prev_run_dirs))

        rows = build_matrix(self.cfg)
        if self.limit is not None:
            rows = rows[: self.limit]

        self._freeze_run(rows)
        matrix_path = self._write_matrix_preview(rows)
        log.info("experiment=%s rows=%d dry_run=%s", self.experiment_uuid, len(rows), self.dry_run)

        if self.confirm and not self._confirm_proceed(rows, matrix_path):
            log.info("run aborted by user before execution")
            return {
                "experiment_uuid": self.experiment_uuid,
                "run_dir": str(self.run_dir),
                "counts": {},
                "aborted": True,
            }

        rows_to_do = [r for r in rows if self.force or not self._is_done(r)]
        log.info("skipping %d completed rows, executing %d", len(rows) - len(rows_to_do), len(rows_to_do))
        self._rows_total = len(rows_to_do)

        # Group rows that share everything except the thinking level so that all
        # thinking variants for the same (case, prompt, model, modality, ablation)
        # run sequentially on the same thread. The context/prompt is built once per
        # group and reused, keeping the prompt byte-identical across variants so the
        # provider can reuse its prompt cache.
        groups = self._group_rows(rows_to_do)
        log.info("dispatching %d rows in %d cache groups", len(rows_to_do), len(groups))

        counts = defaultdict(int)
        max_workers = self.cfg.concurrency.max_workers
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self._run_group, g) for g in groups]
            for fut in as_completed(futures):
                for status in fut.result():
                    counts[status] += 1

        if self._rows_total:
            print(
                f"\n  final: {self._rows_total}/{self._rows_total} (100%)  "
                f"cache={self._cache_hits}  api_calls={self._api_calls}",
                flush=True,
            )

        self._rebuild_csv()
        self._write_responses_csv()
        log.info("done: %s", dict(counts))
        return {"experiment_uuid": self.experiment_uuid, "run_dir": str(self.run_dir), "counts": dict(counts)}

    def _record_progress(self, *, cache_hit: bool) -> None:
        """Increment cache/API counters and print a running tally to stderr."""
        with self._progress_lock:
            if cache_hit:
                self._cache_hits += 1
            else:
                self._api_calls += 1
            done = self._cache_hits + self._api_calls
            total = self._rows_total
            pct = f"{100 * done / total:.0f}%" if total else "?"
            print(
                f"\r  progress: {done}/{total} ({pct})  "
                f"cache={self._cache_hits}  api_calls={self._api_calls}   ",
                end="",
                flush=True,
            )

    # ---------- internals ----------

    def _freeze_run(self, rows: list[Row]) -> None:
        out = {
            "experiment_uuid": self.experiment_uuid,
            "config": self.cfg.model_dump(mode="json", exclude={"repo_root", "config_dir"}),
            "matrix_size": len(rows),
        }
        (self.run_dir / "run.json").write_text(json.dumps(out, indent=2, default=str))

    def _write_matrix_preview(self, rows: list[Row]) -> Path:
        """Write every combination (with its row_id) to a .txt file for review."""
        path = self.run_dir / "matrix.txt"
        lines = [
            f"experiment: {self.cfg.name}",
            f"experiment_uuid: {self.experiment_uuid}",
            f"total combinations: {len(rows)}",
            "",
        ]
        for i, r in enumerate(rows, 1):
            rep_str = f"  rep={r.repetition}" if self.cfg.repetitions > 1 else ""
            lines.append(
                f"{i:>4}. {r.row_id}  case={r.case_uuid}  case_specific_prompt={r.prompt_path.name}  "
                f"model={r.model_config_id}  modality={r.modality}  "
                f"junk_context_ratio={format_ratio(r.ablation_ratio)}  thinking={r.thinking}{rep_str}"
            )
        path.write_text("\n".join(lines) + "\n")
        return path

    def _confirm_proceed(self, rows: list[Row], matrix_path: Path) -> bool:
        """Prompt the user to review the matrix preview before executing."""
        print(f"\nPlanned {len(rows)} combinations. Review the full matrix at:\n  {matrix_path}")
        try:
            answer = input("Proceed with the run? [y/N]: ").strip().lower()
        except EOFError:
            # Non-interactive stdin: do not proceed without explicit confirmation.
            print("No interactive input available; aborting. Re-run with --yes to skip this prompt.")
            return False
        return answer in ("y", "yes")

    def _discover_previous_run_dirs(self) -> list[Path]:
        """Sibling run directories from earlier experiments, most recent first.

        The current run directory is excluded. Each row's identity (row_id) is a
        deterministic hash of case+prompt+model+modality+ablation+thinking, so a
        matching row_id in any of these dirs is the same experiment configuration
        with the same (unchanging) input and can be reused verbatim."""
        runs_root = self.run_dir.parent
        if not runs_root.is_dir():
            return []
        dirs = [
            d for d in runs_root.iterdir()
            if d.is_dir() and d.resolve() != self.run_dir.resolve()
        ]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return dirs

    def _find_previous_result(self, row: Row) -> dict | None:
        """Return the most recent successful prior result for this row, if any."""
        for d in self._prev_run_dirs:
            p = d / "results" / row.row_id / "result.json"
            if not p.is_file():
                continue
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            if data.get("metadata", {}).get("status") == "ok":
                return data
        return None

    def _is_done(self, row: Row) -> bool:
        p = self.results_dir / row.row_id / "result.json"
        if not p.is_file():
            return False
        try:
            data = json.loads(p.read_text())
        except Exception:
            return False
        return data.get("metadata", {}).get("status") == "ok"

    def _provider_for(self, name: str) -> Provider:
        with self._provider_lock:
            if name not in self._provider_cache:
                self._provider_cache[name] = registry.get(name)
            return self._provider_cache[name]

    def _provider_sem(self, name: str) -> threading.Semaphore | None:
        if name not in self._provider_sems:
            limit = self.cfg.concurrency.per_provider.get(name)
            if limit:
                self._provider_sems[name] = threading.Semaphore(limit)
        return self._provider_sems.get(name)

    @staticmethod
    def _cache_group_key(row: Row) -> tuple:
        """Identity of a row ignoring thinking level. Rows sharing this key have an
        identical prompt/context and only differ in the thinking parameter, so they
        can share a prompt cache when issued back-to-back on one thread.

        Repetition is included so that independent repetitions never share a
        provider-side prompt cache, giving truly independent responses."""
        return (row.case_uuid, str(row.prompt_path), row.model_config_id,
                row.modality, row.ablation_ratio, row.repetition)

    def _group_rows(self, rows: list[Row]) -> list[list[Row]]:
        groups: dict[tuple, list[Row]] = {}
        order: list[tuple] = []
        for r in rows:
            key = self._cache_group_key(r)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(r)
        return [groups[k] for k in order]

    def _run_group(self, rows: list[Row]) -> list[str]:
        """Run all thinking variants of one cache group sequentially in this thread.

        The context and prompt are built once and reused so the prompt stays
        byte-identical across thinking variants, maximizing prompt-cache hits."""
        if not rows:
            return []
        ctx: BuiltContext | None = None
        prompt_text: str | None = None
        build_error: Exception | None = None
        try:
            ctx = self._build_context(rows[0])
            prompt_text = self._compose_prompt(rows[0], ctx)
        except Exception as e:  # build failed: propagate to every variant
            build_error = e
        return [
            self._run_row(row, ctx=ctx, prompt_text=prompt_text, build_error=build_error)
            for row in rows
        ]

    def _run_row(
        self,
        row: Row,
        *,
        ctx: BuiltContext | None = None,
        prompt_text: str | None = None,
        build_error: Exception | None = None,
    ) -> str:
        mcfg = apply_thinking(self.cfg.model_configs[row.model_config_id], row.thinking)
        row_dir = self.results_dir / row.row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        log.info("row=%s case=%s model=%s mod=%s ratio=%s thinking=%s", row.row_id, row.case_uuid,
                 row.model_config_id, row.modality, format_ratio(row.ablation_ratio), row.thinking)

        try:
            if build_error is not None:
                raise build_error
            if ctx is None or prompt_text is None:
                ctx = self._build_context(row)
                prompt_text = self._compose_prompt(row, ctx)
            (row_dir / "prompt.txt").write_text(prompt_text)
            (row_dir / "images.json").write_text(json.dumps([str(p) for p in ctx.images], indent=2))

            if self.dry_run:
                return self._write_result(row, mcfg, ctx, prompt_text,
                                          answer="", usage={}, latency_ms=0, status="dry_run", error=None)

            if self.use_previous_runs:
                prev = self._find_previous_result(row)
                if prev is not None:
                    pm = prev.get("metadata", {})
                    src = pm.get("experiment_uuid") or prev.get("experiment_uuid")
                    log.info("row=%s reusing result from experiment=%s", row.row_id, src)
                    self._record_progress(cache_hit=True)
                    return self._write_result(
                        row, mcfg, ctx, prompt_text,
                        answer=prev.get("answer", ""),
                        usage=pm.get("usage", {}),
                        latency_ms=pm.get("latency_ms", 0),
                        status="ok", error=None, reused_from=src,
                        thinking=prev.get("thinking"),
                    )

            req = ChatRequest(
                system=None,
                user_text=prompt_text,
                images=ctx.images,
                model_cfg=mcfg,
            )
            provider = self._provider_for(mcfg.provider)
            self._record_progress(cache_hit=False)
            resp = self._call_with_retry(provider, req, mcfg.provider)
            return self._write_result(row, mcfg, ctx, prompt_text,
                                      answer=resp.text, usage=resp.usage, latency_ms=resp.latency_ms,
                                      status="ok", error=None, thinking=resp.thinking)
        except PermanentProviderError as e:
            log.error("row=%s permanent error: %s", row.row_id, e)
            return self._write_result(row, mcfg, None, "", "", {}, 0, "error",
                                      {"type": type(e).__name__, "message": str(e)})
        except Exception as e:
            log.exception("row=%s unexpected error", row.row_id)
            return self._write_result(row, mcfg, None, "", "", {}, 0, "error",
                                      {"type": type(e).__name__, "message": str(e)})

    def _build_context(self, row: Row) -> BuiltContext:
        repo = self.cfg.repo_root
        assert repo is not None
        case = load_case(repo, row.case_uuid)
        allowed_forms = self.cfg.allowed_filler_forms
        filler_text = filter_filler_by_forms(list_filler_text(repo, filler_subdir=self.cfg.filler_subdir), allowed_forms)
        filler_imgs = filter_filler_by_forms(list_filler_images(repo, filler_subdir=self.cfg.filler_subdir), allowed_forms)
        # Filler RNG depends only on case + ratio so all prompts, models, and
        # modalities for the same case at the same ratio see identical filler.
        rng_key = f"{row.case_uuid}|{format_ratio(row.ablation_ratio)}"
        rng = make_rng(self.cfg.seed, rng_key)
        return build_context(
            case_summaries=case.summaries,
            case_images=case.images,
            filler_text_files=filler_text,
            filler_image_files=filler_imgs,
            modality=row.modality,
            ablation_ratio=row.ablation_ratio,
            rng=rng,
        )

    def _compose_prompt(self, row: Row, ctx: BuiltContext) -> str:
        question = row.prompt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not question:
            raise PermanentProviderError(f"prompt file is empty: {row.prompt_path}")
        if ctx.text is None:
            return question
        return f"{ctx.text}\n\n---\n\nQuestion:\n{question}"

    def _call_with_retry(self, provider: Provider, req: ChatRequest, provider_name: str):
        sem = self._provider_sem(provider_name)
        attempt = 0
        backoff = self.cfg.retry.initial_backoff_s
        while True:
            attempt += 1
            try:
                if sem is not None:
                    with sem:
                        return provider.call(req)
                return provider.call(req)
            except TransientProviderError as e:
                if attempt >= self.cfg.retry.max_attempts:
                    raise
                log.warning("transient error (attempt %d/%d): %s", attempt, self.cfg.retry.max_attempts, e)
                time.sleep(min(backoff, self.cfg.retry.max_backoff_s))
                backoff = min(backoff * 2, self.cfg.retry.max_backoff_s)
            except ProviderError:
                raise

    def _write_result(self, row: Row, mcfg, ctx: BuiltContext | None, prompt_text: str,
                      answer: str, usage: dict, latency_ms: int, status: str, error,
                      reused_from: str | None = None, thinking: str | None = None) -> str:
        row_dir = self.results_dir / row.row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        filler_meta = _filler_meta(ctx) if ctx else {}
        result = {
            "experiment_uuid": self.experiment_uuid,
            "question": prompt_text,
            "answer": answer,
            "thinking": thinking,
            "metadata": {
                "row_id": row.row_id,
                "case_uuid": row.case_uuid,
                "prompt_path": str(row.prompt_path),
                "model_config_id": row.model_config_id,
                "provider": mcfg.provider,
                "model": mcfg.model,
                "modality": row.modality,
                "ablation_ratio": row.ablation_ratio,
                "thinking": row.thinking,
                "difficulty": row.difficulty,
                "repetition": row.repetition,
                "seed": self.cfg.seed,
                "usage": usage,
                "latency_ms": latency_ms,
                "filler": filler_meta,
                "status": status,
                "error": error,
                "reused_from": reused_from,
            },
        }
        (row_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

        summary_row = {
            "experiment_uuid": self.experiment_uuid,
            "row_id": row.row_id,
            "case_uuid": row.case_uuid,
            "prompt_path": str(row.prompt_path),
            "model_config_id": row.model_config_id,
            "provider": mcfg.provider,
            "model": mcfg.model,
            "modality": row.modality,
            "ablation_ratio": row.ablation_ratio,
            "thinking": row.thinking,
            "difficulty": row.difficulty,
            "repetition": row.repetition,
            "seed": self.cfg.seed,
            "usage": usage,
            "latency_ms": latency_ms,
            "filler_text_files": filler_meta.get("text_files", []),
            "filler_image_files": filler_meta.get("image_files", []),
            "prompt_artifact": str(row_dir / "prompt.txt"),
            "result_path": str(row_dir / "result.json"),
            "status": status,
            "error": error,
            "reused_from": reused_from,
        }
        with self._summary_lock:
            with self.summary_jsonl.open("a") as f:
                f.write(json.dumps(summary_row, default=str) + "\n")
        return "reused" if reused_from else status

    def _rebuild_csv(self) -> None:
        if not self.summary_jsonl.exists():
            return
        rows = [json.loads(line) for line in self.summary_jsonl.read_text().splitlines() if line.strip()]
        if not rows:
            return
        # The jsonl is append-only, so a resumed/forced re-run can write a row_id more
        # than once. Keep only the last occurrence per row_id (the most recent result).
        deduped: dict[str, dict] = {}
        for r in rows:
            deduped[r.get("row_id")] = r
        rows = list(deduped.values())
        fields = [
            "experiment_uuid", "row_id", "case_uuid", "prompt_path",
            "model_config_id", "provider", "model", "modality", "ablation_ratio",
            "thinking", "difficulty", "repetition", "seed", "latency_ms", "status", "error", "reused_from",
            "prompt_artifact", "result_path",
        ]
        with self.summary_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), (dict, list)) else r.get(k)) for k in fields})

    def _write_responses_csv(self) -> None:
        """Human-readable CSV: one row per result with the case-specific prompt text
        (from the .txt file) alongside the model's response."""
        fields = [
            "case_uuid", "prompt", "difficulty", "human_validated_answer", "model",
            "modality", "junk_context_ratio", "thinking", "repetition", "input_tokens",
            "output_tokens", "thinking_tokens", "response",
        ]
        out = self.run_dir / "responses.csv"
        prompt_cache: dict[str, str] = {}
        answer_cache: dict[str, str] = {}
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for result_path in sorted(self.results_dir.glob("*/result.json")):
                try:
                    data = json.loads(result_path.read_text())
                except Exception:
                    continue
                meta = data.get("metadata", {})
                prompt_path = meta.get("prompt_path", "")
                if prompt_path not in prompt_cache:
                    try:
                        prompt_cache[prompt_path] = Path(prompt_path).read_text(
                            encoding="utf-8", errors="replace"
                        ).strip()
                    except Exception:
                        prompt_cache[prompt_path] = ""
                if prompt_path not in answer_cache:
                    answer_cache[prompt_path] = _human_validated_answer(prompt_path)
                usage = meta.get("usage") or {}
                w.writerow({
                    "case_uuid": meta.get("case_uuid", ""),
                    "prompt": prompt_cache[prompt_path],
                    "difficulty": meta.get("difficulty", ""),
                    "human_validated_answer": answer_cache[prompt_path],
                    "model": meta.get("model", ""),
                    "modality": meta.get("modality", ""),
                    "junk_context_ratio": meta.get("ablation_ratio", ""),
                    "thinking": meta.get("thinking", ""),
                    "repetition": meta.get("repetition", 0),
                    "input_tokens": (
                        (usage.get("prompt_tokens") or 0)
                        + (usage.get("cache_creation_input_tokens") or 0)
                        + (usage.get("cache_read_input_tokens") or 0)
                    ) or "",
                    "output_tokens": usage.get("completion_tokens", ""),
                    "thinking_tokens": usage.get("thinking_tokens", 0),
                    "response": data.get("answer", ""),
                })
        log.info("wrote responses csv: %s", out)


def _human_validated_answer(prompt_path: str) -> str:
    """Resolve the human-validated answer for a case-specific prompt.

    Prompts live at ``cases/<case>/prompts/qNN.txt`` and the corresponding
    validated answer at ``cases/<case>/answers/aNN.txt`` (same number)."""
    if not prompt_path:
        return ""
    p = Path(prompt_path)
    stem = p.stem  # e.g. "q01"
    if not stem or stem[0] != "q":
        return ""
    answer_path = p.parent.parent / "answers" / f"a{stem[1:]}{p.suffix}"
    try:
        return answer_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def _filler_meta(ctx: BuiltContext) -> dict:
    return {
        "text_files": [str(p) for p in ctx.filler_text_files],
        "image_files": [str(p) for p in ctx.filler_image_files],
        "insertion_positions": ctx.insertion_positions,
        "real_page_count": ctx.real_page_count,
        "filler_page_count": ctx.filler_page_count,
    }
