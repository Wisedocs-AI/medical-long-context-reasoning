from __future__ import annotations

"""Multi-gate, multi-model LLM-as-a-judge scorer.

Implements the evaluation pipeline described in docs/llm_judge.md:

  Gate 0 (free)  — Conciseness: whitespace-stripped char count <= 3x reference
  Gate 1 (LLM)  — Completeness + Accuracy: single call per model, 3-model majority vote

Writes judge_* columns into results_scoring.csv.

Runnable standalone:
    python -m mlcr.scoring runs/<experiment_uuid>
    mlcr score runs/<experiment_uuid>
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_JUDGE_COLS = [
    "judge_concise",
    "judge_complete",
    "judge_accurate",
    "judge_correct",
    "judge_votes",
    "judge_rationales",
    "judge_input_tokens",
    "judge_output_tokens",
    "judge_thinking_tokens",
    "judge_error",
]

_DEFAULT_JUDGE_MODELS = [
    "gemini-3.1-pro-preview",
    "claude-opus-4-8-gcp",
    "gpt-5.5",
]

_JUDGE_SYSTEM = (
    "You are a meticulous evaluation judge for a long-context medical document "
    "question-answering benchmark. You are given a question, a human-validated "
    "reference answer (the ground truth), and a model's response. Evaluate the "
    "response on ALL of the following criteria:\n\n"
    "A response PASSES when ALL hold:\n\n"
    "1. COMPLETENESS\n"
    "   a. Field coverage: every field the question asks for has a corresponding "
    "answer in the response (no missing fields).\n"
    "   b. Detail completeness: all factual detail present in the reference answer "
    "is captured by the response (no missing information).\n\n"
    "2. ACCURACY\n"
    "   a. Severity preservation: the response does not upgrade, downgrade, or alter "
    "the severity or intensity of any condition, injury, finding, or symptom "
    "compared to the reference (e.g., \"mild\" must not become \"moderate\", "
    "\"partial tear\" must not become \"complete tear\", \"improving\" must not "
    "become \"worsening\").\n"
    "   b. Factual accuracy: every fact stated in the response (dates, names, body "
    "parts, measurements, dosages, diagnoses, providers, procedures) matches "
    "the reference exactly. No values are changed, swapped, or fabricated.\n\n"
    "A response FAILS if ANY of these are true:\n"
    "- A field the question asks for is unanswered\n"
    "- Detail from the reference is missing\n"
    "- Severity is changed in either direction (upgrade or downgrade)\n"
    "- Any date, name, number, body part, diagnosis, or measurement is altered\n"
    "- Information is misattributed (e.g., swaps which provider said what)\n"
    "- Fabricated facts not present in the reference are introduced\n\n"
    "Do NOT penalize:\n"
    "- Differences in formatting, phrasing, wording, or ordering\n"
    "- Equivalent representations (e.g., \"10/25/2022\" vs \"October 25, 2022\")\n"
    "- Extra harmless detail beyond what the reference provides, as long as nothing "
    "is missing or incorrect\n\n"
    "Judge meaning, not surface form. Respond with ONLY a single JSON object (no "
    "markdown fencing), using exactly these keys:\n\n"
    '{"complete": boolean, "accurate": boolean, "rationale": string}\n\n'
    "CRITICAL: The rationale MUST be at most 1-2 short sentences (under 150 characters). "
    "If failing, name only the single most important issue. Be extremely brief."
)


# ---------------------------------------------------------------------------
# Deterministic heuristic metrics (no network access needed)
# ---------------------------------------------------------------------------
_HEURISTIC_COLS = ["exact_match", "token_f1", "rouge1_f1", "rouge2_f1", "rougeL_f1", "bleu"]

SCORING_COLS = _HEURISTIC_COLS + _JUDGE_COLS

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation/articles, squeeze ws."""
    s = (s or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    tokens = [t for t in s.split(" ") if t and t not in _ARTICLES]
    return " ".join(tokens)


def _tokens(s: str) -> list[str]:
    norm = normalize_text(s)
    return norm.split(" ") if norm else []


def exact_match(reference: str, prediction: str) -> float:
    return 1.0 if normalize_text(reference) == normalize_text(prediction) else 0.0


def token_f1(reference: str, prediction: str) -> float:
    """SQuAD-style token-level F1 over multiset overlap."""
    ref = _tokens(reference)
    pred = _tokens(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    common = Counter(ref) & Counter(pred)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1(reference: str, prediction: str, n: int) -> float:
    """ROUGE-N F1 over n-gram multiset overlap."""
    ref = _tokens(reference)
    pred = _tokens(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    ref_ng = _ngrams(ref, n)
    pred_ng = _ngrams(pred, n)
    if not ref_ng and not pred_ng:
        return 1.0 if ref == pred else 0.0
    if not ref_ng or not pred_ng:
        return 0.0
    overlap = sum((ref_ng & pred_ng).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_ng.values())
    recall = overlap / sum(ref_ng.values())
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (rolling-row DP)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    """ROUGE-L F1 based on the longest common subsequence of tokens."""
    ref = _tokens(reference)
    pred = _tokens(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    lcs = _lcs_length(ref, pred)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def bleu(reference: str, prediction: str, max_n: int = 4) -> float:
    """Sentence-level BLEU with brevity penalty and add-1 smoothing."""
    ref = _tokens(reference)
    pred = _tokens(prediction)
    if not ref and not pred:
        return 1.0
    if not pred or not ref:
        return 0.0

    log_precisions: list[float] = []
    for n in range(1, max_n + 1):
        pred_ng = _ngrams(pred, n)
        total = sum(pred_ng.values())
        if total == 0:
            log_precisions.append(math.log(1e-9))
            continue
        ref_ng = _ngrams(ref, n)
        overlap = sum((pred_ng & ref_ng).values())
        precision = (overlap + 1) / (total + 1)
        log_precisions.append(math.log(precision))

    geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
    ref_len, pred_len = len(ref), len(pred)
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    return bp * geo_mean


def heuristic_scores(reference: str, prediction: str) -> dict[str, float]:
    """Compute all deterministic metrics for a single (reference, prediction) pair."""
    return {
        "exact_match": round(exact_match(reference, prediction), 4),
        "token_f1": round(token_f1(reference, prediction), 4),
        "rouge1_f1": round(rouge_n_f1(reference, prediction, 1), 4),
        "rouge2_f1": round(rouge_n_f1(reference, prediction, 2), 4),
        "rougeL_f1": round(rouge_l_f1(reference, prediction), 4),
        "bleu": round(bleu(reference, prediction), 4),
    }


# ---------------------------------------------------------------------------
# Gate 0: Conciseness
# ---------------------------------------------------------------------------


def stripped_char_count(text: str) -> int:
    return len(_WS_RE.sub("", text or ""))


def passes_conciseness_gate(response: str, reference: str) -> bool:
    ref_len = stripped_char_count(reference)
    if ref_len == 0:
        return stripped_char_count(response) == 0
    return stripped_char_count(response) <= 3 * ref_len


# ---------------------------------------------------------------------------
# Gate 1: LLM judge (completeness + accuracy)
# ---------------------------------------------------------------------------
def _build_user_text(question: str, reference: str, response: str) -> str:
    return (
        "QUESTION:\n"
        f"{question}\n\n"
        "REFERENCE ANSWER (ground truth):\n"
        f"{reference}\n\n"
        "MODEL RESPONSE (to evaluate):\n"
        f"{response}\n"
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"no JSON object in judge output: {text[:200]!r}")


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return None


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------
class Judge:
    """Single-model judge instance. One per jury member."""

    def __init__(self, model_config_path: Path) -> None:
        from mlcr.config import ModelConfig
        from mlcr.providers.registry import get
        from mlcr.thinking import apply_thinking

        base_cfg = ModelConfig.load(model_config_path)
        self._cfg = apply_thinking(base_cfg, "none")
        self._cfg.temperature = 1.0
        self._cfg.max_output_tokens = 2048
        self._apply_billing_labels()

        self._provider = get(self._cfg.provider)
        self._model_id = base_cfg.id

    def _apply_billing_labels(self) -> None:
        if self._cfg.provider != "google":
            return
        try:
            from mlcr.providers.google_provider import _REQUEST_LABELS
            project_labels = dict(_REQUEST_LABELS)
        except Exception:
            project_labels = {"feature": "long-context-evaluation"}
        extra = dict(self._cfg.extra or {})
        labels = {**project_labels, **dict(extra.get("labels") or {})}
        labels.setdefault("component", "llm-judge")
        extra["labels"] = labels
        self._cfg.extra = extra

    @property
    def model_id(self) -> str:
        return self._model_id

    def evaluate(self, question: str, reference: str, response: str) -> dict[str, Any]:
        """Run the combined completeness + accuracy judge.

        Returns {"complete": bool|None, "accurate": bool|None, "rationale": str, "error": str, "usage": dict}.
        """
        from mlcr.providers.base import ChatRequest

        result: dict[str, Any] = {"complete": None, "accurate": None, "rationale": "", "error": "", "usage": {}}

        if not (response or "").strip():
            result["complete"] = False
            result["accurate"] = False
            result["rationale"] = "empty response"
            return result

        req = ChatRequest(
            system=_JUDGE_SYSTEM,
            user_text=_build_user_text(question, reference, response),
            images=[],
            model_cfg=self._cfg,
        )
        try:
            resp = self._provider.call(req)
            result["usage"] = resp.usage or {}
            parsed = _extract_json(resp.text)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            return result

        result["complete"] = _coerce_bool(parsed.get("complete"))
        result["accurate"] = _coerce_bool(parsed.get("accurate"))
        result["rationale"] = str(parsed.get("rationale", "")).strip()
        return result


# ---------------------------------------------------------------------------
# Jury: 3-model majority vote
# ---------------------------------------------------------------------------
def _majority_bool(bools: list[bool]) -> int | str:
    if len(bools) < 2:
        return ""
    return 1 if sum(bools) >= 2 else 0


class Jury:
    """Manages the 3-model jury and performs majority voting."""

    def __init__(self, model_config_paths: list[Path]) -> None:
        self._judges = [Judge(p) for p in model_config_paths]

    @property
    def model_ids(self) -> list[str]:
        return [j.model_id for j in self._judges]

    def evaluate(
        self, question: str, reference: str, response: str
    ) -> dict[str, Any]:
        """Run all judges and return the majority-vote verdict with full detail."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=len(self._judges)) as ex:
            futures = {
                ex.submit(j.evaluate, question, reference, response): j.model_id
                for j in self._judges
            }
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()

        votes: dict[str, bool | None] = {}
        rationales: dict[str, str] = {}
        completes: list[bool] = []
        accurates: list[bool] = []
        errors: list[str] = []

        for mid, r in results.items():
            if r["error"]:
                votes[mid] = None
                rationales[mid] = r["error"]
                errors.append(f"{mid}: {r['error']}")
            else:
                c = r["complete"]
                a = r["accurate"]
                if c is None or a is None:
                    votes[mid] = None
                    rationales[mid] = "unparseable verdict"
                    errors.append(f"{mid}: unparseable verdict")
                else:
                    votes[mid] = c and a
                    rationales[mid] = r["rationale"]
                    completes.append(c)
                    accurates.append(a)

        valid_votes = [v for v in votes.values() if v is not None]

        if len(valid_votes) >= 2:
            pass_count = sum(1 for v in valid_votes if v)
            majority_pass = pass_count >= 2
        else:
            majority_pass = None

        complete_majority = _majority_bool(completes)
        accurate_majority = _majority_bool(accurates)

        return {
            "majority_pass": majority_pass,
            "complete_majority": complete_majority,
            "accurate_majority": accurate_majority,
            "votes": votes,
            "rationales": rationales,
            "error": "; ".join(errors) if errors else "",
            "usage": {
                "input_tokens": {
                    mid: (r.get("usage") or {}).get("prompt_tokens", 0)
                    + (r.get("usage") or {}).get("cache_creation_input_tokens", 0)
                    + (r.get("usage") or {}).get("cache_read_input_tokens", 0)
                    for mid, r in results.items()
                },
                "output_tokens": {
                    mid: (r.get("usage") or {}).get("completion_tokens", 0)
                    for mid, r in results.items()
                },
                "thinking_tokens": {
                    mid: (r.get("usage") or {}).get("thinking_tokens", 0)
                    for mid, r in results.items()
                },
            },
        }


# ---------------------------------------------------------------------------
# Row-level scoring
# ---------------------------------------------------------------------------
def score_row(
    row: dict[str, str],
    jury: Jury | None,
) -> dict[str, Any]:
    """Score a single row through all gates. Returns judge_* columns."""
    out: dict[str, Any] = {c: "" for c in _JUDGE_COLS}

    reference = row.get("human_validated_answer", "")
    response = row.get("response", "")
    question = row.get("prompt", "")

    # Gate 0: Conciseness
    if not passes_conciseness_gate(response, reference):
        out["judge_concise"] = 0
        out["judge_correct"] = 0
        return out
    out["judge_concise"] = 1

    # Gate 1: LLM judge (completeness + accuracy)
    if jury is None:
        return out

    verdict = jury.evaluate(question, reference, response)

    out["judge_complete"] = verdict["complete_majority"]
    out["judge_accurate"] = verdict["accurate_majority"]
    out["judge_votes"] = json.dumps(
        {k: v for k, v in verdict["votes"].items()}, ensure_ascii=False
    )
    out["judge_rationales"] = json.dumps(
        verdict["rationales"], ensure_ascii=False
    )
    usage = verdict.get("usage") or {}
    out["judge_input_tokens"] = json.dumps(usage.get("input_tokens", {}), ensure_ascii=False)
    out["judge_output_tokens"] = json.dumps(usage.get("output_tokens", {}), ensure_ascii=False)
    out["judge_thinking_tokens"] = json.dumps(usage.get("thinking_tokens", {}), ensure_ascii=False)
    out["judge_error"] = verdict["error"]

    if verdict["majority_pass"] is True:
        out["judge_correct"] = 1
    elif verdict["majority_pass"] is False:
        out["judge_correct"] = 0
    else:
        out["judge_correct"] = ""

    return out


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------
def _row_key(row: dict[str, str]) -> str:
    parts = [
        row.get("case_uuid", ""),
        row.get("prompt", ""),
        row.get("model", ""),
        row.get("modality", ""),
        row.get("junk_context_ratio", ""),
        row.get("thinking", ""),
        row.get("response", ""),
    ]
    return hashlib.blake2b(
        "\x1f".join(parts).encode("utf-8"), digest_size=12
    ).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    cache: dict[str, dict[str, str]] = {}
    try:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("judge_correct", "").strip() and not row.get("judge_error", "").strip():
                    continue
                key = _row_key(row)
                cache[key] = {c: row.get(c, "") for c in _JUDGE_COLS}
    except Exception:
        return {}
    return cache


def _has_judge(scored: dict[str, str]) -> bool:
    return bool(str(scored.get("judge_correct", "")).strip()) or bool(
        str(scored.get("judge_error", "")).strip()
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def score_run(
    run_dir: Path,
    *,
    judge_model_configs: list[Path] | None = None,
    max_workers: int = 4,
    limit: int | None = None,
    checkpoint_every: int = 50,
) -> dict[str, Any]:
    """Score a run (conciseness gate + 3-model jury)."""
    run_dir = Path(run_dir).resolve()

    scoring_path = run_dir / "results_scoring.csv"
    responses_path = run_dir / "responses.csv"

    source_path = scoring_path if scoring_path.is_file() else responses_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Neither results_scoring.csv nor responses.csv found in {run_dir}")

    with source_path.open(newline="") as f:
        reader = csv.DictReader(f)
        in_fields = reader.fieldnames or []
        rows = list(reader)
    if limit is not None:
        rows = rows[:limit]

    out_path = scoring_path
    out_fields = list(in_fields) + [c for c in _HEURISTIC_COLS + _JUDGE_COLS if c not in in_fields]

    # Deterministic heuristic metrics (always computed, no network needed)
    for row in rows:
        reference = row.get("human_validated_answer", "")
        response = row.get("response", "")
        row.update(heuristic_scores(reference, response))

    cache = _load_cache(out_path)

    # Resolve model configs
    repo_root = Path(__file__).resolve().parents[2]
    if judge_model_configs is None:
        models_dir = repo_root / "configs" / "models"
        judge_model_configs = [models_dir / f"{mid}.yaml" for mid in _DEFAULT_JUDGE_MODELS]

    for p in judge_model_configs:
        if not p.is_file():
            raise FileNotFoundError(f"Judge model config not found: {p}")

    jury = Jury(judge_model_configs)
    print(f"Judge jury: {jury.model_ids}", file=sys.stderr)

    # Score rows
    stats = {"gate0_fail": 0, "llm_called": 0, "cached": 0, "errors": 0, "total": len(rows)}
    pending: list[int] = []

    for i, row in enumerate(rows):
        key = _row_key(row)
        cached = cache.get(key)
        if cached and _has_judge(cached):
            for c in _JUDGE_COLS:
                row[c] = cached.get(c, "")
            stats["cached"] += 1
        else:
            for c in _JUDGE_COLS:
                row.setdefault(c, "")
            pending.append(i)

    print(
        f"  judge: {stats['cached']}/{stats['total']} cached, {len(pending)} pending",
        file=sys.stderr, flush=True,
    )

    if pending:
        _write_csv(out_path, rows, out_fields)
        ckpt = max(1, checkpoint_every)
        done = 0

        def _do(idx: int) -> tuple[int, dict[str, Any]]:
            r = rows[idx]
            reference = r.get("human_validated_answer", "")
            response = r.get("response", "")

            # Gate 0 first (no LLM needed)
            if not passes_conciseness_gate(response, reference):
                return idx, {
                    "judge_concise": 0,
                    "judge_complete": "",
                    "judge_accurate": "",
                    "judge_correct": 0,
                    "judge_votes": "",
                    "judge_rationales": "",
                    "judge_error": "",
                }

            return idx, score_row(r, jury)

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futures = [ex.submit(_do, i) for i in pending]
            for fut in as_completed(futures):
                idx, scored = fut.result()
                rows[idx].update(scored)

                if scored.get("judge_concise") == 0:
                    stats["gate0_fail"] += 1
                else:
                    stats["llm_called"] += 1

                if scored.get("judge_error"):
                    stats["errors"] += 1

                done += 1
                pct = f"{100 * (stats['cached'] + done) / stats['total']:.0f}%"
                print(
                    f"\r  judge: {stats['cached'] + done}/{stats['total']} ({pct})  "
                    f"cached={stats['cached']}  gate0_fail={stats['gate0_fail']}  "
                    f"llm_calls={stats['llm_called']}  errors={stats['errors']}   ",
                    end="", file=sys.stderr, flush=True,
                )
                if done % ckpt == 0 or done == len(pending):
                    _write_csv(out_path, rows, out_fields)

        print(file=sys.stderr)

    # Final write
    _write_csv(out_path, rows, out_fields)

    return {
        "run_dir": str(run_dir),
        "rows_scored": len(rows),
        "results_scoring_csv": str(out_path),
        "jury_models": jury.model_ids,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(
        prog="mlcr-score",
        description="Score responses (conciseness gate + 3-model majority vote for completeness + accuracy).",
    )
    p.add_argument("run_dir", type=Path, help="Path to runs/<experiment_uuid>")
    p.add_argument(
        "--judge-models",
        nargs="+",
        default=None,
        help=f"Model config IDs for the jury (default: {', '.join(_DEFAULT_JUDGE_MODELS)})",
    )
    p.add_argument("--max-workers", type=int, default=4, help="Parallel jury evaluations (default: 4)")
    p.add_argument("--limit", type=int, default=None, help="Only score the first N rows")
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Write results_scoring.csv every N scored rows (default: 50)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    args = build_arg_parser().parse_args(argv)

    judge_model_configs = None
    if args.judge_models:
        repo_root = Path(__file__).resolve().parents[2]
        models_dir = repo_root / "configs" / "models"
        judge_model_configs = [models_dir / f"{mid}.yaml" for mid in args.judge_models]

    try:
        result = score_run(
            args.run_dir,
            judge_model_configs=judge_model_configs,
            max_workers=args.max_workers,
            limit=args.limit,
            checkpoint_every=args.checkpoint_every,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
