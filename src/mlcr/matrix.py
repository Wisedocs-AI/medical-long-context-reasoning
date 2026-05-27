from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from mlcr.cases import load_case, load_prompt_difficulties
from mlcr.config import ExperimentConfig


@dataclass(frozen=True)
class Row:
    row_id: str
    case_uuid: str
    prompt_path: Path
    model_config_id: str
    modality: str
    ablation_ratio: float
    thinking: str
    # Evaluation-only annotation; recorded in outputs but never sent to the model.
    difficulty: str = ""
    # Repetition index (0-based). Only meaningful when repetitions > 1.
    repetition: int = 0


def format_ratio(ratio: float) -> str:
    """Compact, deterministic string form of a ratio (e.g. 0 -> "0", 1 -> "1",
    0.5 -> "0.5"). Used for row_id keys and human-readable output."""
    return f"{ratio:g}"


def _row_id(case_uuid: str, prompt_name: str, model_id: str, modality: str,
            ratio: float, thinking: str, filler_subdir: str, repetition: int = 0) -> str:
    key = f"{case_uuid}|{prompt_name}|{model_id}|{modality}|{format_ratio(ratio)}|{thinking}|filler_subdir={filler_subdir}"
    if repetition > 0:
        key += f"|rep={repetition}"
    return hashlib.blake2b(key.encode(), digest_size=8).hexdigest()


def build_matrix(cfg: ExperimentConfig) -> list[Row]:
    assert cfg.repo_root is not None
    rows: list[Row] = []
    for cuid in cfg.cases:
        case = load_case(cfg.repo_root, cuid)
        if not case.prompts:
            raise ValueError(f"case {cuid} has no prompts")
        # Difficulty labels are an evaluation annotation only; they are recorded
        # in outputs but deliberately left out of the row_id and the model prompt.
        difficulties = load_prompt_difficulties(case.root)
        for prompt in case.prompts:
            difficulty = difficulties.get(prompt.stem.lower(), "")
            if cfg.prompt_categories and difficulty not in cfg.prompt_categories:
                continue
            for mid in cfg.models:
                for modality in cfg.modalities:
                    for ratio in cfg.junk_context_ratio:
                        for thinking in cfg.thinking.levels_for(mid):
                            for rep in range(cfg.repetitions):
                                rid = _row_id(cuid, prompt.name, mid, modality, ratio, thinking, cfg.filler_subdir, rep)
                                rows.append(Row(
                                    row_id=rid,
                                    case_uuid=cuid,
                                    prompt_path=prompt,
                                    model_config_id=mid,
                                    modality=modality,
                                    ablation_ratio=ratio,
                                    thinking=thinking,
                                    difficulty=difficulty,
                                    repetition=rep,
                                ))
    return rows
