from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

Modality = Literal["text", "image", "both"]

# Abstract thinking levels swept from the experiment config. Each provider maps
# these to its own native representation (see mlcr.thinking.apply_thinking).
THINKING_LEVELS = ("none", "minimal", "low", "medium", "high")


class ModelConfig(BaseModel):
    id: str
    provider: str
    model: str
    max_output_tokens: int = 4096
    temperature: float = 0.0
    thinking: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)


class ConcurrencyConfig(BaseModel):
    max_workers: int = 4
    per_provider: dict[str, int] = Field(default_factory=dict)


class RetryConfig(BaseModel):
    max_attempts: int = 5
    initial_backoff_s: float = 2.0
    max_backoff_s: float = 60.0


class ThinkingConfig(BaseModel):
    """Per-model thinking levels. Supports a flat list (applied to all models)
    or a dict with 'default' and optional per-model 'overrides'."""
    default: list[str] = Field(default_factory=lambda: ["none"])
    overrides: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("default")
    @classmethod
    def _check_default(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("thinking.default must list at least one level")
        for level in v:
            if level not in THINKING_LEVELS:
                raise ValueError(
                    f"unknown thinking level {level!r}; allowed: {', '.join(THINKING_LEVELS)}"
                )
        return v

    @field_validator("overrides")
    @classmethod
    def _check_overrides(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        for model_id, levels in v.items():
            if not levels:
                raise ValueError(f"thinking.overrides[{model_id!r}] must list at least one level")
            for level in levels:
                if level not in THINKING_LEVELS:
                    raise ValueError(
                        f"unknown thinking level {level!r} in overrides[{model_id!r}]; "
                        f"allowed: {', '.join(THINKING_LEVELS)}"
                    )
        return v

    def levels_for(self, model_id: str) -> list[str]:
        """Return thinking levels for a specific model."""
        return self.overrides.get(model_id, self.default)


def _parse_thinking(raw: Any) -> ThinkingConfig:
    """Parse the 'thinking' field from YAML which can be either a flat list
    (backwards-compatible) or a dict with 'default' and 'overrides'."""
    if raw is None:
        return ThinkingConfig()
    if isinstance(raw, list):
        return ThinkingConfig(default=raw)
    if isinstance(raw, dict):
        return ThinkingConfig.model_validate(raw)
    raise ValueError(f"thinking must be a list or dict, got {type(raw).__name__}")


class ExperimentConfig(BaseModel):
    name: str
    models: list[str]
    cases: list[str]
    modalities: list[Modality]
    # Ratio of filler pages to real case pages. 0 = no filler, 1 = as many filler
    # pages as real pages, 2 = twice as many filler pages, etc. One file (text or
    # image) equals one page; text and images are treated identically.
    junk_context_ratio: list[float]
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)
    prompt_categories: list[str] | None = None
    allowed_filler_forms: list[str] = Field(default_factory=list)
    filler_subdir: str = "empty"
    # Number of independent repetitions per matrix cell. Each repetition gets a
    # unique row_id so the runner calls the model N times for the same input,
    # useful for measuring response variance. Default 1 (no repetition).
    repetitions: int = 1
    seed: int = 0
    output_dir: str = "runs"
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # resolved later (not from YAML)
    config_dir: Path | None = None
    repo_root: Path | None = None
    model_configs: dict[str, ModelConfig] = Field(default_factory=dict)

    @field_validator("repetitions")
    @classmethod
    def _check_repetitions(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"repetitions must be >= 1: got {v}")
        return v

    @field_validator("junk_context_ratio")
    @classmethod
    def _check_junk_context_ratio(cls, v: list[float]) -> list[float]:
        for r in v:
            if r < 0:
                raise ValueError(f"junk_context_ratio must be >= 0: got {r}")
        return v

    @field_validator("thinking", mode="before")
    @classmethod
    def _parse_thinking_field(cls, v: Any) -> ThinkingConfig:
        return _parse_thinking(v)

    @field_validator("filler_subdir")
    @classmethod
    def _check_filler_subdir(cls, v: str) -> str:
        allowed = ("empty", "image_gen", "image_gen_1variant", "image_gen_3variants")
        if v not in allowed:
            raise ValueError(f"unknown filler_subdir {v!r}; allowed: {', '.join(allowed)}")
        return v

    @field_validator("modalities")
    @classmethod
    def _check_modalities(cls, v: list[str]) -> list[str]:
        allowed = {"text", "image", "both"}
        for m in v:
            if m not in allowed:
                raise ValueError(f"unknown modality: {m}")
        return v

    @model_validator(mode="after")
    def _seed_required_if_ablation(self) -> "ExperimentConfig":
        if any(r > 0 for r in self.junk_context_ratio) and self.seed is None:
            raise ValueError("seed is required when any junk_context_ratio > 0")
        return self


def load_experiment(path: Path, repo_root: Path | None = None) -> ExperimentConfig:
    """Load experiment YAML and all referenced model configs.

    Raises ValueError on any validation failure (missing model file,
    missing case directory, bad modality/ablation, etc).
    """
    path = Path(path).resolve()
    repo_root = (repo_root or path.parent.parent.parent).resolve()
    data = yaml.safe_load(path.read_text())
    cfg = ExperimentConfig.model_validate(data)
    cfg.repo_root = repo_root
    cfg.config_dir = path.parent

    models_dir = repo_root / "configs" / "models"
    for mid in cfg.models:
        mp = _find_model_file(models_dir, mid)
        if mp is None:
            raise ValueError(f"model config not found for id '{mid}' in {models_dir}")
        mc = ModelConfig.load(mp)
        if mc.id != mid:
            raise ValueError(f"model file {mp} has id={mc.id!r}, expected {mid!r}")
        cfg.model_configs[mid] = mc

    cases_dir = repo_root / "cases"
    needs_text = any(m in ("text", "both") for m in cfg.modalities)
    needs_image = any(m in ("image", "both") for m in cfg.modalities)
    for cuid in cfg.cases:
        cdir = cases_dir / cuid
        if not cdir.is_dir():
            raise ValueError(f"case directory missing: {cdir}")
        if not (cdir / "prompts").is_dir():
            raise ValueError(f"case {cuid}: prompts/ missing")
        if needs_text and not (cdir / "summaries").is_dir():
            raise ValueError(f"case {cuid}: summaries/ missing (required for text/both modalities)")
        if needs_image and not (cdir / "images").is_dir():
            raise ValueError(f"case {cuid}: images/ missing (required for image/both modalities)")

    if cfg.allowed_filler_forms:
        from mlcr.cases import list_filler_forms

        available = list_filler_forms(repo_root, filler_subdir=cfg.filler_subdir)
        unknown = [f for f in cfg.allowed_filler_forms if f not in available]
        if unknown:
            raise ValueError(
                f"allowed_filler_forms not found in filler_files: {unknown}. "
                f"available forms: {sorted(available)}"
            )

    return cfg


def _find_model_file(models_dir: Path, mid: str) -> Path | None:
    for ext in (".yaml", ".yml", ".json"):
        p = models_dir / f"{mid}{ext}"
        if p.is_file():
            return p
    return None
