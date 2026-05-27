from __future__ import annotations

from mlcr.config import ModelConfig

# Maps the abstract experiment-level thinking level to each provider's native
# representation. The model config carries only the base model; the per-row
# thinking level is injected here so a single model definition can be swept
# across multiple thinking budgets.

# Gemini 2.x models use a token-budget API instead of thinking_level.
_THINKING_BUDGET_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0")

_THINKING_BUDGET_MAP = {
    "minimal": 0,
    "low": 4096,
    "medium": 16384,
    "high": 32768,
}


def _uses_thinking_budget(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _THINKING_BUDGET_MODELS)


def apply_thinking(mcfg: ModelConfig, level: str) -> ModelConfig:
    """Return a copy of `mcfg` with provider-specific thinking settings for `level`.

    `level` is one of mlcr.config.THINKING_LEVELS. `"none"` disables thinking.
    Providers that don't support thinking simply ignore it.

    Note: not every level is valid for every provider (e.g. Gemini accepts
    MINIMAL/LOW/MEDIUM/HIGH, while Anthropic adaptive effort expects
    low/medium/high). Choose `thinking` levels compatible with the models in the
    experiment.
    """
    m = mcfg.model_copy(deep=True)

    if level == "none":
        m.thinking = None
        return m

    if mcfg.provider == "google":
        if _uses_thinking_budget(mcfg.model):
            budget = _THINKING_BUDGET_MAP.get(level, 0)
            if budget == 0:
                m.thinking = {"thinking_budget": 0}
            else:
                m.thinking = {"thinking_budget": budget, "include_thoughts": True}
        else:
            m.thinking = {"thinking_level": level.upper(), "include_thoughts": True}
    elif mcfg.provider == "anthropic_gcp":
        m.thinking = {"type": "adaptive"}
        extra = dict(m.extra or {})
        output_config = dict(extra.get("output_config") or {})
        output_config["effort"] = level
        extra["output_config"] = output_config
        m.extra = extra
    elif mcfg.provider == "azure_openai":
        m.thinking = {"effort": level}
    # Other providers (e.g. fake) ignore thinking entirely.

    return m
