# Experiment Configs

Each YAML file in this directory defines one experiment — a complete specification of which models, cases, and parameters to evaluate.

## How It Works

The `mlcr` runner reads an experiment config and builds a **test matrix** by taking the Cartesian product of:

```
cases × prompts × models × modalities × junk_context_ratio × thinking_levels × repetitions
```

Each cell in this matrix becomes one API call to a model. The runner executes all cells concurrently (up to `concurrency.max_workers`), retries on transient failures, and writes structured results to `output_dir`.

## Config Fields

### Required

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable experiment name |
| `models` | list[string] | Model IDs to evaluate (must have a matching file in `configs/models/`) |
| `cases` | list[string] | Case UUIDs to test against (must exist as directories under `cases/`) |
| `modalities` | list[`text` \| `image` \| `both`] | How case documents are presented to the model |
| `junk_context_ratio` | list[float] | Ratio of filler pages to real pages. `0` = no filler, `1` = equal filler, `2` = twice as many filler pages, etc. |

### Thinking

Controls how much "thinking" budget each model gets. Supports a flat list (same for all models) or per-model overrides:

```yaml
# Simple: same levels for all models
thinking: [none, minimal]

# Advanced: per-model overrides
thinking:
  default: [minimal]
  overrides:
    claude-opus-4-8-gcp: [low]
    gpt-5.5: [low]
```

Available levels: `none`, `minimal`, `low`, `medium`, `high`. Each provider maps these to its native representation.

### Filtering

| Field | Type | Description |
|-------|------|-------------|
| `prompt_categories` | list[string] \| null | Only run prompts matching these difficulty tiers. `null` = run all. |
| `allowed_filler_forms` | list[string] | Restrict which filler forms are sampled (by prefix). Empty = allow all. |

### Filler / Noise

| Field | Type | Description |
|-------|------|-------------|
| `filler_subdir` | string | Which filler pool to sample from under `filler_files/`. One of: `empty`, `image_gen`, `image_gen_1variant`, `image_gen_3variants` |
| `seed` | int | Random seed for reproducible filler sampling and shuffling |

### Execution

| Field | Type | Description |
|-------|------|-------------|
| `repetitions` | int | Number of independent runs per matrix cell (default: 1). Useful for measuring variance. |
| `output_dir` | string | Where results are written (default: `runs`) |
| `concurrency.max_workers` | int | Max parallel API calls (default: 4) |
| `retry.max_attempts` | int | Retries on transient API failures (default: 5) |
| `retry.initial_backoff_s` | float | Initial backoff delay (default: 2.0) |
| `retry.max_backoff_s` | float | Max backoff cap (default: 60.0) |
