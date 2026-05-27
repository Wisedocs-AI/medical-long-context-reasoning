# Mock Data Generation

Tools for generating realistic fake-filled form images used as filler material in long-context evaluation experiments.

## Why This Exists

The `mlcr` evaluation framework tests how well LLMs process documents when surrounded by noise. To create convincing noise, we need filled form images that look realistic but contain entirely fake data. These tools populate the `filler_files/` pool that the evaluation harness samples from when constructing long-context prompts.

The experiment configs specify a `filler_subdir` (e.g. `image_gen_1variant`), and the framework interleaves those filler images with real test documents at a configurable ratio — testing whether models can still answer correctly despite the added context.

## Tool

### `ai_form_filler.py` — AI-Powered Generation

Uses Gemini 3.1 Flash Image to automatically generate filled form images from blank templates, with a human-in-the-loop approval step.

**When to use:** Bulk generation where manual annotation would be too slow.

**3-Phase Workflow:**

1. **Triage** — Review each blank form page. Press `Y` to copy as-is (pages that don't need filling) or `N` to queue for AI generation.
2. **Concurrent Generation** — Queued pages are sent to Gemini (up to 16 concurrent threads). The model receives the blank form plus a detailed prompt with faker profile data and returns a filled image.
3. **Batch Approval** — Review each generated result. Approve, reject, or regenerate.

Progress is checkpointed to `.progress.json` so the process can be interrupted and resumed.

## Output

Images are produced at:

```
filler_files/<output-subdir>/images/{prefix}-v{1..N}-{page}.jpg
```

Where:
- `prefix` — the form identifier (derived from the blank source filename before the first `_`)
- `v1/v2/v3` — variant number, each with a different fake identity
- `page` — page number within multi-page forms

## Setup

### Environment Variables

Create a `.env` file in the repo root:

```bash
GOOGLE_CLOUD_PROJECT=<your-gcp-project>
GOOGLE_APPLICATION_CREDENTIALS=<path-to-service-account-key.json>
GOOGLE_CLOUD_LOCATION=us-central1  # optional, defaults to us-central1
```

### Dependencies

Install with the `google` extra:

```bash
pip install -e '.[google]'
```

| Package | Purpose |
|---------|---------|
| `tkinter` | GUI (standard library) |
| `Pillow` | Image manipulation |
| `Faker` | Fake identity generation |
| `python-dotenv` | Loading `.env` credentials |
| `google-genai` | Gemini API client |
| `google-auth` | Service account auth |

### Running

```bash
# Default: 3 variants per page, output to filler_files/image_gen/
python mock_data_generation/ai_form_filler.py

# Custom output subdirectory and variant count
python mock_data_generation/ai_form_filler.py --output-subdir image_gen_1variant --variants 1

# Reproducible generation with a fixed seed
python mock_data_generation/ai_form_filler.py --seed 42
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-subdir` | `image_gen` | Subdirectory under `filler_files/` for output |
| `--variants` | `3` | Number of filled variants to generate per page |
| `--seed` | random | Seed for Faker data generation (for reproducibility) |

### Keyboard Shortcuts (in-app)

| Key | Phase | Action |
|-----|-------|--------|
| `Y` | Triage | Copy original as all variants |
| `N` | Triage | Queue page for AI generation |
| `G` | Triage | Generate a preview immediately |
| `A` / `Enter` | Approval | Approve and save |
| `R` | Approval | Reject (skip) |

## Input Structure

Blank form images are expected at:

```
filler_files/empty/images/
```

Images should follow the naming convention `{prefix}_{page}.jpg` (e.g. `formA_001.jpg`, `formA_002.jpg`). The prefix groups pages into a single logical form.
