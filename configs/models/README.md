# Model Configs

Each YAML file in this directory defines one model that can be referenced from experiment configs.

## File Naming

The filename (without extension) must match the `id` field inside the file:

```
configs/models/<id>.yaml
```

For example, `gemini-2.5-flash.yaml` must contain `id: gemini-2.5-flash`.

## Schema

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | string | yes | — | Unique identifier, referenced from experiment `models` list |
| `provider` | string | yes | — | Which API client to use (see providers below) |
| `model` | string | yes | — | Model name sent to the provider API |
| `max_output_tokens` | int | no | 4096 | Maximum tokens in the model response |
| `temperature` | float | no | 0.0 | Sampling temperature |
| `extra` | dict | no | `{}` | Provider-specific parameters passed through to the API call |

## Providers

| Provider ID | API | Required Environment Variables |
|-------------|-----|-------------------------------|
| `google` | Vertex AI (Gemini) | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`, optionally `GOOGLE_CLOUD_LOCATION` |
| `anthropic_gcp` | Anthropic via GCP | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`, `ANTHROPIC_GCP_REGION` |
| `azure_openai` | Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
Install the appropriate optional dependency for your provider:

```bash
pip install -e '.[google]'       # google
pip install -e '.[anthropic]'    # anthropic / anthropic_gcp
pip install -e '.[azure]'        # azure_openai
pip install -e '.[all]'          # all providers
```

## Thinking Levels

Thinking budget is typically controlled at the **experiment** level (not here), allowing the same model config to be swept across multiple thinking levels. See `configs/experiments/README.md` for details.

## Examples

### Google (Gemini)

```yaml
id: gemini-2.5-flash
provider: google
model: gemini-2.5-flash
max_output_tokens: 16384
temperature: 1.0
```

### Anthropic via GCP

```yaml
id: claude-opus-4-8-gcp
provider: anthropic_gcp
model: claude-opus-4-8@default
max_output_tokens: 16384
temperature: 1.0
```

### Azure OpenAI

```yaml
id: gpt-5.5
provider: azure_openai
model: gpt-5.5
max_output_tokens: 16384
temperature: 1.0
```

## Adding a New Model

1. Create `configs/models/<model-id>.yaml`
2. Set `id` to match the filename
3. Choose the correct `provider`
4. Set `model` to the exact model name the provider API expects
5. Reference the `id` in your experiment config's `models` list
