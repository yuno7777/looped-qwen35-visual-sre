# OpenCode DeepSeek CoT Dataset Builder

A production-grade, zero-dependency Python pipeline that wraps `opencode run` (non-interactive mode) to collect Chain-of-Thought (CoT) reasoning traces and schema-validated JSON tool calls from `opencode/deepseek-v4.1-flash-free`, formatted directly for **Qwen3.5 multimodal SFT**.

## Project Structure

```text
cot_dataset_builder/
├── config.py             # Model ID, rate-limit/backoff settings, system prompt, JSON tool schemas
├── prompts.jsonl         # Starter input tasks (id, prompt, optional image_path, expected_tool)
├── generate_dataset.py   # Main orchestrator: preflight, subprocess wrapper, checkpointing, SFT formatter
├── validator.py          # Rejection-sampling validator for CoT quality & JSON schema compliance
└── README.md
```

### Generated Output Files
- `dataset_raw.jsonl` — Every completed model trace with parsed `cot`, `answer`, token usage, and validation status.
- `dataset_clean.jsonl` — Only traces that pass rejection sampling, formatted for Qwen3.5 multimodal SFT.
- `failed_traces.jsonl` — Raw stdout/stderr and failure reasons for execution errors, missing images, or rejected traces.

## Quick Start

### 1. Preflight Check (Verify Model ID)
Run the preflight check first to verify `opencode/deepseek-v4.1-flash-free` is active in your local OpenCode setup:

```bash
cd cot_dataset_builder
python generate_dataset.py --preflight-only
```

If your model ID differs, the preflight check lists all available DeepSeek models from `opencode models`. Update `MODEL_ID` in `config.py` or pass `--model <provider/model>`.

### 2. Run the Pipeline
Run a 1-item smoke test first:

```bash
python generate_dataset.py --limit 1
```

Run the full batch:

```bash
python generate_dataset.py
```

### 3. Resume After Interruption or Rate Limits
Checkpointing is automatic. Re-running `python generate_dataset.py` reads completed `id`s from `dataset_raw.jsonl` and `dataset_clean.jsonl`, skips them, and resumes cleanly from the remaining prompts.

## CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | `opencode/deepseek-v4.1-flash-free` | OpenCode `provider/model` string |
| `--input` | `prompts.jsonl` | Path to input prompts JSONL |
| `--sleep` | `10` | Seconds to wait between requests |
| `--max-retries` | `5` | Retries with exponential backoff on errors/throttling |
| `--limit` | *(none)* | Process at most `N` pending prompts |
| `--preflight-only` | `false` | Run `opencode models` check and exit |
| `--skip-preflight` | `false` | Skip the `opencode models` preflight check |

## Customizing for Your Niche

1. **Tool Schemas (`config.py`)**: Edit `TOOL_SCHEMAS` with your own tool names and JSON Schema `parameters` definitions.
2. **Prompts (`prompts.jsonl`)**: Add one JSON object per line:
   ```json
   {"id": "sample_001", "prompt": "Your task...", "image_path": "samples/screen.png", "expected_tool": "your_tool"}
   ```
   Set `"image_path": null` for text-only tasks. When `image_path` is provided, `generate_dataset.py` automatically passes `-f <IMAGE_PATH>` to `opencode run` and formats the SFT user turn with `<image>\n{prompt}` and `"images": ["..."]`.

## Qwen3.5 SFT Output Format (`dataset_clean.jsonl`)

```json
{
  "id": "task_004",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>\nExtract the vendor name..."},
    {"role": "assistant", "content": "<think>\n...\n</think>\n{\"tool\": \"extract_invoice_fields\", \"arguments\": {...}}"}
  ],
  "images": ["samples/invoice_001.png"]
}
```
