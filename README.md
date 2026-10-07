# VRIQ evaluation code

Code for **VRIQ: Benchmarking and Diagnosing the Visual-Reasoning IQ of Vision–Language Models**.

This folder is the runnable source. It does not contain the puzzle images, model outputs, or API keys. Set keys in the environment. Do not paste them into these files.

The released data is on the Hugging Face Hub:

- [VRIQ](https://huggingface.co/datasets/tina-khezresmaeilzadeh/VRIQ) (1,391 items: abstract 788, natural 603)
- [DiagVRIQ](https://huggingface.co/datasets/tina-khezresmaeilzadeh/DiagVRIQ) (209 natural items with verified descriptions)

## Environment

Python 3.12.

```bash
cd /path/to/vriq
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API keys, only for the providers you call:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

Open-weight runs also need PyTorch and `transformers==4.51.3`. Install a CUDA build of PyTorch that matches the GPUs. The paper used two NVIDIA RTX A6000 GPUs (48 GB each). vLLM is optional and is only needed for `--gen_engine vllm`.

## Data

With no `--dataset_root`, the scripts download the released datasets from the Hugging Face Hub and cache the images locally:

- End-to-end runs load [tina-khezresmaeilzadeh/VRIQ](https://huggingface.co/datasets/tina-khezresmaeilzadeh/VRIQ). Pass `--split abstract` or `--split natural`.
- DiagVRIQ runs load [tina-khezresmaeilzadeh/DiagVRIQ](https://huggingface.co/datasets/tina-khezresmaeilzadeh/DiagVRIQ), including the verified descriptions.

Rows with a blank question or answer are skipped. Images are resized to at most 250,000 pixels, with each side at least 28 pixels.

`--dataset_root` still accepts a local category-folder tree (CSV columns `PID`, `Category`, `Question`, `Ground truth`) if you already have one.

## Paper sections

Reported averages in the paper are unweighted means of the five category accuracies. Each item is evaluated once.

### End-to-end accuracy (Section 6.1)

Run every command from this folder so Python can see the packages.

`e2e.evaluate` is the shared evaluator. `--reasoning_effort` defaults to `high`, so pass the paper setting explicitly. For GPT-5.1 and GPT-5.2 pass `--reasoning_effort none`. For o3 and GPT-5-mini pass `--reasoning_effort medium`.

```bash
python -m e2e.evaluate \
  --split abstract \
  --model_name_path gpt-5.6-sol \
  --gen_engine openai \
  --reasoning_effort high \
  --max_new_tokens 16384 \
  --outputs_dir ./results \
  --tag abstract_sol_high
```

Open-weight models, including Qwen3-VL-32B-Thinking, use greedy decoding at temperature 0.

Reasoning settings used in the paper:

| Model | Setting |
|---|---|
| GPT-5.6 Sol | `high` on the main tables; `medium` on DiagVRIQ and as the second main-table row |
| Claude Opus 5 | adaptive thinking, effort `high` |
| Gemini 3.1 Pro Preview | thinking level `high` |
| GPT-5.1, GPT-5.2 | reasoning effort omitted, which selects `none` |
| o3, GPT-5-mini | reasoning effort omitted, which selects `medium` |
| Open-weight models, including Qwen3-VL-32B-Thinking | greedy, temperature 0 |

For o3 with tools, omit `--no_o3_tools`. For the matched no-tool condition, pass `--no_o3_tools`.

Claude and Gemini use their own scripts. Each command scores one split. Repeat it with `--split natural` for the other domain. Opus 5 uses adaptive thinking at effort high. Gemini 3.1 Pro Preview uses thinking level high.

```bash
python -m e2e.claude --split abstract --model claude-sonnet-4-6
python -m e2e.claude --split abstract --model claude-opus-5 --max_tokens 8192
python -m e2e.gemini --split abstract --model gemini-2.5-pro
python -m e2e.gemini --split abstract --model gemini-3.1-pro-preview
```

### DiagVRIQ describe-then-solve (Section 6.2)

The model writes a description, then an answer, in one response. GPT-5.6 Sol on DiagVRIQ uses medium reasoning effort.

Open-weight models use greedy decoding at temperature 0.

```bash
python -m diag.claude --phase two_stage
python -m diag.openai
python -m diag.gemini
```

Check each script's `--help` for the model flag. Outputs go under `./results`.

The perception judge is GPT-5.2. It compares the model description with the verified reference and returns a binary perception-success label. Human agreement and the Claude Opus 4.8 cross-check are in `diag/judge.py`.

```bash
python -m diag.judge --help
```

### Ground-truth description augmentation (Section 6.3)

The verified description is given to the model together with the image. The paper reports LLaVA-1.6-Mistral-7B, Qwen2.5-VL-7B, and GPT-4o. Claude Opus 5 and GPT-5.6 Sol are not in that table.

```bash
python -m augment.llava
python -m augment.qwen
python -m augment.gpt4o
```

### Tool-augmented inference (Section 6.4)

o3 is run on all 1,391 items with `code_interpreter` enabled, and again with tools off. Both conditions use the same prompt. This is one bundled setting: image edits, extra computation, and code execution together.

```bash
python -m e2e.evaluate \
  --split natural \
  --model_name_path o3 \
  --gen_engine openai \
  --reasoning_effort medium \
  --outputs_dir ./results \
  --tag o3_tools
```

Add `--no_o3_tools` for the no-tool condition.

## What is not in this folder

Puzzle images, result JSON files, virtual environments, and API keys stay out of the repository. The original working copy was not modified.
