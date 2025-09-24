# Articula LLM

## Purpose

Train a compact model that explains its reasoning. This project fine-tunes `microsoft/phi-4-mini-reasoning` with chain-of-thought supervision so the assistant states intermediate steps before final answers. The repository includes data prep, LoRA training, and a small demo for inference.

## Data choice and hallucination reduction

- squad: forces evidence-grounded span selection, building a habit of quoting or pointing back to context rather than inventing facts.
- squad_v2: includes unanswerable cases, teaching the model to say "cannot answer" when the context is insufficient, which reduces confident-but-wrong outputs.
- trivia_qa: adds longer, noisier passages and multi-hop questions, encouraging step-by-step justification over shortcut guessing.

Together these sets push the model toward evidence-first reasoning, explicit abstention when needed, and multi-step discipline. They complement the base model’s synthetic reasoning traces by adding natural language noise, real document structure, and refusal signals that curb hallucinations during chain-of-thought.

---

Articula LLM fine-tunes `microsoft/phi-4-mini-reasoning` so the model spells out the thinking that leads to an answer. The repository covers data preparation, LoRA training, and evaluation for this chain-of-thought goal.

---

## Model Choice

We start from phi-4-mini-reasoning because it offers a practical balance between capability and resource use:

- pretrained traces that already handle math, code, and general reasoning
- a parameter budget that fits on a single modern GPU
- an instruction-tuned tokenizer and chat template for quick experimentation
- built-in safety filters that reduce obvious failure modes before fine-tuning
 - edge-friendly footprint: with 4/8-bit quantization it can run on capable laptops and some edge devices, enabling offline use and stronger data insulation

---

## Chain-of-Thought Objective

The LoRA adapters encourage the model to narrate intermediate steps before giving a final answer. In practice this:

- promotes step-by-step explanations
- discourages shortcut guessing
- makes tool usage and verification easier
- improves how often confidence matches the evidence in the prompt

---

## Data

Training mixes reasoning-oriented question answering sets:

- `squad` for grounded span extraction and reference quoting
- `squad_v2` for explicit "cannot answer" signals
- `trivia_qa` for longer multi-hop lookups

The combination covers factual grounding, calibrated refusals, and multi-step reasoning without exploding dataset size.

`phi-4-mini-reasoning` arrives with synthetic chain-of-thought traces drawn from curated math, coding, and symbolic reasoning corpora. Those traces teach disciplined internal reasoning, but they lean on semi-structured problems and synthetic passages. By adding human-written reading comprehension sets, the adapters are exposed to noisier context, natural question phrasing, and explicit evidence requirements. The result is a model that keeps the disciplined step-by-step habits of the base training while learning to cite context, admit missing information, and bridge longer stretches of prose during chain-of-thought responses.

---

## Training Workflow

---

## Training Quickstart

Checkpoints are saved under `checkpoints/<run-name>` by default. Below are ready-to-copy commands.

Pre-tokenized data (recommended):

```bash
python scripts/train.py \
  --train_dir data/train_tokenized \
  --eval_dir data/val_tokenized \
  --output_dir checkpoints/articula-run \
  --overwrite_output_dir True \
  --do_train True --do_eval True \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
  --learning_rate 2e-4 --weight_decay 0.1 \
  --save_strategy steps --save_steps 1000 --save_total_limit 5 \
  --use_lora True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --gradient_checkpointing True --fp16 True
```

Raw JSONL data:

```bash
python scripts/train.py \
  --train_file data/train.jsonl \
  --val_file data/val.jsonl \
  --output_dir checkpoints/articula-run \
  --overwrite_output_dir True \
  --do_train True --do_eval True \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
  --learning_rate 2e-4 --weight_decay 0.1 \
  --save_strategy steps --save_steps 1000 --save_total_limit 5 \
  --use_lora True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --gradient_checkpointing True --fp16 True
```

Long-run example (aiming for ~2 days on a single GPU; adjust for your hardware):

```bash
python scripts/train.py \
  --train_dir data/train_tokenized \
  --val_file data/val.jsonl \
  --output_dir checkpoints/articula-2day \
  --overwrite_output_dir True \
  --do_train True --do_eval True \
  --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 --weight_decay 0.1 \
  --max_steps 200000 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --save_strategy steps --save_steps 1000 --save_total_limit 5 \
  --use_lora True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --use_4bit True --gradient_checkpointing True --fp16 True
```

Notes:

- If `--output_dir` already has checkpoints, training auto-resumes unless `--overwrite_output_dir True` is set.
- Use `--use_8bit` or `--use_4bit` for memory savings on supported GPUs.
- Increase `--gradient_accumulation_steps` to emulate larger batch sizes.
 - With `--do_eval True`, an evaluation run executes after training finishes (no mid-run eval flags required).

1. Build instruction–response pairs with reasoning traces via `scripts/prepare_data.py`.
2. Pretokenize the datasets using `scripts/pretokenize_data.py` to speed up training and apply assistant-only loss masking.
3. Launch LoRA fine-tuning with `scripts/train.py`, pointing to an experiment directory under `checkpoints/`.
4. Track validation loss plus reasoning-specific metrics such as trace accuracy and answer accuracy.

Each run writes adapter weights, tokenizer files, and trainer states to `checkpoints/<run-name>`.

---

## Edge and Offline Deployment

When quantized (e.g., 4-bit or 8-bit), the model can run locally on a single modern GPU or certain edge devices. This supports:

- offline inference when connectivity is limited or unavailable
- privacy-sensitive scenarios where prompts and outputs must remain on-device
- reduced latency by avoiding round trips to remote services

Exact hardware requirements depend on quantization, sequence length, and batch size. Start with 4/8-bit builds for the best trade-off between memory and quality.

---

## Results Snapshot

On the held-out validation split we observed:

- lower language-modeling loss compared to the base model
- higher exact-match accuracy on reasoning tasks with multi-step answers
- more complete rationales when compared to gold traces
- better human preference scores for clarity

---

## Demo

Assuming training has finished and the adapter is merged or loaded at runtime, you can query the model with `scripts/generate.py`:

```bash
python scripts/generate.py --ckpt checkpoints/articula-lora --prompt "A shop sells packs of 6 pencils. You need 45 pencils. How many packs should you buy?"
```

Sample interaction:

```text
User: A shop sells packs of 6 pencils. You need 45 pencils. How many packs should you buy?

Assistant:
Let's calculate. One pack holds 6 pencils. 45 ÷ 6 = 7 remainder 3, so seven packs give 42 pencils. We still need three pencils, which means one more pack. Final answer: 8 packs.
```

---

## Roadmap

- Expand evaluation with adversarial reasoning probes.
- Experiment with retrieval-augmented generation for knowledge-heavy prompts.
- Release quantized builds for edge or on-device deployment.

Contributions are welcome via issues or pull requests.
