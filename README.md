# FunRL-Qwen3.5

GRPO-based reinforcement learning fine-tuning of **Qwen3.5-4B** for tool-calling / function-calling reliability, inspired by [FunRL](https://github.com/BingguangHao/RLFC) (*Exploring Superior Function Calls via Reinforcement Learning*, [arXiv:2508.05118](https://arxiv.org/html/2508.05118v1)).

Trained entirely on a **free Kaggle T4 GPU** using 4-bit QLoRA via [Unsloth](https://github.com/unslothai/unsloth).

> **Note:** FunRL's official code/weights aren't publicly released yet (pending review). This project reimplements the core idea described in the paper — entropy-regularized GRPO for tool-calling — rather than using their code directly.

## Results

Evaluated on BFCL's `parallel` category (multi-call prompts) — **never seen during training**:

| Model | Avg. score (n=15) |
|---|---|
| Base Qwen3.5-4B | 0.33 |
| Fine-tuned (this repo) | **0.73** |

More than double, on genuinely unseen, harder task structure — suggests the training generalized rather than memorized.

**Caveats:** small training set and small eval set by design (free T4, one day) — custom scorer (not the official BFCL harness), LoRA adapter rather than a full fine-tune. Directional result, not a formal benchmark claim.

## Approach

Two-stage GRPO training on the [Berkeley Function-Calling Leaderboard (BFCL)](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) dataset, continuing the same LoRA adapter across stages:

1. **`simple`** — single function, single call (400 examples, 1 epoch)
2. **`multiple`** — disambiguating between several candidate functions in the prompt (200 examples, 1 epoch)

Reward function does structural comparison against BFCL's ground truth (function name match + per-argument value match against allowed value sets) rather than exact-string or AST equality.

The "FunRL-style" part is a lowered KL coefficient (`beta=0.02` vs TRL's default `0.04`) in the GRPO loss, letting the policy explore more diverse chain-of-thought / call patterns instead of collapsing early to the reference model's behavior — a practical stand-in for the paper's "strategic entropy injection," since the official implementation isn't available yet.

## Setup

```bash
pip install --upgrade unsloth unsloth_zoo -q
pip install --upgrade transformers trl datasets -q
```

Key config:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-4B",
    max_seq_length = 2048,
    load_in_4bit = True,
    fast_inference = False,  # vLLM doesn't support Qwen3.5 yet
)

model = FastLanguageModel.get_peft_model(
    model, r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)

config = GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=256,
    beta=0.02,
    learning_rate=1e-6,
    chat_template_kwargs={"enable_thinking": False},  # critical — see notes below
)
```

## Notable gotchas (if you're replicating this)

- **Qwen3.5-4B can't train in fp16** — Unsloth falls back to float32 automatically. bf16 needs Ampere+ (A100/H100); T4 (compute capability 7.5) doesn't support it.
- **Qwen3.5 is multimodal (VLM)** — `tokenizer` is actually a `Qwen3VLProcessor`, which doesn't expose `pad_token_id` directly. Patch it: `tokenizer.pad_token_id = tokenizer.tokenizer.pad_token_id`.
- **Thinking-token truncation** — Qwen3.5 is a reasoning model by default; without `enable_thinking=False`, it burns the entire completion budget on `<think>` reasoning and never reaches the actual tool call, tanking reward to the floor. This was the single biggest bug in early training runs.
- **vLLM doesn't support Qwen3.5 yet** — must set `fast_inference=False` in Unsloth, which means slower rollout generation (no fast-inference speedup).
- **Kaggle free-tier session instability** — long GRPO runs (400 steps, several hours on a T4) are prone to disconnects that kill the kernel. Use tight `save_steps` (10–25) and `save_total_limit` so a disconnect costs minutes, not hours, and resume with `trainer.train(resume_from_checkpoint=True)`.

## Repo contents

- `funrl-qwen3-5-annotated-v2.ipynb` — full training + evaluation notebook (both stages, reward curves, base-vs-fine-tuned comparison). Left largely as-run, including real debugging cells from mid-training recovery — not scrubbed into a clean linear script.

## Acknowledgments

- [FunRL / RLFC paper](https://arxiv.org/html/2508.05118v1) by BingguangHao et al. for the underlying method this reimplements
- [Unsloth](https://github.com/unslothai/unsloth) for making QLoRA GRPO feasible on a single T4
- [Gorilla LLM / BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) for the training and evaluation dataset
