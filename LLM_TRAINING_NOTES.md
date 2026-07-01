# LLM Training Notes

Study notes covering the full training pipeline for the GPT model built in this project, including GRPO alignment.

---

## Part 1 — Full LLM Training Pipeline

### Phase 1 — Data Preparation (Done)

Covered in `src/working_with_text_data.ipynb`:

- **Text collection & cleaning** — gather raw corpus, remove noise
- **Tokenization** — convert text to tokens (`SimpleTokenizer`, `tiktoken` / BPE)
- **Vocabulary building** — map tokens to integer IDs + special tokens (`<|unk|>`, `<|endoftext|>`)
- **Dataset & DataLoader** — sliding window with `stride` to produce `(input, target)` pairs (`GPTDatasetV1`)
- **Embeddings** — token embeddings + positional embeddings combined

### Phase 2 — Model Architecture (Done)

Covered in `src/working_with_text_data.ipynb` and `src/GPTModel.ipynb`:

- **Self-attention** — Query/Key/Value projections (`SelfAttention_v1/v2`)
- **Causal (masked) attention** — upper triangular mask to prevent future token leakage (`CausalAttention`)
- **Multi-head attention** — parallel heads combined (`MultiHeadAttention`)
- **Feed-forward network** — 2-layer MLP with GELU activation
- **Layer normalization** — Pre-LN style (`LayerNorm`)
- **Residual connections** — skip connections around each sub-layer
- **Full Transformer block** — assembling the pieces above (`TransformerBlock`)
- **GPT model** — stacking N Transformer blocks + final linear head projecting to vocab size (`GPTModel`)

### Phase 3 — Pre-training

Core "learning language" phase:

- **Objective** — next-token prediction (causal language modeling)
- **Loss function** — cross-entropy between predicted logits and target token IDs
- **Optimizer** — AdamW with weight decay
- **Learning rate schedule** — warmup + cosine decay
- **Training loop** — forward pass → compute loss → backward pass → gradient clipping → optimizer step
- **Gradient accumulation** — simulate larger batch sizes if GPU memory is limited
- **Checkpoint saving** — save model weights periodically

### Phase 4 — Evaluation During Pre-training

- **Perplexity** — primary metric; lower = better language modeling
- **Train vs. validation loss** — detect overfitting
- **Text generation samples** — qualitative check that the model produces coherent text
- **Learning curves** — plot loss over steps to detect instabilities

### Phase 5 — Loading Pre-trained Weights (Optional Shortcut)

Covered in `model_testing/gpt-oss.ipynb`:

- Load weights from OpenAI GPT-2 into your architecture
- Useful for validating your architecture is correct before training from scratch

### Phase 6 — Supervised Fine-tuning / Instruction Tuning (Done)

Covered in `fine_tuning/`:

- **SFT** — train on `(instruction, response)` pairs so the model follows instructions
- Done with TRL + Qwen via `trl_sft_demo.ipynb` and `qwen3_fine_tuning.ipynb`
- **LoRA / QLoRA** — fine-tune only low-rank adapter weights (Unsloth checkpoints in `fine_tuning/outputs/`)

### Phase 7 — Alignment (Advanced, Post-SFT)

- **RLHF** — Reinforcement Learning from Human Feedback (PPO-based)
- **DPO** — Direct Preference Optimization — covered in `src/dpo.ipynb`
- **GRPO** — Group Relative Policy Optimization — see Part 2 below

### Summary Roadmap

```
[Done]     Phase 1: Tokenization + DataLoader
[Done]     Phase 2: Full GPT model architecture
[Next]     Phase 3: Pre-training loop (loss, optimizer, scheduler, checkpoints)
[Next]     Phase 4: Evaluation (perplexity, loss curves, generation)
[Optional] Phase 5: Load GPT-2 weights to validate architecture
[Done*]    Phase 6: SFT fine-tuning (done on Qwen, can be applied to this model)
[Future]   Phase 7: Alignment (DPO done, GRPO — see Part 2)
```

---

## Part 2 — GRPO Alignment From Scratch

Based on the GRPO implementation done in `rl_study/src/pong_grpo_cnn/`.

### Conceptual Mapping: Pong GRPO → LLM GRPO

| Pong GRPO | LLM GRPO |
|---|---|
| CNN actor (`GRPOAgent`) | `GPTModel` (the policy `π_θ`) |
| Action from environment step | Token generated autoregressively |
| Environment reward (score delta) | Reward function scoring a full completion |
| Group = T timesteps in rollout | Group = G completions for the same prompt |
| `get_action(state)` → 1 log_prob | `generate(prompt, G)` → G sequence log_probs |
| `evaluate_actions(states, actions)` | `compute_sequence_log_probs(completion)` |
| No reference model | Frozen reference model `π_ref` (KL penalty) |
| `compute_group_advantages(gamma)` | `compute_group_advantages()` (same formula, no gamma) |

---

### Step 1 — Starting Point: Policy + Reference Model

- The **policy model** is `GPTModel` (loaded with GPT-2 weights or SFT-tuned)
- Create a **frozen reference model**: a copy of the policy at GRPO start time, never updated
- New vs Pong: the reference model is needed to compute the KL penalty that prevents the model from drifting too far from its original behavior

---

### Step 2 — Define a Reward Function

Replaces the Atari environment's reward signal. Rule-based examples (no reward model needed):

- **Format reward** — does the response follow the expected structure?
- **Correctness reward** — for math/coding tasks, does it produce the right answer?
- **Length reward** — penalize responses that are too short or too long

The existing `instruction-data-with-preference.json` (1100 entries) can derive simple binary rewards from `chosen` vs `rejected` labels, or a rule-based checker can be written.

---

### Step 3 — Prepare a Prompt Dataset

- A collection of prompts the model will respond to
- Unlike DPO, only the **prompts** are needed — GRPO generates its own completions at training time
- Build a `DataLoader` that yields batches of tokenized prompts

---

### Step 4 — Generate G Completions Per Prompt (Rollout Collection)

**Pong equivalent:**
```python
for _ in range(ROLLOUT_LENGTH):
    action, log_prob = agent.get_action(state)
```

**LLM version:**

- For each prompt in the batch, generate G completions using temperature sampling (e.g., G=4 or G=8)
- For each completion, record:
  - The generated token IDs
  - The **token-level log probs** from the current policy (summed over response tokens, masking the prompt — same masking logic as `chosen_mask` in `src/dpo.ipynb`)
- Done under `torch.no_grad()` — just collecting experience

---

### Step 5 — Score Completions with Reward Function

- Run each of the G completions through the reward function
- Result: `[r_1, r_2, ..., r_G]` for each prompt

---

### Step 6 — Compute Group Advantages

Exact same formula as `compute_group_advantages()` in Pong:

```
A_i = (r_i - mean(r_1..G)) / (std(r_1..G) + eps)
```

Key difference: the group is the **G completions for one prompt**, not the T timesteps of a rollout.
No gamma/discounting needed — the reward is per-completion, not per-step.

---

### Step 7 — GRPO Loss

**Same clipped surrogate as Pong** (`compute_grpo_loss`):

```
ratio    = exp(new_log_prob - old_log_prob)
L_policy = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

**New: KL penalty** (not in Pong — needed to prevent forgetting):

```
KL      = log_prob_policy - log_prob_reference   (per token, summed over completion)
L_total = L_policy + β * KL
```

No value loss — GRPO is critic-free, same as Pong.

---

### Step 8 — Training Loop

Same structure as `train_grpo()` in Pong:

1. Sample a batch of prompts
2. Generate G completions with current policy (`torch.no_grad()`)
3. Score completions → rewards
4. Compute group advantages
5. For `GRPO_EPOCHS` epochs:
   - Recompute log probs from current policy (grad enabled)
   - Compute KL from reference model (no grad)
   - Compute loss → backprop → gradient clipping → optimizer step
6. Log metrics (reward trend, KL divergence, clip fraction, approx KL)
7. Checkpoint periodically

---

### Step 9 — Evaluation

- Generate sample completions qualitatively
- Monitor **mean reward** per update (equivalent to Pong's episode reward curve)
- Monitor **KL from reference** — should stay bounded (too high = model drifted dangerously)
- Monitor **clip fraction** and **approx KL** as in Pong

---

### What's the Same vs What's New

| Same as Pong GRPO | New for LLM GRPO |
|---|---|
| Clipped surrogate loss | KL penalty against reference model |
| Group normalization formula | Token-level log prob computation with masking |
| No value function / critic | G completions per prompt (not T rollout steps) |
| Multiple GRPO epochs | Reward function (replaces environment) |
| Gradient clipping | Autoregressive generation for sampling |
| MLflow + W&B logging | — |

The biggest conceptual shift: the **group** changes. In Pong, the group was all timesteps in the rollout. In the LLM, the group is the **G different completions generated for the same prompt** — making the advantage estimate per-completion rather than per-step.
