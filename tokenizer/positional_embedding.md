# Positional encodings for LLM token embeddings (quick explainer)

Self‑attention is permutation‑invariant: without positions, a Transformer can’t tell “the cat sat” from “sat the cat.” Positional encodings inject order information alongside token embeddings so the model can reason over sequences.

Two broad families are common:

- Absolute positions: add a position‑dependent vector to token embeddings before attention.
- Relative positions: inject position information inside attention (modify Q/K or attention logits), so the model learns relations like “current token is 3 to the left of that token.”

## Absolute positional encodings

### Sinusoidal (Vaswani et al., 2017)

- Construction: a fixed, non‑learned table P[pos, d] made of sin/cos waves with exponentially scaled frequencies across embedding dimensions.
- Use: x = token_embed(ids) + P[pos]. No extra parameters; deterministic.
- Pros
  - Parameter‑free; easy to implement.
  - Some ability to generalize beyond trained context length.
- Cons
  - Pattern is fixed; doesn’t adapt to data.
  - Can degrade when extrapolating far beyond training lengths.

Example (PyTorch):

```python
import math
import torch

def sinusoidal_positions(max_len, d_model, device=None, dtype=torch.float32):
    pos = torch.arange(max_len, device=device, dtype=dtype).unsqueeze(1)      # [L,1]
    i = torch.arange(0, d_model, 2, device=device, dtype=dtype)               # [D/2]
    inv_freq = torch.exp(-math.log(10000.0) * (i / d_model))                  # [D/2]
    pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)            # [L,D]
    angles = pos * inv_freq                                                   # [L,D/2]
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe

# usage (ids: [B,L], tok_emb: nn.Embedding(vocab, d_model))
B, L, d_model = 2, 8, 16
ids = torch.randint(0, 5000, (B, L))
tok_emb = torch.nn.Embedding(5000, d_model)
pe = sinusoidal_positions(L, d_model, device=ids.device)
x = tok_emb(ids) + pe[:L].unsqueeze(0)  # [B,L,D]
```

### Learned absolute embeddings

- Construction: a learned lookup table of size [max_len, d_model]; like token embeddings but indexed by absolute position.
- Use: x = token_embed(ids) + pos_table[pos].
- Pros
  - Simple; often strong within the trained context window.
- Cons
  - No extrapolation beyond max_len without resizing/retraining.
  - Ties model to a fixed context length.

## Relative positional encodings

These encode distances i−j between tokens rather than absolute indices. They’re applied inside attention, typically affecting Q/K or the attention logits.

### T5/Shaw‑style relative attention bias

- Idea: add a learned bias to attention logits based on relative offset (i−j). Offsets are bucketed so far distances share parameters.
- Implementation sketch: attention_logits += rel_bias[bucket(i−j)] (often per‑head in T5).
- Pros
  - Captures order and distance; generalizes across lengths.
  - Strong empirical results in encoders/seq‑to‑seq.
- Cons
  - Extra parameters/compute; needs bucket design (range and granularity).

### RoPE (rotary position embeddings)

- Idea: apply a position‑dependent rotation to Q and K so their dot products carry relative phase information. Widely used in modern decoder‑only LLMs.
- Where: before computing attention scores, rotate Q and K per head.
- Pros
  - Efficient; no learned parameters for positions; good long‑context behavior.
  - Cache‑friendly at inference; works well in large LLMs.
- Cons
  - For ultra‑long contexts you often need scaling tweaks (e.g., NTK‑aware scaling, XPos, YaRN).

Minimal RoPE application (per head_dim, D even):

```python
import torch

def apply_rope(x, base=10000.0):
    # x: [B,H,L,D], D even
    B, H, L, D = x.shape
    half = D // 2
    t = torch.arange(L, device=x.device, dtype=x.dtype)                                 # [L]
    freq = base ** (-torch.arange(half, device=x.device, dtype=x.dtype) / half)         # [half]
    ang = torch.einsum('l,d->ld', t, freq)                                              # [L,half]
    cos = ang.cos().unsqueeze(0).unsqueeze(0)                                           # [1,1,L,half]
    sin = ang.sin().unsqueeze(0).unsqueeze(0)                                           # [1,1,L,half]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# Example usage in attention: q = apply_rope(q); k = apply_rope(k)
```

### ALiBi (Attention with Linear Biases)

- Idea: add a per‑head linear bias proportional to token distance directly to attention logits.
- Sketch: attention_logits[b,h,i,j] += slopes[h] \* (j − i) (for causal decoders, j ≤ i).
- Pros
  - Extremely simple; extrapolates well to longer sequences.
- Cons
  - Can underperform RoPE/T5 on some tasks; quality depends on slope schedule.

## Practical guidance (decoder‑only LLMs)

- Default choice: RoPE (used in many popular LLMs) for robust long‑context behavior and efficiency.
- Want simplicity and strong extrapolation with minimal changes? ALiBi is attractive.
- Small/short‑context models: learned absolute embeddings are simple and effective; sinusoidal is great for education/baselines.
- Inference caching: absolute PEs are added to embeddings and don’t alter attention structure; relative methods change attention logits or Q/K but remain cache‑friendly.
- Extending context: with RoPE consider NTK‑aware scaling, XPos, or YaRN; with absolute learned PEs you typically must resize/retrain.

## Why they’re useful

- Enable order‑aware representations so attention can model syntax, dependencies, and temporal relationships.
- Relative methods (T5/Shaw, RoPE, ALiBi) often generalize better to lengths not seen in training and encode distances more directly.

## References

- Vaswani et al., 2017 — Attention Is All You Need: <https://arxiv.org/abs/1706.03762>
- Shaw et al., 2018 — Self‑Attention with Relative Position Representations: <https://arxiv.org/abs/1803.02155>
- Raffel et al., 2020 — T5: Exploring the Limits of Transfer Learning with a Unified Text‑to‑Text Transformer: <https://arxiv.org/abs/1910.10683>
- Su et al., 2021 — RoFormer: Rotary Position Embedding: <https://arxiv.org/abs/2104.09864>
- Press et al., 2021 — ALiBi: Train Short, Test Long: <https://arxiv.org/abs/2108.12409>
- Optional overview — The Annotated Transformer: <http://nlp.seas.harvard.edu/2018/04/03/attention.html>
