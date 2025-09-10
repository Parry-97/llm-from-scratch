# Notes: Final Transformation in GPT-2–like Models

## 1. Final Stage of Transformer Block

- Hidden state: a **768D vector** representing contextual meaning.
- Final linear transformation: maps hidden state to **50257D vocabulary logits**.
- Formula:

  $$
  z = W h, \quad W \in \mathbb{R}^{50257 \times 768}, \quad h \in \mathbb{R}^{768}
  $$

- Each row $w_i$ of $W$ is the embedding of a token, so:

  $$
  z_i = w_i \cdot h
  $$

**Interpretation:** The logits are just dot products: alignment of hidden state with each word vector.

---

## 2. Does It Lose Contextual Information?

- **Preserved:** The linear projection keeps contextual meaning — it’s testing compatibility with every word.
- **Lost:** Only when collapsing probabilities into a single token (sampling/argmax) do we lose rich nuance.

**Analogy:**

- Hidden state = a detailed portfolio.
- Projection = cover letter showing how portfolio matches job listings (tokens).
- Sampling a token = picking _one bullet point_.

---

## 3. Weight Tying in GPT-2

- GPT-2 **ties input embeddings and output projection** (same matrix, transposed).
- Ensures words live in a **shared semantic space** for both encoding (reading) and decoding (generating).

**Benefits:**

1. Consistency of word meanings.
2. Efficiency (fewer parameters, less overfitting).
3. Every update to prediction improves input representations too.

---

## 4. Geometric Intuition

- **Tied embeddings:**
  - Hidden state is like a dart thrown onto the same map where word vectors live.
  - Dot product measures alignment → high score = likely next token.

- **Untied embeddings:**
  - Input and output live in **separate maps**.
  - Model wastes effort learning a “bridge” between them.
  - Less stable, less efficient.

---

## 5. Expressiveness and Rank

- Hidden state: 768D.
- Vocabulary logits: 50kD.
- Rank of $W$ ≤ 768 → the 50k scores come from a **768D bottleneck**.
- This bottleneck enforces **shared structure** in how words are related.

---

## 6. PyTorch Linear Layer Convention

- `nn.Linear(in_features, out_features)` stores weights as $(out\_features, in\_features)$.
- Multiplication:

  $$
  y = x W^T + b
  $$

- For GPT-2:
  - $x \in \mathbb{R}^{768}$.
  - $W \in \mathbb{R}^{50257 \times 768}$.
  - Output logits: $50257$.

👉 **Rule of thumb:**

- **Rows = output directions (word embeddings).**
- **Dot product with hidden state = token score.**

---

## 7. Key Takeaways

- The final transformation is not “throwing away” context; it’s scoring words.
- The _loss of richness_ happens only when you collapse to one token.
- Tied embeddings unify understanding and generation in one semantic geometry.
- Linear layer math = dot products with each word vector.
- PyTorch stores weights as (out, in), so rows = outputs.
