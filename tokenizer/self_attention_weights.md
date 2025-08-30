# Why Trainable Weight Matrices in Self-Attention?

This document summarizes our discussion about why the Transformer architecture uses trainable projection matrices for **queries (Q)**, **keys (K)**, and **values (V)**, instead of directly using raw embeddings.

---

## 1. The Simple Alternative (No Trainable Weights)

If we skipped the trainable weights:

1. Tokens are represented as embeddings.
2. Attention scores are computed by dot products of embeddings.
3. Weights are normalized with softmax.
4. Context = weighted sum of embeddings.

This would still give context mixing, **but only in the raw embedding space**.

---

## 2. Why Use Trainable Projections?

The embedding space is not always the right space to compare tokens.  
Learnable projections provide flexibility:

- **Wq**: decides what features matter when asking a question (the query).
- **Wk**: decides how tokens present themselves to be compared (the key).
- **Wv**: decides what information is passed along if a token is attended to (the value).

This separation lets the model learn **task-specific matching** rather than being stuck with raw embedding similarity.

---

## 3. The Role of Keys (Wk)

### Without Wk

- Queries would always compare against raw embeddings.
- Queries must adapt to the quirks of the embedding space.
- Limits the model’s ability to capture nuanced relationships.

### With Wk

- Keys define a **learned "address" space** for comparison.
- Queries and keys meet in a **shared latent space**.
- Makes it possible to distinguish tokens that are semantically close but play different roles (e.g., subject vs. object).

---

## 4. Multi-Head Attention Benefit

Each head learns a **different latent space** by having its own Wq, Wk, Wv.

- One head may learn to track grammatical roles.
- Another may learn semantic similarity.
- Another may learn positional dependencies.

If all heads used raw embeddings, they would be redundant.

---

## 5. What is a Latent Space?

A **latent space** is a hidden, learned feature space where relationships are redefined in ways useful for the task.

- Embedding space: raw word similarities (dog ≈ cat).
- Latent space: new features (syntax, position, role).

**Analogy:**  
Word embeddings are like GPS coordinates (geography).  
But if we want to compare cities by population, we need a new space where “distance” reflects population.  
That’s what Wq and Wk create: a latent space for attention matching.

---

## 6. Toy Example (Why Wk Matters)

Suppose embeddings:

- `dog = [1, 1]`
- `cat = [1, 0.9]`

A query `q = [1, 0]` asks: “Who is the subject?”

- **Without Wk**:  
  q ⋅ dog = 1, q ⋅ cat = 1 → they look identical.

- **With Wk (learned projection)**:  
  Keys can be re-mapped so dot products distinguish dog vs. cat **based on grammatical role**, even though embeddings were nearly identical.

Thus, **Wk allows more nuanced, higher-level matching between queries and values**.

---

## 7. Summary

- **Wq**: shapes the question (what I’m looking for).
- **Wk**: shapes how tokens are recognized (the address).
- **Wv**: shapes what information is carried (the content).

Together, they enable flexible, nuanced, task-specific self-attention that raw embeddings alone cannot provide.
