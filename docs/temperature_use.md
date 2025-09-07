# Temperature Sampling in LLMs

### 1. Softmax without temperature

Given logits $z_i$, probabilities are:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

--- ### 2. Adding temperature

With temperature $T$:

$$
p_i(T) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

- $T = 1$: normal softmax.
- $T < 1$: sharper distribution (peaked).
- $T > 1$: flatter distribution (more random).

---

### 3. Why scaling changes the shape

The key is the **ratio of probabilities**:

$$
\frac{p_i(T)}{p_k(T)} = e^{(z_i - z_k)/T}.
$$

- Denominator cancels out.
- The effect depends only on **differences in logits** scaled by $1/T$.

---

### 4. Intuition

- **Low temperature ($T < 1$)**:
  - Differences between logits get magnified.
  - One or few tokens dominate.

- **High temperature ($T > 1$)**:
  - Differences shrink.
  - Probabilities move closer together.

- **Extreme case ($T \to \infty$)**:
  - All differences vanish.
  - Distribution → uniform (all tokens equally likely).

---

### 5. Example

Logits: $[2, 1, 0]$

- $T = 1$: $[0.67, 0.24, 0.09]$
- $T = 0.5$: $[0.87, 0.12, 0.02]$ (more peaked)
- $T = 2$: $[0.49, 0.30, 0.20]$ (flatter)

---

### 6. Big picture

- Temperature doesn’t just “scale everything.”
- It changes **entropy** of the distribution.
- Controls trade-off between **deterministic** and **exploratory** text generation.

---
