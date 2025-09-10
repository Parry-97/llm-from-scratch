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

- Temperature doesn't just "scale everything."
- It changes **entropy** of the distribution.
- Controls trade-off between **deterministic** and **exploratory** text generation.

---

## Top-k Sampling in LLMs

### 7. What is Top-k Sampling?

Top-k sampling is a decoding strategy that restricts the sampling pool to only the **k most likely tokens** at each generation step.

**Algorithm**:
1. Compute probabilities for all tokens in vocabulary
2. Sort tokens by probability (descending)
3. Keep only the top k tokens
4. Renormalize probabilities among these k tokens
5. Sample from this reduced distribution

---

### 8. Mathematical Formulation

Given original probabilities $p_1 \geq p_2 \geq ... \geq p_V$ (sorted):

1. **Select top-k tokens**: $\mathcal{K} = \{1, 2, ..., k\}$

2. **Renormalize**:
   $$
   \tilde{p}_i = \begin{cases}
   \frac{p_i}{\sum_{j \in \mathcal{K}} p_j} & \text{if } i \in \mathcal{K} \\
   0 & \text{otherwise}
   \end{cases}
   $$

3. **Sample** from distribution $\tilde{p}$

---

### 9. Why Use Top-k?

**Problems it solves**:
- **Long tail problem**: LLMs assign small probabilities to many unlikely tokens
- **Degeneracy**: Pure sampling can select very improbable tokens
- **Quality control**: Restricts generation to plausible tokens

**Benefits**:
- Prevents nonsensical token selection
- Maintains diversity (unlike greedy decoding)
- Computationally efficient (only need to sort top-k)

---

### 10. Effect of k Value

- **Small k (e.g., k=5)**:
  - Very restrictive
  - High quality but less diverse
  - Similar to greedy but with slight variation

- **Medium k (e.g., k=40)**:
  - Balance between quality and diversity
  - Most commonly used in practice

- **Large k (e.g., k=100+)**:
  - More diverse outputs
  - Risk of including poor quality tokens
  - Approaches pure sampling

---

### 11. Example

Vocabulary: ["the", "a", "cat", "dog", "xyz", "???"]
Probabilities: [0.4, 0.3, 0.15, 0.10, 0.03, 0.02]

**Without top-k**: All tokens possible (including "xyz", "???")

**With k=3**:
- Keep: ["the", "a", "cat"] with probs [0.4, 0.3, 0.15]
- Renormalize: [0.47, 0.35, 0.18]
- Sample from these 3 only

---

### 12. Combining with Temperature

Top-k and temperature are often used **together**:

1. Apply temperature to logits: $z_i' = z_i / T$
2. Compute softmax: $p_i = \text{softmax}(z_i')$
3. Apply top-k filtering
4. Sample from filtered distribution

**Common combinations**:
- Low temp + small k = **focused, high-quality**
- High temp + large k = **creative, diverse**
- Medium both = **balanced generation**

---

### 13. Top-k vs Other Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Greedy** | Always pick max | Deterministic | Repetitive |
| **Top-k** | Sample from top k | Quality + diversity | Fixed cutoff |
| **Top-p** | Sample from cumulative p | Dynamic cutoff | More complex |
| **Beam Search** | Keep n best sequences | High quality | Computationally expensive |

---

### 14. Implementation Considerations

```python
def top_k_sampling(logits, k, temperature=1.0):
    # Apply temperature
    logits = logits / temperature
    
    # Get top-k indices and values
    top_k_values, top_k_indices = torch.topk(logits, k)
    
    # Set all other logits to -inf
    filtered_logits = torch.full_like(logits, -float('inf'))
    filtered_logits.scatter_(0, top_k_indices, top_k_values)
    
    # Apply softmax and sample
    probs = F.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, 1)
    return next_token
```

---

### 15. Practical Tips

1. **Start with k=40-50** for general text generation
2. **Adjust based on task**:
   - Code generation: lower k (10-20)
   - Creative writing: higher k (50-100)
   - Factual Q&A: lower k with low temperature
3. **Monitor token diversity** in outputs
4. **Consider vocabulary size** when choosing k
5. **Use with temperature** for fine-grained control

---
