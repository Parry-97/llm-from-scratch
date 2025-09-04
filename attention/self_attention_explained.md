# Understanding Self-Attention: Matrix Operations and Attention Scores

## Overview

This document explains the mathematical operations in the self-attention mechanism, specifically focusing on why the transpose operation is needed for computing attention scores but not for context vectors, and what each row of the attention scores represents.

## The Self-Attention Implementation

The self-attention mechanism involves three key learnable weight matrices:

- **W_query**: Transforms inputs into query vectors
- **W_key**: Transforms inputs into key vectors
- **W_value**: Transforms inputs into value vectors

```python
# From self_attention.py
keys = x @ self.W_key        # shape: [6,2]
queries = x @ self.W_query   # shape: [6,2]
values = x @ self.W_value    # shape: [6,2]

attn_scores = queries @ keys.T  # shape: [6,6]
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
context_vec = attn_weights @ values  # shape: [6,2]
```

## Why Transpose is Needed for Attention Scores

### The Mathematical Reasoning

The attention mechanism needs to compute **pairwise similarities** between all queries and keys to determine how much each position should attend to every other position.

#### Attention Scores Computation

```python
attn_scores = queries @ keys.T  # [6,2] @ [2,6] = [6,6]
```

- **Input shapes**: Both queries and keys are `[6, 2]` (6 positions, 2 dimensions)
- **Goal**: Compute dot product between each query and each key
- **Required output**: `[6, 6]` matrix where element `[i,j]` = similarity(query_i, key_j)

The transpose is essential because:

1. Query at position `i` is row `i` of the queries matrix
2. Key at position `j` is row `j` of the keys matrix
3. To compute their dot product, we need to align query row `i` with key row `j`
4. Transposing keys converts rows to columns, enabling this alignment

### Why No Transpose for Context Vector

```python
context_vec = attn_weights @ values  # [6,6] @ [6,2] = [6,2]
```

The context vector computation performs **weighted aggregation** of values:

- **Attention weights**: `[6, 6]` - how much each position attends to all positions
- **Values**: `[6, 2]` - value vectors at each position
- **Output**: `[6, 2]` - weighted combination for each position

No transpose needed because:

- Row `i` of attention weights contains weights for position `i`
- These weights directly multiply with value vectors (rows of values matrix)
- Matrix multiplication naturally performs the weighted sum

## Visual Example

Consider a simplified case with 3 positions and 2 dimensions:

### Computing Attention Scores (Requires Transpose)

```
Queries: [[q1_d1, q1_d2],    Keys: [[k1_d1, k1_d2],
          [q2_d1, q2_d2],            [k2_d1, k2_d2],
          [q3_d1, q3_d2]]             [k3_d1, k3_d2]]
         (3×2)                       (3×2)

Need: q1·k1, q1·k2, q1·k3, q2·k1, q2·k2, q2·k3, q3·k1, q3·k2, q3·k3

Solution: Queries @ Keys.T = (3×2) @ (2×3) = (3×3)

Result: [[q1·k1, q1·k2, q1·k3],
         [q2·k1, q2·k2, q2·k3],
         [q3·k1, q3·k2, q3·k3]]
```

### Computing Context Vectors (No Transpose)

```
Weights: [[w11, w12, w13],    Values: [[v1_d1, v1_d2],
          [w21, w22, w23],              [v2_d1, v2_d2],
          [w31, w32, w33]]              [v3_d1, v3_d2]]
         (3×3)                          (3×2)

Computation: Weights @ Values = (3×3) @ (3×2) = (3×2)

Result for row i: wi1*v1 + wi2*v2 + wi3*v3
```

## Understanding Attention Score Rows

Each row in the attention scores matrix has a specific interpretation:

### What Each Row Represents

**Row `i` of the attention scores matrix represents how strongly the query at position `i` relates to all keys in the sequence.**

For a `[6,6]` attention scores matrix:

```
Attention Scores Matrix:
        [to_pos_0, to_pos_1, to_pos_2, to_pos_3, to_pos_4, to_pos_5]
Row 0:  [  score,    score,    score,    score,    score,    score  ]  ← Query at position 0
Row 1:  [  score,    score,    score,    score,    score,    score  ]  ← Query at position 1
Row 2:  [  score,    score,    score,    score,    score,    score  ]  ← Query at position 2
Row 3:  [  score,    score,    score,    score,    score,    score  ]  ← Query at position 3
Row 4:  [  score,    score,    score,    score,    score,    score  ]  ← Query at position 4
Row 5:  [  score,    score,    score,    score,    score,    score  ]  ← Query at position 5
```

### After Softmax: Attention Weights

After applying softmax normalization:

```python
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
```

> [!IMPORTANT]
> In words, the **softmax** applies the standard exponential function to each element z_i
> and normalizes these values by dividing by the sum of all these exponentials.

Each row becomes a **probability distribution** (sums to 1):

- **Row 0**: Probability distribution for how position 0 combines information from all positions
- **Row 1**: Probability distribution for how position 1 combines information from all positions
- And so on...

### Practical Example

Consider processing the sentence "The cat sat":

**Attention Scores (before softmax):**

```
         ["The", "cat", "sat"]
"The":   [ 0.9,   2.1,   0.3 ]  ← "The" has high affinity for "cat"
"cat":   [ 1.5,   3.2,   2.8 ]  ← "cat" relates to itself and "sat"
"sat":   [ 0.2,   2.9,   3.5 ]  ← "sat" strongly relates to itself
```

**Attention Weights (after softmax):**

```
         ["The", "cat", "sat"]
"The":   [ 0.20,  0.70,  0.10]  ← 70% attention on "cat"
"cat":   [ 0.10,  0.45,  0.45]  ← Balanced between self and "sat"
"sat":   [ 0.05,  0.35,  0.60]  ← 60% self-attention
```

### Impact on Context Vectors

When computing context vectors:

```python
context_vec = attn_weights @ values
```

Each row of attention weights determines the final output:

- **Position 0's output**: 20% of value₀ + 70% of value₁ + 10% of value₂
- **Position 1's output**: 10% of value₀ + 45% of value₁ + 45% of value₂
- **Position 2's output**: 5% of value₀ + 35% of value₁ + 60% of value₂

## Key Insights

1. **Transpose for Similarity**: The transpose in `queries @ keys.T` enables computing pairwise similarities between all query-key pairs.

2. **No Transpose for Aggregation**: The context vector computation `attn_weights @ values` directly performs weighted aggregation without needing transpose.

3. **Row Interpretation**: Each row in the attention matrix represents an "attention distribution" - answering "When I'm at position `i`, how much should I look at each position to gather information?"

4. **Softmax Application**: Applied along `dim=-1` (across rows) to ensure each position's attention weights form a valid probability distribution.

5. **Self-Attention**: The diagonal of the attention matrix often has high values, as positions frequently need to attend to themselves.

## Implementation Comparison

### Using nn.Parameter (SelfAttention)

```python
self.W_query = torch.nn.Parameter(torch.randn(d_in, d_out))
keys = x @ self.W_key  # Manual matrix multiplication
```

### Using nn.Linear (SelfAttention_v2)

```python
self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
keys = self.W_key(x)  # Linear layer handles the operation
```

Both achieve the same result, but `nn.Linear`:

- Provides better weight initialization (Kaiming/He initialization)
- Optionally includes bias terms
- Offers cleaner, more maintainable code
- Internally handles the matrix multiplication

## Summary

The self-attention mechanism elegantly uses matrix operations to:

1. **Compute relationships** between all positions (requiring transpose for dot products)
2. **Create attention distributions** (softmax normalization per row)
3. **Aggregate information** (weighted sum without transpose)

This design allows each position to dynamically determine what information from the entire sequence is most relevant for its computation, forming the foundation of modern transformer architectures.
