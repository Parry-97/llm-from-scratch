- Think of each **column of `values`** as one feature across all tokens.

- For a given query $i$, each feature in the **context vector row** is computed
  by taking the **corresponding feature column** of `values` and forming a
  **weighted sum across tokens** using the attention weights `attn_weights[i,:]`.

- Doing this for all feature columns at once gives the **entire row** of the
  context vector for that token.

So: **row = token context vector**, **column = feature**, and the attention
weights control how each token’s features contribute to that row.

It’s really just a fancy way of saying: "we mix features from all tokens
according to attention, feature by feature, and collect them into the
token’s context vector row."

# 2 tokens, 2 features

```python
values = [[v0_f0, v0_f1],
          [v1_f0, v1_f1]] # shape [2,2]
```

# Attention weights for token 0 (row 0)

```python
attn_weights = [[0.3, 0.7]] # shape [1,2]
```
