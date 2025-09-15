# 🔑 Notes on Accuracy, Differentiability & PyTorch Evaluation Modes

## 1. Why Accuracy is Not Differentiable

- **Definition**:

  $$
  \text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}\{ \hat{y}_i = y_i \}
  $$

  where $\hat{y}_i = \arg\max_j p(y=j \mid x_i, \theta)$.

- **Reason**:
  - Uses **`argmax`** and **indicator functions** → piecewise constant, discontinuous.
  - Small changes in logits usually **don’t change** predicted class.
  - Gradient is **0 almost everywhere**, and undefined at class-boundary jumps.

- **Consequence**:
  Accuracy provides **no gradient signal** for optimization.

- **Solution**:
  Use **differentiable surrogate losses**:
  - Cross-entropy (most common for classification).
  - Hinge loss (e.g., SVMs).

---

## 2. PyTorch Evaluation Modes

### `torch.no_grad()`

- Context manager: disables **autograd tracking** inside block.
- Reduces memory usage & speeds up inference.
- Example:

  ```python
  with torch.no_grad():
      outputs = model(inputs)
  ```

### `model.eval()`

- Sets model into **evaluation mode**:
  - **Dropout** → disabled (deterministic output).
  - **BatchNorm** → uses running statistics.

- Does **not** affect gradient tracking.

### `torch.inference_mode()`

- Newer (PyTorch ≥ 1.9) inference context.
- Extends `torch.no_grad()` by also disabling **autograd version counter updates**.
- This makes it **faster** and **more memory-efficient** than `torch.no_grad()`.
- Best for pure inference (when you’re not doing in-place modifications that might later need gradients).
- Example:

  ```python
  with torch.inference_mode():
      outputs = model(inputs)
  ```

---

## 3. Best Practices for Inference

```python
model.eval()  # switch layers to inference mode
with torch.inference_mode():  # efficient no-grad evaluation
    outputs = model(inputs)
```

- Use **`model.eval()`** once before inference to set layer behavior.
- Use **`torch.inference_mode()`** (preferred) or `torch.no_grad()` to disable gradients.

---

## 📎 Appendix: Autograd Version Counters

### What they are

- Every PyTorch tensor has a hidden **version counter**.
- It increments whenever the tensor is **modified in-place** (e.g., `x.add_(1)`, `x *= 2`).
- Autograd uses it to detect if a saved tensor has changed between forward and backward passes.

### Why needed

- Prevents incorrect gradients.
- If a tensor needed for backward was changed in-place, autograd raises an error:

  ```
  RuntimeError: one of the variables needed for gradient computation
  has been modified by an inplace operation
  ```

### Relation to inference modes

- **`torch.no_grad()`**: gradients disabled, **version counters still updated**.
- **`torch.inference_mode()`**: gradients disabled **and counters not updated**, for speed.

⚠️ This means:

- `inference_mode` is faster and more memory-efficient.
- But if you later try to compute gradients with tensors modified under `inference_mode`, PyTorch may not catch the inconsistency.

---

## 📊 Comparison Table

| Feature                    | `model.eval()`                          | `torch.no_grad()`                 | `torch.inference_mode()`                          |
| -------------------------- | --------------------------------------- | --------------------------------- | ------------------------------------------------- |
| Affects gradient tracking? | ❌ No                                   | ✅ Yes (disables autograd)        | ✅ Yes (disables autograd)                        |
| Affects Dropout/BatchNorm? | ✅ Yes (switches to inference behavior) | ❌ No                             | ❌ No                                             |
| Version counter updates?   | ✅ Yes                                  | ✅ Yes                            | ❌ No (skipped for efficiency)                    |
| Typical usage              | Switch layer behavior for eval          | Disable gradients (training/eval) | Faster inference-only replacement for `no_grad()` |
| Performance                | Neutral                                 | Saves some memory & compute       | Best performance & memory efficiency              |

---

✅ **Key Takeaways**:

- Accuracy is not differentiable → we train with surrogate losses (e.g., cross-entropy).
- In PyTorch inference:
  - `model.eval()` → adjusts **Dropout/BatchNorm**.
  - `torch.no_grad()` → disables gradient tracking.
  - `torch.inference_mode()` → newer, faster, skips **version counter updates**.

- Autograd version counters protect against **in-place ops breaking gradients**.

---
