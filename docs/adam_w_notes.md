# 📘 Notes on RMSProp, Adam, Bias Correction, and AdamW

---

## 🔧 RMSProp: Root Mean Square Propagation

### Purpose

- Adapts the learning rate for each parameter individually.
- Prevents large updates when gradients are large.

### Key Idea

Keep an **exponentially decaying average** of squared gradients:

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g_t^2
$$

### Update Rule

$$
\theta_t \leftarrow \theta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

- $\eta$: learning rate
- $\gamma$: decay rate (commonly 0.9)
- $\epsilon$: small constant to avoid division by zero

---

## 🔣 Gradient Per Parameter

- Every parameter (scalar, vector, or tensor) gets **one scalar gradient per element**.
- The gradient tensor has the **same shape** as the parameter tensor.
- Optimizers like RMSProp apply updates **element-wise**.

Example in PyTorch:

```python
for param in model.parameters():
    grad = param.grad  # same shape as param
```

---

## 🚀 Adam Optimizer = Momentum + RMSProp + Bias Correction

### Core Components

1. **First moment (momentum):**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

2. **Second moment (RMS):**

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

3. **Bias correction:**

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

4. **Update:**

$$
\theta_t \leftarrow \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

---

## 🧮 Why Bias Correction?

- At early steps, $m_t$ and $v_t$ are biased toward zero because $m_0 = 0$, $v_0 = 0$.
- Bias correction **rescales** them so they reflect the actual gradient magnitude.

Example at $t = 1$:

- $m_1 = 0.06$, but:

$$
\hat{m}_1 = \frac{0.06}{1 - 0.9} = 0.6
$$

This corrects the underestimation from initializing with 0.

---

## ⚠️ Why Original Adam Gets Weight Decay Wrong

- In original Adam, weight decay was added **as part of the gradient**:

$$
g_t \leftarrow g_t + \lambda \theta
$$

Problem:

- That decay is then **scaled** by Adam’s adaptive learning rate mechanism.
- So decay behaves inconsistently across parameters and time steps.

### Consequences

- You can't cleanly control regularization.
- The decay is distorted by moment estimates and adaptive scaling.
- **Generalization suffers.**

---

## ✅ AdamW: Correct Weight Decay

### Key Fix

Separate weight decay from the gradient update:

$$
\theta_t \leftarrow \theta_{t-1} - \eta \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

Benefits:

- **Uniform decay** across all parameters
- **Independent** of gradient/momentum
- Standard in modern models (e.g. BERT, Transformers)

---

## 🧠 Key Takeaways

- RMSProp adapts learning rates per parameter using squared gradients.
- Adam builds on RMSProp with momentum and bias correction.
- Bias correction helps early optimization be more accurate.
- Original Adam **misapplies weight decay** — it interacts badly with adaptivity.
- **AdamW** decouples decay, fixing the problem and improving generalization.

---
