# Adam vs. AdamW: Understanding the Optimizers

## Introduction: What is an Optimizer?

In deep learning, an **optimizer** is an algorithm that modifies the attributes of the neural network, such as its weights and learning rate, to minimize the loss function. The loss function measures the difference between the model's prediction and the actual target. The goal is to make this difference as small as possible.

The most basic optimizer is **Stochastic Gradient Descent (SGD)**, which updates the weights in the direction opposite to the gradient of the loss function. However, more advanced optimizers have been developed to improve the speed and stability of the training process.

## Weighted Averages in Optimization: Exponential Moving Averages

Before diving into sophisticated optimizers, it's crucial to understand **Exponential Moving Averages (EMAs)**, which form the mathematical backbone of modern optimization algorithms.

### The Mathematics of EMA

An EMA smooths a sequence of values over time. For a sequence of values $x_1, x_2, ..., x_t$, the EMA at time $t$ is computed as:

$$\text{EMA}_t = \beta \cdot \text{EMA}_{t-1} + (1 - \beta) \cdot x_t$$

Where:

- $\beta$ (beta) is the decay factor, typically between 0 and 1
- $\text{EMA}_0$ is initialized to 0 or sometimes to $x_1$ for faster warmup

When unrolled, this becomes a weighted sum where older values have exponentially decaying influence:

$$\text{EMA}_t = (1 - \beta) \sum_{i=1}^{t} \beta^{t-i} \cdot x_i$$

### Bias Correction

During early iterations, EMAs are biased toward zero (if initialized to zero). To correct this:

$$\widehat{\text{EMA}}_t = \frac{\text{EMA}_t}{1 - \beta^t}$$

This correction ensures unbiased estimates, especially crucial in the first few iterations.

### Practical Interpretation

- **Effective window size**: Approximately $\frac{1}{1-\beta}$ timesteps. For $\beta=0.9$, this means roughly the last 10 values significantly influence the average.
- **Half-life**: The number of steps for a value's influence to halve is $\frac{\log(0.5)}{\log(\beta)}$.
- **Smoothing effect**: EMAs act as low-pass filters, removing high-frequency noise from signals.

### Role in Optimization

EMAs serve three critical functions in modern optimizers:

1. **Gradient smoothing (momentum)**: EMA of gradients reduces noise in stochastic gradient estimates
2. **Adaptive scaling**: EMA of squared gradients enables per-parameter learning rate adaptation (RMSProp, Adam)
3. **Parameter averaging**: EMA of model parameters during training can improve generalization at inference

This smoothing mechanism underlies both momentum-based methods and adaptive learning rate algorithms that we'll explore next.

## Momentum and Its Regularization Effects

**Momentum** is one of the most important innovations in optimization, transforming how neural networks navigate complex loss landscapes.

### Classical Momentum

The momentum update rule maintains a velocity vector $v$ that accumulates gradients over time:

$$v_t = \beta_1 \cdot v_{t-1} + (1 - \beta_1) \cdot g_t$$
$$\theta_{t+1} = \theta_t - \alpha \cdot v_t$$

Where:

- $g_t$ is the gradient at time $t$
- $\beta_1$ is the momentum coefficient (typically 0.9)
- $\alpha$ is the learning rate
- $\theta$ represents the model parameters

### Physical Intuition

Imagine a ball rolling down a hill:

- **Acceleration in consistent directions**: The ball gains speed when the slope remains downward
- **Damping oscillations**: When encountering ravines, momentum helps the ball roll through rather than oscillating back and forth
- **Escaping local minima**: Built-up velocity can carry the ball over small bumps

### Implicit Regularization Effects

Momentum provides regularization through several mechanisms:

1. **Noise averaging**: By smoothing noisy stochastic gradients, momentum reduces variance and implicitly biases optimization toward flatter minima, which often generalize better.

2. **Trajectory smoothing**: The accumulated velocity creates smoother optimization paths, avoiding erratic jumps that might lead to sharp, overfitted solutions.

3. **Effective batch size increase**: Momentum effectively increases the number of gradients influencing each update, similar to using larger batch sizes.

### Parameter EMA for Regularization

A related but distinct technique is maintaining an EMA of the parameters themselves:

$$\theta_{\text{ema},t} = \gamma \cdot \theta_{\text{ema},t-1} + (1 - \gamma) \cdot \theta_t$$

Using $\theta_{\text{ema}}$ for evaluation instead of $\theta_t$:

- Approximates an ensemble of models from the optimization trajectory
- Often improves generalization performance
- Reduces sensitivity to the exact stopping point

### Key Distinctions

- **Momentum on gradients**: Affects the optimization dynamics during training
- **EMA on parameters**: A post-processing technique for better inference
- **Weight decay**: An explicit penalty on parameter magnitudes (covered later with AdamW)

While momentum smooths the optimization direction, we still need adaptive scaling for parameters with different gradient magnitudes—which brings us to RMSProp.

## RMSProp: Root Mean Square Propagation

**RMSProp** addresses a critical limitation in vanilla gradient descent: different parameters often require different learning rates due to varying gradient magnitudes.

### The Problem with Fixed Learning Rates

Consider training a neural network where:

- Some weights receive consistently large gradients (e.g., frequently activated neurons)
- Others receive small, sparse gradients (e.g., rarely activated features)

Using the same learning rate for all parameters leads to:

- Slow learning for parameters with small gradients
- Instability for parameters with large gradients

### The RMSProp Solution

RMSProp maintains a per-parameter EMA of squared gradients to adaptively scale learning rates:

$$s_t = \rho \cdot s_{t-1} + (1 - \rho) \cdot g_t^2$$
$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{g_t}{\sqrt{s_t} + \epsilon}$$

Where:

- $s_t$ is the EMA of squared gradients (second moment)
- $\rho$ is the decay rate (typically 0.9 or 0.99)
- $\epsilon$ is a small constant for numerical stability (typically $10^{-8}$)
- All operations are element-wise

### Algorithm Pseudocode

```python
def rmsprop(parameters, gradients, state, alpha=0.001, rho=0.9, epsilon=1e-8):
    if state.s is None:
        state.s = zeros_like(parameters)

    # Update second moment estimate
    state.s = rho * state.s + (1 - rho) * gradients**2

    # Update parameters with adaptive learning rate
    parameters = parameters - alpha * gradients / (sqrt(state.s) + epsilon)

    return parameters, state
```

### Key Insights

1. **Adaptive scaling**: Parameters with large gradients get smaller effective learning rates (divided by larger $\sqrt{s_t}$)
2. **Gradient normalization**: The division by $\sqrt{s_t}$ normalizes gradients to similar magnitudes
3. **EMA advantage**: Unlike AdaGrad (which accumulates all squared gradients), RMSProp's EMA allows adaptation to changing gradient statistics

### Hyperparameter Guidelines

- **$\rho$ (rho)**: Controls the timescale of adaptation
  - Higher values (0.99): Slower adaptation, more stable
  - Lower values (0.9): Faster adaptation, more responsive
- **$\alpha$ (alpha)**: Base learning rate, typically $10^{-4}$ to $10^{-3}$
- **$\epsilon$ (epsilon)**: Prevents division by zero, but also acts as a regularizer
  - Larger values: More conservative updates
  - Smaller values: More aggressive scaling

### Practical Considerations

- **Initialization**: $s_0$ is typically initialized to zero
- **Sensitivity**: Performance can be sensitive to $\epsilon$, especially for sparse gradients
- **Non-convex optimization**: Works well for neural networks with varying gradient scales across layers

Combining RMSProp's adaptive scaling with momentum's directional smoothing leads us to one of the most successful optimizers: Adam.

## From Momentum and RMSProp to Adam: Putting It All Together

The **Adam optimizer** elegantly combines the benefits of momentum and RMSProp, creating a robust algorithm that has become the default choice for training deep neural networks.

### The Synthesis

Adam maintains two EMAs per parameter:

1. **First moment** ($m_t$): EMA of gradients (like momentum)
2. **Second moment** ($v_t$): EMA of squared gradients (like RMSProp)

### The Adam Update Rules

The complete Adam algorithm:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Why Bias Correction Matters

Without bias correction, early iterations would be heavily biased toward zero (since $m_0 = v_0 = 0$). The correction factors $\frac{1}{1-\beta^t}$ ensure:

- Unbiased gradient estimates from the start
- Proper scaling especially in the critical early training phase
- Convergence guarantees in the convex case

### Default Hyperparameters and Their Meaning

- **$\beta_1 = 0.9$**: Momentum coefficient
  - Effective gradient window: ~10 steps
  - Provides substantial smoothing without excessive lag
- **$\beta_2 = 0.999$**: Second moment coefficient
  - Effective squared gradient window: ~1000 steps
  - Very slow adaptation of per-parameter scales
- **$\epsilon = 10^{-8}$**: Numerical stability constant
- **$\alpha = 10^{-3}$**: Base learning rate (often requires tuning)

### The Power of Combination

Adam's effectiveness comes from the synergy of its components:

1. **Momentum** ($m_t$) provides:
   - Smooth optimization trajectories
   - Faster convergence in consistent gradient directions
   - Escape from saddle points and shallow local minima

2. **Adaptive scaling** ($v_t$) provides:
   - Automatic per-parameter learning rate tuning
   - Robustness to gradient scale differences
   - Natural handling of sparse gradients

3. **Bias correction** provides:
   - Reliable behavior from initialization
   - Theoretical convergence guarantees
   - Predictable early training dynamics

### Connection to What Follows

While Adam successfully combines momentum and adaptive learning rates, it inherits a subtle issue with weight decay regularization from its adaptive components. The coupling between the adaptive learning rate and weight decay makes regularization less predictable than intended. This limitation led to the development of AdamW, which we'll explore next, showing how decoupling weight decay from the adaptive mechanisms creates an even more effective optimizer.

## Adam: Adaptive Moment Estimation

For a long time, **Adam** has been the default, go-to optimizer for training deep neural networks. Its name comes from "Adaptive Moment Estimation," and it combines two powerful concepts to achieve its performance:

1. **Momentum**: This technique helps accelerate SGD in the relevant direction and dampens oscillations. It does this by accumulating a moving average of past gradients, much like a ball rolling down a hill gains momentum. This is the **first moment** (the mean of the gradients).

2. **RMSProp (Root Mean Square Propagation)**: This method adapts the learning rate for each parameter individually. It uses a moving average of the _squared_ gradients to scale the learning rate. Parameters with larger gradients receive smaller updates, and parameters with smaller gradients receive larger updates. This is the **second moment** (the uncentered variance of the gradients).

### The Problem: Weight Decay in Adam

**Weight decay** is a common regularization technique used to prevent overfitting. It works by adding a penalty to the loss function for large weights. In standard SGD, this is equivalent to subtracting a small fraction of the weight at each update step.

However, in Adam, this form of weight decay (often implemented as L2 regularization) becomes coupled with the adaptive learning rate mechanism. The amount of decay applied to a weight is scaled by the second-moment estimate (`v`). This means that weights with large historical gradients (and thus a large `v`) receive smaller effective weight decay. This coupling makes weight decay less effective and less predictable than in SGD.

## AdamW: Decoupled Weight Decay

**AdamW** (Adam with Weight Decay) was proposed to fix this issue. It decouples the weight decay from the gradient update step, making regularization much more effective.

### The Fix

Instead of adding the L2 regularization penalty to the loss function, AdamW applies the weight decay directly to the weights _after_ the main Adam optimization step.

The conceptual process is:

1. Calculate the gradient and perform the standard Adam update using momentum and the adaptive learning rate.
2. After the weights have been updated, apply the weight decay directly:
   `weight = weight - learning_rate * weight_decay * weight`

### Why is AdamW Better?

1. **More Effective Regularization**: The weight decay is now a fixed factor, just as intended. It is no longer influenced by the gradient history, leading to better generalization and preventing the model from overfitting as effectively as it should.
2. **Improved Model Performance**: By decoupling these two components, AdamW often converges faster and finds better solutions than Adam, especially in complex models like Transformers.
3. **Better Hyperparameter Tuning**: The learning rate and weight decay hyperparameters can be tuned more independently, simplifying the process of finding a good model configuration.

## Adam vs. AdamW: Key Differences

| Feature | Adam | AdamW |
|:--------|:-----|:------|
| **Weight Decay** | Coupled with gradient updates<br/>(implemented as L2 regularization). | Decoupled from the gradient<br/>update step. |
| **Mechanism** | Adds the L2 penalty to the loss function.<br/>The decay's effect is scaled by the<br/>adaptive learning rates. | Applies weight decay directly to the<br/>weights after the optimizer has<br/>updated them. |
| **Effectiveness** | The decay is less effective for weights<br/>with large historical gradients. | The decay is more predictable and<br/>stable, leading to better<br/>generalization. |
| **Recommendation** | Was the standard for many years. | Is now the recommended default for<br/>most deep learning models,<br/>especially Transformers. |

> [!NOTE]
> You can check out the related videos from DeepLearning.AI

## Conclusion

For training modern neural networks like the Large Language Models this project is focused on, **AdamW is the preferred optimizer**. It addresses a fundamental flaw in how Adam handles weight decay, resulting in more stable training, better model performance, and more effective regularization. While Adam is still a powerful algorithm, AdamW is its direct successor and the superior choice in most deep learning applications today.
