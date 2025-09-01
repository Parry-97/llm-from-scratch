# PyTorch Batched Matrix Multiplication & Broadcasting Guide

## Table of Contents
1. [Basic Concepts](#basic-concepts)
2. [Batched Matrix Multiplication](#batched-matrix-multiplication)
3. [Broadcasting Rules](#broadcasting-rules)
4. [Common Scenarios](#common-scenarios)
5. [Code Examples](#code-examples)

---

## Basic Concepts

### What is Batched Matrix Multiplication?
Batched matrix multiplication allows you to perform multiple matrix multiplications in parallel. Instead of looping through matrices, PyTorch processes them all at once, leveraging GPU parallelization.

### Key Functions
- `torch.matmul()` - General matrix multiplication with broadcasting
- `torch.bmm()` - Batch matrix multiplication (strict, no broadcasting)
- `@` operator - Syntactic sugar for `torch.matmul()`

---

## Batched Matrix Multiplication

### Standard Case: Same Batch Size
When both tensors have the same batch dimension:
```
A: (batch_size, m, n)  # batch_size matrices of shape (m, n)
B: (batch_size, n, p)  # batch_size matrices of shape (n, p)
Result: (batch_size, m, p)  # batch_size matrices of shape (m, p)
```

Each matrix in batch `i` of A is multiplied with the corresponding matrix in batch `i` of B.

### Example:
- A has shape `(10, 3, 4)` → 10 matrices, each 3×4
- B has shape `(10, 4, 5)` → 10 matrices, each 4×5
- Result has shape `(10, 3, 5)` → 10 matrices, each 3×5

---

## Broadcasting Rules

### PyTorch Broadcasting Principles

1. **Dimension Alignment**: Dimensions are compared from right to left
2. **Compatible Dimensions**: Two dimensions are compatible when:
   - They are equal, OR
   - One of them is 1, OR
   - One of them doesn't exist

3. **Broadcasting Process**:
   - If one tensor has fewer dimensions, prepend 1s to its shape
   - Dimensions of size 1 are stretched to match the other tensor

### Broadcasting in Matrix Multiplication

#### Case 1: Missing Batch Dimension
```
A: (batch_size, m, n)
B: (n, p)              # No batch dimension
Result: (batch_size, m, p)
```
B is broadcast to all batches in A.

#### Case 2: Batch Size of 1
```
A: (3, m, n)          # Batch size 3
B: (1, n, p)          # Batch size 1
Result: (3, m, p)
```
B's single batch is broadcast to match A's 3 batches.

#### Case 3: Multiple Broadcast Dimensions
```
A: (2, 1, m, n)       # 2 groups, 1 batch each
B: (1, 3, n, p)       # 1 group, 3 batches
Result: (2, 3, m, p)  # 2 groups, 3 batches each
```

---

## Common Scenarios

### ✅ Compatible Broadcasts

| Tensor A Shape | Tensor B Shape | Result Shape | Explanation |
|---------------|---------------|--------------|-------------|
| (3, 4, 5) | (5, 6) | (3, 4, 6) | B broadcasts to all 3 batches |
| (3, 4, 5) | (1, 5, 6) | (3, 4, 6) | B's batch=1 broadcasts to 3 |
| (1, 4, 5) | (3, 5, 6) | (3, 4, 6) | A's batch=1 broadcasts to 3 |
| (2, 1, 4, 5) | (1, 3, 5, 6) | (2, 3, 4, 6) | Both broadcast |

### ❌ Incompatible Broadcasts

| Tensor A Shape | Tensor B Shape | Why It Fails |
|---------------|---------------|--------------|
| (3, 4, 5) | (5, 5, 6) | Batch sizes 3 and 5 incompatible |
| (2, 4, 5) | (3, 5, 6) | Batch sizes 2 and 3 incompatible |
| (2, 3, 4, 5) | (3, 2, 5, 6) | Dimensions (2,3) and (3,2) incompatible |

---

## Code Examples

### Example 1: Basic Batched Multiplication
```python
import torch

# Same batch size
A = torch.randn(10, 3, 4)  # 10 batches of 3x4 matrices
B = torch.randn(10, 4, 5)  # 10 batches of 4x5 matrices
C = torch.matmul(A, B)      # Result: (10, 3, 5)

# Each C[i] = A[i] @ B[i]
```

### Example 2: Broadcasting Single Matrix
```python
# Broadcasting a single matrix to multiple batches
batch_A = torch.randn(8, 3, 4)  # 8 different 3x4 matrices
single_B = torch.randn(4, 5)    # One 4x5 matrix

# single_B is applied to all 8 matrices in batch_A
result = torch.matmul(batch_A, single_B)  # Shape: (8, 3, 5)
```

### Example 3: Broadcasting with Size-1 Batch
```python
# Different batch sizes with broadcasting
A = torch.randn(5, 3, 4)  # 5 batches
B = torch.randn(1, 4, 6)  # 1 batch (will broadcast)

result = torch.matmul(A, B)  # Shape: (5, 3, 6)
# B[0] is used for all 5 multiplications
```

### Example 4: Complex Broadcasting
```python
# Multiple broadcast dimensions
A = torch.randn(2, 1, 3, 4)  # 2 groups, 1 batch each
B = torch.randn(1, 5, 4, 6)  # 1 group, 5 batches

result = torch.matmul(A, B)  # Shape: (2, 5, 3, 6)
# Creates 2 groups × 5 batches = 10 total matrix multiplications
```

### Example 5: Practical Use Case - Attention Mechanism
```python
# Simplified attention mechanism
batch_size = 32
seq_len = 100
d_model = 512

Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

# Compute attention scores
scores = torch.matmul(Q, K.transpose(-2, -1))  # (32, 100, 100)
scores = scores / (d_model ** 0.5)

# Apply softmax and compute weighted values
attn_weights = torch.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)  # (32, 100, 512)
```

---

## Key Takeaways

1. **Efficiency**: Batched operations are much faster than loops, especially on GPUs
2. **Broadcasting**: Allows flexible tensor shapes but follow the rules
3. **Memory**: Broadcasting doesn't copy data, it's a view operation
4. **Debugging**: Always check tensor shapes with `.shape` when debugging
5. **torch.bmm vs torch.matmul**: Use `bmm` when you want strict batch matching without broadcasting

---

## Common Pitfalls & Solutions

### Pitfall 1: Unexpected Broadcasting
```python
# Intended: Element-wise multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 3)
# Wrong: This does matrix multiplication!
result = A @ B  # Shape: (3, 3)

# Correct for element-wise:
result = A * B.T  # Shape: (3, 4)
```

### Pitfall 2: Incompatible Batch Sizes
```python
# This will fail
A = torch.randn(3, 4, 5)
B = torch.randn(5, 5, 6)
# RuntimeError: batch dimensions don't match

# Solution: Ensure compatible batch dimensions
B_compatible = B[0:1]  # Take first batch and make it broadcastable
result = torch.matmul(A, B_compatible)  # Works!
```

### Pitfall 3: Memory Issues with Large Broadcasts
```python
# Be careful with large broadcasts
A = torch.randn(1000, 1, 512, 768)
B = torch.randn(1, 1000, 768, 256)
# Result would be (1000, 1000, 512, 256) - potentially huge!

# Consider processing in chunks if memory is limited
```
