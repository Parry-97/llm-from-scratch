"""
Batched Matrix Multiplication in PyTorch - A Comprehensive Guide
================================================================

Batched matrix multiplication allows you to perform multiple matrix multiplications
in parallel, which is essential for efficient deep learning operations.
"""

import torch
import torch.nn.functional as F


def basic_concepts():
    """
    Understanding the basics of batched matrix multiplication
    """
    print("=" * 60)
    print("1. BASIC CONCEPTS")
    print("=" * 60)

    # Regular matrix multiplication (2D tensors)
    A = torch.randn(3, 4)  # Shape: (3, 4)
    B = torch.randn(4, 5)  # Shape: (4, 5)
    C = torch.matmul(A, B)  # Shape: (3, 5)
    print(f"Regular matmul: {A.shape} @ {B.shape} = {C.shape}")

    # Batched matrix multiplication (3D tensors)
    # Think of it as a stack of matrices
    batch_size = 2
    A_batch = torch.randn(batch_size, 3, 4)  # Shape: (2, 3, 4)
    B_batch = torch.randn(batch_size, 4, 5)  # Shape: (2, 4, 5)
    C_batch = torch.matmul(A_batch, B_batch)  # Shape: (2, 3, 5)
    print(f"Batched matmul: {A_batch.shape} @ {B_batch.shape} = {C_batch.shape}")

    # What's happening under the hood:
    # For each i in range(batch_size):
    #     C_batch[i] = A_batch[i] @ B_batch[i]

    # Verify this manually
    C_manual = torch.stack([A_batch[0] @ B_batch[0], A_batch[1] @ B_batch[1]])
    print(f"Are results equal? {torch.allclose(C_batch, C_manual)}")
    print()


def torch_bmm_vs_matmul():
    """
    Comparing torch.bmm() and torch.matmul() for batched operations
    """
    print("=" * 60)
    print("2. TORCH.BMM vs TORCH.MATMUL")
    print("=" * 60)

    batch_size = 4
    A = torch.randn(batch_size, 3, 5)
    B = torch.randn(batch_size, 5, 2)

    # torch.bmm() - specifically for batched matrix multiplication
    # Requires both inputs to be 3D tensors with same batch size
    result_bmm = torch.bmm(A, B)
    print(f"torch.bmm: {A.shape} @ {B.shape} = {result_bmm.shape}")

    # torch.matmul() - more general, handles various dimensions
    result_matmul = torch.matmul(A, B)
    print(f"torch.matmul: {A.shape} @ {B.shape} = {result_matmul.shape}")

    print(f"Results equal? {torch.allclose(result_bmm, result_matmul)}")
    print()


def broadcasting_in_batched_matmul():
    """
    Understanding broadcasting rules in batched matrix multiplication
    """
    print("=" * 60)
    print("3. BROADCASTING IN BATCHED MATMUL")
    print("=" * 60)

    # Case 1: Batch dimension broadcasting
    A = torch.randn(10, 3, 4)  # 10 matrices of shape (3, 4)
    B = torch.randn(4, 5)  # Single matrix (4, 5) - will be broadcast
    C = torch.matmul(A, B)  # B is broadcast to all batch elements
    print(f"Broadcasting single matrix: {A.shape} @ {B.shape} = {C.shape}")

    # Case 2: More complex broadcasting
    A = torch.randn(1, 3, 4)  # Single batch, will be broadcast
    B = torch.randn(5, 4, 2)  # 5 batches
    C = torch.matmul(A, B)  # A is broadcast to match B's batch size
    print(f"Broadcasting batch dimension: {A.shape} @ {B.shape} = {C.shape}")

    # Case 3: Full broadcasting with multiple dimensions
    A = torch.randn(2, 1, 3, 4)  # Shape: (2, 1, 3, 4)
    B = torch.randn(1, 5, 4, 6)  # Shape: (1, 5, 4, 6)
    C = torch.matmul(A, B)  # Shape: (2, 5, 3, 6)
    print(f"Full broadcasting: {A.shape} @ {B.shape} = {C.shape}")
    print()


def practical_examples():
    """
    Practical examples of batched matrix multiplication in deep learning
    """
    print("=" * 60)
    print("4. PRACTICAL DEEP LEARNING EXAMPLES")
    print("=" * 60)

    # Example 1: Attention mechanism (simplified)
    batch_size = 2
    seq_length = 5
    d_model = 8

    # Query, Key, Value matrices
    Q = torch.randn(batch_size, seq_length, d_model)
    K = torch.randn(batch_size, seq_length, d_model)
    V = torch.randn(batch_size, seq_length, d_model)

    # Attention scores: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq, seq)
    print(f"Attention scores: {Q.shape} @ {K.shape} = {scores.shape}")

    # Apply softmax and multiply with V
    attention_weights = F.softmax(scores / (d_model**0.5), dim=-1)
    attention_output = torch.matmul(attention_weights, V)
    print(
        f"Attention output: {attention_weights.shape} @ {V.shape} = {attention_output.shape}"
    )
    print()

    # Example 2: Batch linear transformation
    batch_size = 4
    input_features = 10
    output_features = 5
    num_samples = 32

    # Input data: (batch, num_samples, input_features)
    X = torch.randn(batch_size, num_samples, input_features)

    # Weight matrix: (batch, input_features, output_features)
    # Different weights for each batch element
    W = torch.randn(batch_size, input_features, output_features)

    # Batched linear transformation
    Y = torch.matmul(X, W)  # (batch, num_samples, output_features)
    print(f"Batched linear: {X.shape} @ {W.shape} = {Y.shape}")
    print()


def einsum_for_complex_operations():
    """
    Using einsum for more complex batched operations
    """
    print("=" * 60)
    print("5. EINSUM FOR COMPLEX BATCHED OPERATIONS")
    print("=" * 60)

    # Example: Multi-head attention-like operation
    batch = 2
    heads = 4
    seq_len = 6
    d_k = 8

    Q = torch.randn(batch, heads, seq_len, d_k)
    K = torch.randn(batch, heads, seq_len, d_k)

    # Using einsum for batched matrix multiplication with multiple dimensions
    # 'bhqd,bhkd->bhqk' means:
    # b=batch, h=heads, q=query_seq, k=key_seq, d=dimension
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    print(f"Einsum multi-head attention: {Q.shape} @ {K.shape} = {scores.shape}")

    # Equivalent using matmul
    scores_matmul = torch.matmul(Q, K.transpose(-2, -1))
    print(f"Matmul equivalent: {scores_matmul.shape}")
    print(f"Results equal? {torch.allclose(scores, scores_matmul)}")
    print()


def performance_tips():
    """
    Performance tips for batched matrix multiplication
    """
    print("=" * 60)
    print("6. PERFORMANCE TIPS")
    print("=" * 60)

    print("Key performance considerations:")
    print("1. Use contiguous tensors when possible:")
    print("   tensor = tensor.contiguous()")
    print()
    print("2. Batch operations are much faster than loops:")
    print("   GOOD: torch.bmm(A_batch, B_batch)")
    print("   BAD:  for i in range(batch): C[i] = A[i] @ B[i]")
    print()
    print("3. torch.matmul() vs torch.bmm():")
    print("   - bmm() is slightly faster for 3D tensors")
    print("   - matmul() is more flexible with broadcasting")
    print()
    print("4. Memory layout matters:")
    print("   - Keep batch dimension first for better memory access")
    print("   - Use transpose(-2, -1) instead of permute when possible")
    print()
    print("5. For very large batches, consider:")
    print("   - Gradient checkpointing")
    print("   - Mixed precision training (fp16)")
    print("   - Splitting into smaller chunks if memory is limited")
    print()


def common_shapes_reference():
    """
    Common shape patterns in deep learning
    """
    print("=" * 60)
    print("7. COMMON SHAPE PATTERNS IN DEEP LEARNING")
    print("=" * 60)

    print("Transformer self-attention:")
    print("  Q, K, V: (batch, seq_len, d_model)")
    print("  Scores: (batch, seq_len, seq_len)")
    print()
    print("CNN batch processing:")
    print("  Input: (batch, channels, height, width)")
    print("  After flatten: (batch, channels * height * width)")
    print("  Linear: (batch, features_in) @ (features_in, features_out)")
    print()
    print("RNN/LSTM:")
    print("  Input: (batch, seq_len, input_size)")
    print("  Hidden: (batch, hidden_size)")
    print("  Weight: (input_size + hidden_size, hidden_size)")
    print()
    print("Multi-head attention:")
    print("  Input: (batch, seq_len, d_model)")
    print("  After reshape: (batch, heads, seq_len, d_k)")
    print("  where d_k = d_model // heads")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BATCHED MATRIX MULTIPLICATION IN PYTORCH")
    print("=" * 60 + "\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run all examples
    basic_concepts()
    torch_bmm_vs_matmul()
    broadcasting_in_batched_matmul()
    practical_examples()
    einsum_for_complex_operations()
    performance_tips()
    common_shapes_reference()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Batched matrix multiplication in PyTorch:
1. Performs multiple matrix multiplications in parallel
2. First dimension is typically the batch dimension
3. torch.matmul() handles broadcasting automatically
4. torch.bmm() is specific for 3D batched operations
5. Essential for efficient deep learning computations
6. GPU acceleration makes batched ops much faster than loops

Key insight: Instead of iterating through batch elements,
batched operations process all elements simultaneously!
    """)
