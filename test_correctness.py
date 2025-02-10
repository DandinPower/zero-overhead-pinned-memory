# test_correctness.py
import pytest
import torch
import random
import math

from zero_overhead_pinned_memory import memalign_pin_memory, cuda_pin_memory

@pytest.mark.parametrize("shape", [
    (10,),
    (128, 128),
    (64, 64, 64),
])
@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.bfloat16
])
def test_memalign_pin_memory(shape, dtype):
    """Test that memalign_pin_memory produces a pinned CPU tensor
    with the same shape & values as the original."""
    
    # Seed for reproducibility (optional)
    random.seed(42)
    torch.manual_seed(42)
    
    # Create a random CPU tensor
    original = torch.randn(shape, dtype=dtype)
    
    # Use our pinned memory function
    pinned = memalign_pin_memory(original)
    
    # Check that the returned tensor is pinned and on CPU
    assert pinned.device.type == 'cpu', "Tensor is not on CPU."
    assert pinned.is_pinned(), "Returned tensor is not marked as pinned."
    
    # Check shape and dtype
    assert pinned.shape == original.shape, "Shape mismatch."
    assert pinned.dtype == original.dtype, "Dtype mismatch."
    
    # Check that data is preserved
    # (use allclose for floating dtypes, or equal for integer dtypes)
    if pinned.is_floating_point():
        assert torch.allclose(pinned, original), "Pinned tensor values differ from original (float)."
    else:
        assert torch.equal(pinned, original), "Pinned tensor values differ from original (int)."
    
    # (Optional) If CUDA is available, try transferring to GPU
    if torch.cuda.is_available():
        try:
            gpu_tensor = pinned.to('cuda')
            # Just a functional check: shapes & data should match
            if gpu_tensor.is_floating_point():
                assert torch.allclose(gpu_tensor.cpu(), original), (
                    "GPU copy does not match original values."
                )
            else:
                assert torch.equal(gpu_tensor.cpu(), original), (
                    "GPU copy does not match original values."
                )
        except Exception as e:
            pytest.fail(f"Transferring pinned tensor to GPU raised an exception: {e}")

@pytest.mark.parametrize("shape", [
    (10,),
    (128, 128),
    (64, 64, 64),
])
@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.bfloat16
])
def test_cuda_pin_memory(shape, dtype):
    """Test that cuda_pin_memory produces a pinned CPU tensor
    with the same shape & values as the original."""
    
    # Seed for reproducibility (optional)
    random.seed(42)
    torch.manual_seed(42)
    
    # Create a random CPU tensor
    original = torch.randn(shape, dtype=dtype)
    
    # Use our pinned memory function
    pinned = cuda_pin_memory(original)
    
    # Check that the returned tensor is pinned and on CPU
    assert pinned.device.type == 'cpu', "Tensor is not on CPU."
    assert pinned.is_pinned(), "Returned tensor is not marked as pinned."
    
    # Check shape and dtype
    assert pinned.shape == original.shape, "Shape mismatch."
    assert pinned.dtype == original.dtype, "Dtype mismatch."
    
    # Check that data is preserved
    # (use allclose for floating dtypes, or equal for integer dtypes)
    if pinned.is_floating_point():
        assert torch.allclose(pinned, original), "Pinned tensor values differ from original (float)."
    else:
        assert torch.equal(pinned, original), "Pinned tensor values differ from original (int)."
    
    # (Optional) If CUDA is available, try transferring to GPU
    if torch.cuda.is_available():
        try:
            gpu_tensor = pinned.to('cuda')
            # Just a functional check: shapes & data should match
            if gpu_tensor.is_floating_point():
                assert torch.allclose(gpu_tensor.cpu(), original), (
                    "GPU copy does not match original values."
                )
            else:
                assert torch.equal(gpu_tensor.cpu(), original), (
                    "GPU copy does not match original values."
                )
        except Exception as e:
            pytest.fail(f"Transferring pinned tensor to GPU raised an exception: {e}")
