# Zero Overhead Pinned Memory

A PyTorch extension that provides alternative methods to create pinned CPU memory tensors—bypassing the extra memory overhead introduced by PyTorch’s power-of-two allocation strategy. This extension now supports **two distinct manual allocation methods** to give you flexibility based on your requirements.

## Motivation

In PyTorch, When you set `pin_memory=True` or `.pin_memory()` for tensor, the  call is dispatched to PyTorch’s C++ backend, where the allocation size is first rounded up to a power of two (using a function similar to `c10::llvm::PowerOf2Ceil`) before calling CUDA’s memory allocation routines (like `cudaHostAlloc`). For further discussion about this implementation [Discussion on GitHub Issue #95823](https://github.com/pytorch/pytorch/issues/95823). This design decision was made to:

- **Simplify Caching and Reduce Fragmentation:** Uniform block sizes simplify the CUDA caching allocator’s bookkeeping, allowing it to better reuse memory and reduce fragmentation.

While these benefits can be significant, a downside is that if the requested size is not a power of two, PyTorch ends up allocating considerably more pinned memory than necessary—especially noticeable with large tensors. Moreover, in scenarios such as with DeepSpeed, we do not routinely create and release page-locked memory; instead, we retain the allocated space until the process ends, making this kind of fragmentation prevention less effective.

This repository offers two alternative approaches that bypass this power-of-two allocation strategy:

- **Method 1: `memalign_pin_memory`**  
  Manually allocates aligned memory using `posix_memalign` (with 4096-byte alignment) and then registers that memory with CUDA via `cudaHostRegister`. This method avoids the power-of-two overhead with only minimal additional metadata overhead.

- **Method 2: `cuda_pin_memory`**  
  Allocates pinned CPU memory directly using CUDA’s `cudaHostAlloc` API with the `cudaHostAllocMapped` flag, offering another alternative to the default PyTorch allocation.

Both methods copy the data from the source tensor into the manually allocated pinned memory, ensuring that you end up with a tensor of the same shape and options as the original.

## Features

1. **Pinned Memory Tensor Creation**  
   The extension provides two CUDA extension functions:
   - **`memalign_pin_memory`**  
     Allocates pinned CPU memory using `posix_memalign` and then registers the memory with CUDA via `cudaHostRegister`.
   - **`cuda_pin_memory`**  
     Allocates pinned CPU memory using CUDA’s `cudaHostAlloc` API.

   Both functions copy data from your original CPU tensor into the pinned memory, thereby bypassing PyTorch’s power-of-two allocation strategy.

2. **Memory Usage Comparison**  
   Scripts are provided for measuring and comparing memory usage between the standard PyTorch pinned allocator and these manual approaches.

3. **Performance Benchmark**  
   Code is available to compare CPU-to-GPU transfer speeds and allocation overhead among:
   - PyTorch's built-in pinned memory (`pin_memory=True`)
   - The manual approach using `memalign_pin_memory`
   - The manual approach using `cuda_pin_memory`

4. **Correctness Tests**  
   A suite of tests ensures that the pinned memory is allocated, registered, and freed correctly, and that data integrity is maintained during GPU transfers.

5. **DeepSpeed Patch**  
   Optionally, you can patch DeepSpeed to use these manual allocation methods (currently support `memalign_pin_memory`) for its pinned memory allocations.

## Installation

1. **Prerequisites:**  
   Ensure you have a compatible **NVIDIA driver**, **CUDA toolkit** (with `nvcc`), and **PyTorch** installed and configured with GPU support.

2. **Install development dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the extension from source:**  
   ```bash
   pip install .
   ```

## Usage

After installation, the extension provides two functions that you can import from the module (e.g., `zero_overhead_pinned_memory`):

- **`memalign_pin_memory`**  
  ```python
  import torch
  from zero_overhead_pinned_memory import memalign_pin_memory

  cpu_tensor = torch.randn(100, 100)
  pinned_tensor = memalign_pin_memory(cpu_tensor)
  ```
  This function allocates pinned memory via `posix_memalign` and registers it with CUDA.

- **`cuda_pin_memory`**  
  ```python
  import torch
  from zero_overhead_pinned_memory import cuda_pin_memory

  cpu_tensor = torch.randn(100, 100)
  pinned_tensor = cuda_pin_memory(cpu_tensor)
  ```
  This function allocates pinned memory using `cudaHostAlloc`.

### 1. Running Tests

Use `pytest` to run the correctness tests:
```bash
pytest test_correctness.py
```
These tests verify that the pinned memory is correctly allocated, registered, and that its data remains consistent when transferred to the GPU.

### 2. Memory Usage Comparison

To compare the actual host memory usage between power-of-two and non-power-of-two allocations, run:
```bash
python memory_usage.py
```
This script measures memory usage in a separate process for accurate results.

### 3. Performance Benchmark

To compare allocation and transfer speeds between:
1. PyTorch's built-in pinned memory (`pin_memory=True`),
2. The manual pinned memory using `memalign_pin_memory`,
3. The manual pinned memory using `cuda_pin_memory`,

run:
```bash
python performance_benchmark.py
```
The benchmark prints average allocation times (CPU creation + pinning) and average GPU transfer times for all approaches.

### 4. DeepSpeed Patch

If you want DeepSpeed to use these methods for its pinned memory allocations, apply the monkey patch:
```python
from zero_overhead_pinned_memory import patch_deepspeed_zero_overhead_pinned_memory

patch_deepspeed_zero_overhead_pinned_memory()  
# This replaces deepspeed.accelerator.cuda_accelerator.CUDA_Accelerator.pin_memory
# with our zero-overhead pinned memory method.
```

## Reference

For more information on CUDA host memory registration, please refer to the official NVIDIA documentation:

- [CUDA Runtime API: cudaHostRegister](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g81fd4101862bbefdb42a62d60e515eea)
- [CUDA Runtime API: cudaHostAlloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902)

---

**Enjoy faster and more memory-efficient pinned allocations!**  
If you find any bugs or have suggestions, please open an issue or submit a pull request.
