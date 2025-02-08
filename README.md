# Zero Overhead Pinned Memory

A PyTorch extension that provides an alternative way to create pinned CPU memory tensors, potentially avoiding large internal alignment overheads seen when using PyTorch's built-in `pin_memory`.

## Motivation

In PyTorch, if you allocate a tensor with `pin_memory=True` (or call `tensor.pin_memory()`), under the hood PyTorch uses `cudaHostAlloc`. When the size you request is **not** a power of two, this can lead to internal alignment to the next power-of-two size, potentially causing large overhead for large tensors. As tensor sizes grow, the jump to the next power of two becomes significant, meaning additional pinned memory is allocated that you don't strictly need.

This repository provides an alternative approach that manually allocates host memory using `posix_memalign` and then calls `cudaHostRegister` and `cudaHostUnregister` to pin/unpin the memory. This way, we avoid the large power-of-two alignment overhead. There is a small overhead for metadata, but it is significantly smaller than power-of-two padding for large tensor sizes.

## Features

1. **Pinned Memory Tensor Creation**  
   A CUDA extension that creates a pinned CPU tensor by manually allocating aligned memory and then copying data from your original tensor.

2. **Memory Usage Comparison**  
   Scripts for measuring and comparing memory usage between the standard PyTorch pinned allocator (`cudaHostAlloc`) and this manual approach.

3. **Performance Benchmark**  
   Code to compare CPU-to-GPU transfer speeds and creation overhead.

4. **Correctness Test**  
   Tests to ensure the pinned memory is allocated, registered, unregistered, and transferred correctly.

5. **DeepSpeed Patch**  
   A monkey-patch method to override DeepSpeed's default pinning behavior, allowing you to use this zero-overhead approach inside DeepSpeed.


## Installation

1. Ensure you have an appropriate **NVIDIA driver**, **CUDA toolkit** (including `nvcc`), and **PyTorch** installed and working with your GPU.
2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install this extension from source:
   ```bash
   pip install .
   ```

## Usage

### 1. Running Tests

Use `pytest` to run the correctness tests:
```bash
pytest test_correctness.py
```
These tests verify that tensors are pinned correctly and that their data remains consistent when transferred to the GPU.

### 2. Memory Usage Comparison

To see the difference in actual host memory usage for power-of-two vs. non-power-of-two allocations, run:
```bash
python memory_usage.py
```
This script will measure unpinned vs. pinned memory usage in a separate process to ensure accurate memory measurements.

### 3. Performance Benchmark

To compare allocation and transfer speeds between:
1. **Our manual pinned memory approach** (`memalign_pin_memory`).
2. **PyTorch's built-in pinned memory** (`pin_memory=True`).

Run:
```bash
python performance_benchmark.py
```
This will print out average allocation times (CPU creation + pinning) and average GPU transfer times for both approaches.

### 4. DeepSpeed Patch

If you want DeepSpeed to use this approach for its pinned memory allocations, you can apply the monkey patch:

```python
from zero_overhead_pinned_memory import patch_deepspeed_zero_overhead_pinned_memory

patch_deepspeed_zero_overhead_pinned_memory()  
# This replaces deepspeed.accelerator.cuda_accelerator.CUDA_Accelerator.pin_memory
# with our zero-overhead pinned memory method
```

## Reference

For more information on CUDA host memory registration, see the official NVIDIA documentation:

- [CUDA Runtime API: cudaHostRegister](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g81fd4101862bbefdb42a62d60e515eea)

**Enjoy faster and more memory-efficient pinned allocations!** If you find any bugs or have suggestions, please open an issue or submit a pull request.