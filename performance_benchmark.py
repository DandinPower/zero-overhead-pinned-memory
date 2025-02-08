import time
import torch

from zero_overhead_pinned_memory import memalign_pin_memory

def approach1(tensor_size, dtype):
    """
    Create a CPU tensor without pinned memory and then pin it manually.
    """
    tensor = memalign_pin_memory(torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=False))
    return tensor

def approach2(tensor_size, dtype):
    """
    Create a CPU tensor that is already pinned.
    """
    tensor = torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=True)
    return tensor

def benchmark_creation(method, tensor_size, dtype, num_iters=100):
    """
    Benchmark the CPU-side creation (including pinning) time.
    """
    times = []
    for _ in range(num_iters):
        start = time.time()
        _ = method(tensor_size, dtype)
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / len(times)
    return avg_time

def benchmark_transfer(method, tensor_size, dtype, num_iters=100, device='cuda'):
    """
    Benchmark the time to transfer a pinned CPU tensor to the GPU.
    We use torch.cuda.Event for accurate timing since the copy is asynchronous.
    """
    # Warm-up iterations.
    for _ in range(10):
        t = method(tensor_size, dtype)
        _ = t.to(device)
        torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(num_iters):
        t = method(tensor_size, dtype)
        # Ensure CPU work is done.
        torch.cuda.synchronize()
        start_evt.record()
        _ = t.to(device)
        end_evt.record()
        # Wait for the GPU copy to finish.
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))  # elapsed_time returns ms.
    avg_time = sum(times) / len(times)
    return avg_time

if __name__ == '__main__':
    # Define tensor parameters.
    tensor_size = (1024, 1024, 1024)
    dtype = torch.float32

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit(0)

    num_iters = 5

    # Benchmark the CPU creation and pinning times.
    creation_time1 = benchmark_creation(approach1, tensor_size, dtype, num_iters)
    creation_time2 = benchmark_creation(approach2, tensor_size, dtype, num_iters)
    print("CPU creation + pinning time (per iteration):")
    print(f"  Approach 1 (our method): {creation_time1*1e3:.3f} ms")
    print(f"  Approach 2 (pytorch): {creation_time2*1e3:.3f} ms")

    # Benchmark the GPU transfer times.
    transfer_time1 = benchmark_transfer(approach1, tensor_size, dtype, num_iters)
    transfer_time2 = benchmark_transfer(approach2, tensor_size, dtype, num_iters)
    print("\nGPU transfer time (per iteration):")
    print(f"  Approach 1 (our method): {transfer_time1:.3f} ms")
    print(f"  Approach 2 (pytorch): {transfer_time2:.3f} ms")
