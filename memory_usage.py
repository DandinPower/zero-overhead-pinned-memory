import torch
import psutil
import os
import gc
import multiprocessing

from zero_overhead_pinned_memory import memalign_pin_memory

def get_memory_usage():
    """Returns the current and peak memory usage (in GB)."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss, mem_info.vms

def measure_register_pinned_memory_overhead(tensor_size, dtype, queue):
    """Measures memory overhead of pinned memory allocation and sends results to the queue."""
    # Measure unpinned memory usage
    mem_before, _ = get_memory_usage()
    tensor = torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=False)
    mem_after, _ = get_memory_usage()
    unpinned_memory_size = mem_after - mem_before
    del tensor
    gc.collect()

    # Measure pinned memory usage
    mem_before, _ = get_memory_usage()
    tensor = memalign_pin_memory(torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=False))
    mem_after, _ = get_memory_usage()
    pinned_memory_size = mem_after - mem_before
    del tensor
    gc.collect()

    # Calculate overhead
    overhead = pinned_memory_size / unpinned_memory_size if unpinned_memory_size > 0 else float('inf')

    # Send results back through the queue
    queue.put((unpinned_memory_size, pinned_memory_size, overhead))

def get_register_pinned_overhead(tensor_size, dtype):
    """Runs the memory test in a separate process and returns the results."""
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=measure_register_pinned_memory_overhead, args=(tensor_size, dtype, queue))
    process.start()
    process.join()

    return queue.get()  # Retrieve results from the process

def measure_torch_pinned_memory_overhead(tensor_size, dtype, queue):
    """Measures memory overhead of pinned memory allocation and sends results to the queue."""
    # Measure unpinned memory usage
    mem_before, _ = get_memory_usage()
    tensor = torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=False)
    mem_after, _ = get_memory_usage()
    unpinned_memory_size = mem_after - mem_before
    del tensor
    gc.collect()

    # Measure pinned memory usage
    mem_before, _ = get_memory_usage()
    tensor = torch.zeros(tensor_size, device='cpu', dtype=dtype, pin_memory=True)
    mem_after, _ = get_memory_usage()
    pinned_memory_size = mem_after - mem_before
    del tensor
    gc.collect()

    # Calculate overhead
    overhead = pinned_memory_size / unpinned_memory_size if unpinned_memory_size > 0 else float('inf')

    # Send results back through the queue
    queue.put((unpinned_memory_size, pinned_memory_size, overhead))

def get_torch_pinned_overhead(tensor_size, dtype):
    """Runs the memory test in a separate process and returns the results."""
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=measure_torch_pinned_memory_overhead, args=(tensor_size, dtype, queue))
    process.start()
    process.join()

    return queue.get()  # Retrieve results from the process

if __name__ == "__main__":
    print(f"Power of 2")
    print(f"pytorch(baseline)")
    unpinned, pinned, _ = get_torch_pinned_overhead(int(4 * 1024**3), torch.float16)
    print(f"unpinned: {unpinned}")
    print(f"pinned: {pinned}")
    
    print(f"our method")
    unpinned, pinned, _ = get_register_pinned_overhead(int(4 * 1024**3), torch.float16)
    print(f"unpinned: {unpinned}")
    print(f"pinned: {pinned}")
    
    print(f"Non Power of 2")
    print(f"pytorch(baseline)")
    unpinned, pinned, _ = get_torch_pinned_overhead(int(4.01 * 1024**3), torch.float16)
    print(f"unpinned: {unpinned}")
    print(f"pinned: {pinned}")
    
    print(f"our method")
    unpinned, pinned, _ = get_register_pinned_overhead(int(4.01 * 1024**3), torch.float16)
    print(f"unpinned: {unpinned}")
    print(f"pinned: {pinned}")