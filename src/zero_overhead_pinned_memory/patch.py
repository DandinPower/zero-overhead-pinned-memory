from zero_overhead_pinned_memory import memalign_pin_memory

def zero_overhead_pin_memory(self, tensor, align_bytes=1):
    if tensor.is_pinned():
        return tensor
    return memalign_pin_memory(tensor)

def patch_deepspeed_zero_overhead_pinned_memory():
    import deepspeed
    print("Apply zero overhead pin_memory to deepspeed CUDA_Accelerator")
    deepspeed.accelerator.cuda_accelerator.CUDA_Accelerator.pin_memory = zero_overhead_pin_memory