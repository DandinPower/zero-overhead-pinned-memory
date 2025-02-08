#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdlib>      // for posix_memalign, free
#include <cstring>      // for memset
#include <stdexcept>    // for std::runtime_error

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA ERROR (%s:%d): %s\n",                      \
                   __FILE__, __LINE__, cudaGetErrorString(err));            \
      std::exit(err);                                                       \
    }                                                                       \
  } while (0)

void posix_memory_free(void* ptr) {
    if (!ptr) {
        std::fprintf(stderr, "WARNING: Attempted to free a null pointer.\n");
        return;
    }

    // std::printf("INFO: Freeing pinned memory at address %p\n", ptr);

    CUDA_CHECK(cudaHostUnregister(ptr)); 
    CUDA_CHECK(cudaDeviceSynchronize());

    free(ptr);
}

torch::Tensor memalign_pin_memory(torch::Tensor& src) {
    TORCH_CHECK(src.device().is_cpu(), 
                "ERROR: Expected a CPU tensor, but received a tensor on device: ", 
                src.device().str());

    const size_t DMA_ALIGNMENT = 4096;
    size_t size = src.numel() * src.element_size(); 
    size_t allocate_size = ((size + DMA_ALIGNMENT - 1) / DMA_ALIGNMENT) * DMA_ALIGNMENT;

    void* data_ptr = nullptr;
    int ret = posix_memalign(&data_ptr, DMA_ALIGNMENT, allocate_size);
    if (ret != 0) {
        throw std::runtime_error("ERROR: Failed to allocate aligned memory. posix_memalign returned: " + std::to_string(ret));
    }

    std::memset(data_ptr, 0, allocate_size);

    CUDA_CHECK(cudaHostRegister(data_ptr, allocate_size, cudaHostRegisterDefault)); 
    CUDA_CHECK(cudaDeviceSynchronize());

    auto tensor = torch::from_blob(data_ptr, src.sizes(), posix_memory_free, src.options());
    tensor.copy_(src);

    // std::printf("INFO: Allocated pinned memory at %p (size: %zu bytes, aligned to %zu bytes)\n", 
                // data_ptr, size, DMA_ALIGNMENT);

    return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     m.def("memalign_pin_memory", &memalign_pin_memory,
          "Allocate pinned CPU memory aligned to 4096 bytes and copy the input tensor data into it. "
          "Returns a tensor with the same shape and options as the input.");
}