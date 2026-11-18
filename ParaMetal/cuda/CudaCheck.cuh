#pragma once

#include <cuda_runtime.h>

#include <iostream>

namespace cudaUtils {

inline bool cudaCheck(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return true;
    (void)cudaGetLastError();
    std::cerr << "[CUDA] " << operation << " failed: "
              << cudaGetErrorString(result) << std::endl;
    return false;
}

// Diagnostic: logs a raw CUDA result without altering control flow.
inline void cudaTrace(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        (void)cudaGetLastError();
        std::cerr << "[CUDA_TRACE] " << operation << " failed: "
                  << cudaGetErrorString(result) << std::endl;
    }
}

} // namespace cudaUtils
