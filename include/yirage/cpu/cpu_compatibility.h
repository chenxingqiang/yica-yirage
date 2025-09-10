/* Copyright 2025-2026 YICA TEAM
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef YIRAGE_CPU_ONLY

// CPU-only compatibility layer for CUDA types and functions

#include <cstdint>
#include <vector>
#include <memory>

// Replace CUDA types with CPU equivalents
namespace yirage {
namespace cpu {

// CUDA type replacements
using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;

// CUDA constants
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInvalidValue = 1;

// CUDA memory types
enum cudaMemoryType {
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
};

// CUDA memory copy kinds
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

// CUDA device properties (simplified)
struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int multiProcessorCount;
    int major;
    int minor;
};

// CUDA pointer attributes
struct cudaPointerAttributes {
    cudaMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
};

// Dimension types
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};

struct int3 {
    int x, y, z;
    int3(int x = 0, int y = 0, int z = 0) : x(x), y(y), z(z) {}
    bool operator==(const int3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct int2 {
    int x, y;
    int2(int x = 0, int y = 0) : x(x), y(y) {}
};

// CUDA function replacements (no-ops or CPU implementations)
inline cudaError_t cudaGetDeviceCount(int* count) {
    *count = 0;  // No CUDA devices
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int device) {
    return (device == 0) ? cudaSuccess : cudaErrorInvalidValue;
}

inline cudaError_t cudaGetDevice(int* device) {
    *device = 0;
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    if (device != 0) return cudaErrorInvalidValue;
    strcpy(prop->name, "CPU Device");
    prop->totalGlobalMem = 8ULL * 1024 * 1024 * 1024;  // 8GB
    prop->multiProcessorCount = 1;
    prop->major = 0;
    prop->minor = 0;
    return cudaSuccess;
}

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
    *ptr = std::aligned_alloc(64, size);  // 64-byte aligned
    return (*ptr != nullptr) ? cudaSuccess : cudaErrorInvalidValue;
}

inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
    *ptr = std::aligned_alloc(64, size);
    return (*ptr != nullptr) ? cudaSuccess : cudaErrorInvalidValue;
}

inline cudaError_t cudaMallocManaged(void** ptr, size_t size) {
    *ptr = std::aligned_alloc(64, size);
    return (*ptr != nullptr) ? cudaSuccess : cudaErrorInvalidValue;
}

inline cudaError_t cudaFree(void* ptr) {
    if (ptr) std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaFreeHost(void* ptr) {
    if (ptr) std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    std::memcpy(dst, src, size);
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* ptr, int value, size_t size) {
    std::memset(ptr, value, size);
    return cudaSuccess;
}

inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    *total = 8ULL * 1024 * 1024 * 1024;  // 8GB
    *free = *total / 2;  // Assume 50% free
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;  // No-op for CPU
}

inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    *stream = nullptr;  // CPU streams are no-op
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return cudaSuccess;  // No-op
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return cudaSuccess;  // No-op
}

inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
    *event = nullptr;
    return cudaSuccess;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
    return cudaSuccess;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return cudaSuccess;
}

inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    *ms = 0.0f;  // No timing for CPU
    return cudaSuccess;
}

inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
    attributes->type = cudaMemoryTypeHost;
    attributes->device = 0;
    attributes->devicePointer = const_cast<void*>(ptr);
    attributes->hostPointer = const_cast<void*>(ptr);
    return cudaSuccess;
}

// Thread and block index replacements (for kernel-like code)
namespace detail {
    extern thread_local dim3 blockIdx;
    extern thread_local dim3 blockDim;
    extern thread_local dim3 threadIdx;
    extern thread_local dim3 gridDim;
}

// Make CUDA-like variables available globally
#define blockIdx yirage::cpu::detail::blockIdx
#define blockDim yirage::cpu::detail::blockDim
#define threadIdx yirage::cpu::detail::threadIdx
#define gridDim yirage::cpu::detail::gridDim

// Helper macros for CPU compatibility
#define __global__ 
#define __device__ 
#define __host__ 
#define __shared__ static
#define __syncthreads() do {} while(0)
#define __forceinline__ inline

// Error checking macro for CPU (no-op)
#define checkCUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        /* Log error but don't abort in CPU mode */ \
    } \
} while(0)

} // namespace cpu
} // namespace yirage

// Make CPU types available in global namespace for compatibility
using yirage::cpu::cudaError_t;
using yirage::cpu::cudaStream_t;
using yirage::cpu::cudaEvent_t;
using yirage::cpu::cudaMemoryType;
using yirage::cpu::cudaMemcpyKind;
using yirage::cpu::cudaDeviceProp;
using yirage::cpu::cudaPointerAttributes;
// Note: dim3, int3, int2 types are defined in utils/containers.h to avoid conflicts

// CUDA function replacements
using yirage::cpu::cudaGetDeviceCount;
using yirage::cpu::cudaSetDevice;
using yirage::cpu::cudaGetDevice;
using yirage::cpu::cudaGetDeviceProperties;
using yirage::cpu::cudaMalloc;
using yirage::cpu::cudaMallocHost;
using yirage::cpu::cudaMallocManaged;
using yirage::cpu::cudaFree;
using yirage::cpu::cudaFreeHost;
using yirage::cpu::cudaMemcpy;
using yirage::cpu::cudaMemset;
using yirage::cpu::cudaMemGetInfo;
using yirage::cpu::cudaDeviceSynchronize;
using yirage::cpu::cudaStreamCreate;
using yirage::cpu::cudaStreamDestroy;
using yirage::cpu::cudaStreamSynchronize;
using yirage::cpu::cudaEventCreate;
using yirage::cpu::cudaEventDestroy;
using yirage::cpu::cudaEventRecord;
using yirage::cpu::cudaEventSynchronize;
using yirage::cpu::cudaEventElapsedTime;
using yirage::cpu::cudaPointerGetAttributes;

#endif // YIRAGE_CPU_ONLY
