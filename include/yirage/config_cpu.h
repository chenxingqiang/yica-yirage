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

// CPU-specific configuration for YiRage
// This file provides CPU alternatives to CUDA-specific configurations

#ifdef YIRAGE_CPU_ONLY

// Include CPU compatibility layer
#include "yirage/cpu/cpu_compatibility.h"

// CPU-specific configuration constants
namespace yirage {
namespace config {

// Memory configuration (adapted for CPU)
constexpr size_t MAX_DMEM_DATA_SIZE = 2ULL * 1024 * 1024 * 1024; // 2GB for CPU
constexpr size_t MAX_DMEM_FP_SIZE = 512ULL * 1024 * 1024;        // 512MB for fingerprints
constexpr size_t MAX_SMEM_FP_SIZE = 64 * 1024;                   // 64KB shared memory simulation
constexpr int MAX_NUM_THREADBLOCKS_PER_KERNEL = 1024;            // CPU thread blocks
constexpr int MAX_TENSOR_DIMS = 8;                               // Maximum tensor dimensions

// CPU thread configuration
constexpr int DEFAULT_CPU_THREADS = 8;                           // Default CPU threads
constexpr int MAX_CPU_THREADS = 64;                              // Maximum CPU threads

// CPU-specific kernel configuration
constexpr int CPU_WARP_SIZE = 32;                                // Simulated warp size
constexpr int CPU_BLOCK_SIZE = 256;                              // Simulated block size

} // namespace config

// CPU-specific type definitions
namespace type {

// Fingerprint types for CPU
using FPType = uint32_t;
constexpr FPType FP_PQ = 2147483647;  // Large prime for CPU fingerprinting
constexpr FPType FP_P = 2147483647;   // Large prime
constexpr FPType FP_Q = 65537;        // Smaller prime
constexpr FPType FP_EXP_BASE = 3;     // Exponential base

// CPU activation types
enum class CPUActivationType {
    NONE = 0,
    RELU = 1,
    GELU = 2,
    SILU = 3,
    TANH = 4,
    SIGMOID = 5
};

// CPU memory types
enum class CPUMemoryType {
    HOST = 0,
    DEVICE = 1,  // Same as host for CPU
    UNIFIED = 2  // Same as host for CPU
};

} // namespace type

// CPU runtime configuration
namespace runtime {

struct CPURuntimeConfig {
    int num_threads;
    int my_thread_id;
    int num_workers;
    int max_seq_length;
    bool use_openmp;
    bool use_mkl;
    
    CPURuntimeConfig() 
        : num_threads(config::DEFAULT_CPU_THREADS)
        , my_thread_id(0)
        , num_workers(1)
        , max_seq_length(2048)
        , use_openmp(true)
        , use_mkl(false) {}
};

} // namespace runtime

} // namespace yirage

// CPU helper macros
#define CPU_PARALLEL_FOR _Pragma("omp parallel for") for
#define CPU_PARALLEL_SECTIONS _Pragma("omp parallel sections")
#define CPU_SECTION _Pragma("omp section")
#define CPU_BARRIER _Pragma("omp barrier")

// CPU kernel launch macro (replaces CUDA <<<>>> syntax)
#define LAUNCH_CPU_KERNEL(kernel_name, grid_dim, block_dim, shared_mem, stream, ...) \
    do { \
        yirage::cpu::detail::gridDim = grid_dim; \
        yirage::cpu::detail::blockDim = block_dim; \
        for (unsigned int bz = 0; bz < grid_dim.z; bz++) { \
            for (unsigned int by = 0; by < grid_dim.y; by++) { \
                for (unsigned int bx = 0; bx < grid_dim.x; bx++) { \
                    yirage::cpu::detail::blockIdx = yirage::cpu::dim3(bx, by, bz); \
                    CPU_PARALLEL_FOR \
                    for (unsigned int tx = 0; tx < block_dim.x; tx++) { \
                        yirage::cpu::detail::threadIdx = yirage::cpu::dim3(tx, 0, 0); \
                        kernel_name(__VA_ARGS__); \
                    } \
                } \
            } \
        } \
    } while(0)

#else

// Include original CUDA configuration
#include "yirage/config.h"

#endif // YIRAGE_CPU_ONLY
