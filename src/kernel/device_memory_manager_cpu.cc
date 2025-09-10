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

#include "yirage/kernel/device_memory_manager.h"

#ifdef YIRAGE_CPU_ONLY
#include "yirage/cpu/cpu_compatibility.h"
#else
#include "yirage/utils/cuda_helper.h"
#endif

#include <cstdlib>
#include <cstring>

namespace yirage {
namespace kernel {

using namespace yirage::type;
using namespace yirage::config;

DeviceMemoryManager *DeviceMemoryManager::singleton = nullptr;

DeviceMemoryManager::DeviceMemoryManager(int _num_gpus, int _gpu_id)
    : num_gpus(_num_gpus), gpu_id(_gpu_id) {
    
#ifdef YIRAGE_CPU_ONLY
    printf("YiRage::DeviceMemoryManager: CPU-only mode, gpu_id(%d) num_gpus(%d)\n", gpu_id, num_gpus);
    
    // CPU-only initialization
    initialize_cpu_lookup_tables();
    initialize_cpu_memory_pools();
    
#else
    // Original CUDA initialization code
    checkCUDA(cudaSetDevice(gpu_id));
    printf("YiRage::DeviceMemoryManager: gpu_id(%d) num_gpus(%d)", gpu_id, num_gpus);
    
    // Initialize CUDA lookup tables and memory
    initialize_cuda_lookup_tables();
    initialize_cuda_memory_pools();
#endif
}

DeviceMemoryManager::~DeviceMemoryManager() {
#ifdef YIRAGE_CPU_ONLY
    cleanup_cpu_resources();
#else
    cleanup_cuda_resources();
#endif
}

#ifdef YIRAGE_CPU_ONLY

void DeviceMemoryManager::initialize_cpu_lookup_tables() {
    // CPU implementation of lookup tables
    printf("Initializing CPU lookup tables...\n");
    
    // Allocate exponential lookup table with alignment
    size_t aligned_size = (sizeof(FPType) * FP_Q + 15) / 16 * 16;
    exp_lookup_table = static_cast<FPType*>(std::aligned_alloc(64, aligned_size));
    
    // Check PQ relations
    assert(FP_Q < FP_P);
    assert((FP_P - 1) % FP_Q == 0);
    
    // Initialize exponential table
    FPType exp_table[FP_Q];
    exp_table[0] = 1;
    for (int i = 1; i < FP_Q; i++) {
        exp_table[i] = (exp_table[i - 1] * FP_EXP_BASE) % FP_P;
    }
    assert((exp_table[FP_Q - 1] * FP_EXP_BASE) % FP_P == 1);
    std::memcpy(exp_lookup_table, exp_table, sizeof(FPType) * FP_Q);
    
    // Initialize division p lookup table
    div_p_lookup_table = static_cast<FPType*>(
        std::aligned_alloc(64, sizeof(FPType) * FP_P));
    
    FPType div_p_table[FP_P];
    for (uint32_t i = 0; i < FP_P; i++) {
        div_p_table[i] = 1;
        for (uint32_t j = 1; j < FP_P; j++) {
            if ((i * j) % FP_P == 1) {
                div_p_table[i] = j;
                break;
            }
        }
    }
    std::memcpy(div_p_lookup_table, div_p_table, sizeof(FPType) * FP_P);
    
    // Initialize division q lookup table
    div_q_lookup_table = static_cast<FPType*>(
        std::aligned_alloc(64, sizeof(FPType) * FP_Q));
    
    FPType div_q_table[FP_Q];
    for (uint32_t i = 0; i < FP_Q; i++) {
        div_q_table[i] = 1;
        for (uint32_t j = 1; j < FP_Q; j++) {
            if ((i * j) % FP_Q == 1) {
                div_q_table[i] = j;
                break;
            }
        }
    }
    std::memcpy(div_q_lookup_table, div_q_table, sizeof(FPType) * FP_Q);
    
    // Initialize sqrt lookup tables
    sqrt_p_lookup_table = static_cast<FPType*>(
        std::aligned_alloc(64, sizeof(FPType) * FP_P));
    sqrt_q_lookup_table = static_cast<FPType*>(
        std::aligned_alloc(64, sizeof(FPType) * FP_Q));
    
    // Initialize sqrt tables (simplified for CPU)
    for (uint32_t i = 0; i < FP_P; i++) {
        sqrt_p_lookup_table[i] = static_cast<FPType>(std::sqrt(i) + 0.5);
    }
    for (uint32_t i = 0; i < FP_Q; i++) {
        sqrt_q_lookup_table[i] = static_cast<FPType>(std::sqrt(i) + 0.5);
    }
}

void DeviceMemoryManager::initialize_cpu_memory_pools() {
    // CPU memory pools
    for (int i = 0; i < num_gpus; i++) {
        fp_base_ptr[i] = static_cast<char*>(
            std::aligned_alloc(64, yirage::config::MAX_DMEM_FP_SIZE));
        data_base_ptr[i] = static_cast<char*>(
            std::aligned_alloc(64, yirage::config::MAX_DMEM_DATA_SIZE));
    }
    
    stensor_fp_base_ptr = static_cast<char*>(
        std::aligned_alloc(64, yirage::config::MAX_SMEM_FP_SIZE * 
                              yirage::config::MAX_NUM_THREADBLOCKS_PER_KERNEL));
}

void DeviceMemoryManager::cleanup_cpu_resources() {
    if (exp_lookup_table) std::free(exp_lookup_table);
    if (div_p_lookup_table) std::free(div_p_lookup_table);
    if (div_q_lookup_table) std::free(div_q_lookup_table);
    if (sqrt_p_lookup_table) std::free(sqrt_p_lookup_table);
    if (sqrt_q_lookup_table) std::free(sqrt_q_lookup_table);
    
    for (int i = 0; i < num_gpus; i++) {
        if (fp_base_ptr[i]) std::free(fp_base_ptr[i]);
        if (data_base_ptr[i]) std::free(data_base_ptr[i]);
    }
    
    if (stensor_fp_base_ptr) std::free(stensor_fp_base_ptr);
}

#endif // YIRAGE_CPU_ONLY

DeviceMemoryManager *DeviceMemoryManager::get_instance() {
    if (singleton == nullptr) {
#ifdef YIRAGE_CPU_ONLY
        // CPU-only: simulate single device
        singleton = new DeviceMemoryManager(1 /*num_gpus*/, 0 /*device_id*/);
#else
        int num_gpus;
        checkCUDA(cudaGetDeviceCount(&num_gpus));
        singleton = new DeviceMemoryManager(1 /*num_gpus*/, 0 /*device_id*/);
#endif
    }
    return singleton;
}

/*static*/
void DeviceMemoryManager::set_gpu_device_id(int gpu_id) {
    // set_gpu_device_id must be called before creating DeviceMemoryManager
    assert(singleton == nullptr);
#ifdef YIRAGE_CPU_ONLY
    // CPU-only: ignore device ID, always use 0
    singleton = new DeviceMemoryManager(1 /*num_gpus*/, 0 /*gpu_id*/);
#else
    int num_gpus;
    checkCUDA(cudaGetDeviceCount(&num_gpus));
    singleton = new DeviceMemoryManager(1 /*num_gpus*/, gpu_id /*gpu_id*/);
#endif
}

void cython_set_gpu_device_id(int gpu_id) {
    DeviceMemoryManager::set_gpu_device_id(gpu_id);
}

} // namespace kernel
} // namespace yirage
