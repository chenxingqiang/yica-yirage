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

#include "yirage/backend/backend_interface.h"
#include <chrono>
#include <memory>
#include <unordered_map>
#include <mutex>

#ifdef YIRAGE_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace yirage {
namespace backend {

class CudaBackend : public BackendInterface {
public:
    CudaBackend();
    virtual ~CudaBackend();
    
    // Backend identification
    BackendType get_backend_type() const override { return BackendType::CUDA; }
    std::string get_backend_name() const override { return "CUDA"; }
    
    // Device management
    int get_device_count() override;
    void set_device(int device_id) override;
    int get_current_device() override;
    DeviceInfo get_device_info(int device_id) override;
    
    // Memory management
    void* allocate(size_t size, MemoryType mem_type = MemoryType::DEVICE) override;
    void deallocate(void* ptr) override;
    void memcpy(void* dst, const void* src, size_t size, 
               MemoryType dst_type = MemoryType::DEVICE,
               MemoryType src_type = MemoryType::DEVICE) override;
    void memset(void* ptr, int value, size_t size) override;
    
    // Stream management
    void synchronize() override;
    void* create_stream() override;
    void destroy_stream(void* stream) override;
    void synchronize_stream(void* stream) override;
    
    // Performance profiling
    void start_profiling() override;
    void stop_profiling() override;
    double get_elapsed_time_ms() override;
    
    // Kernel execution
    void launch_kernel(const std::string& kernel_name,
                     void** inputs, 
                     void** outputs,
                     const KernelConfig& config,
                     const std::vector<size_t>& input_sizes,
                     const std::vector<size_t>& output_sizes) override;

#ifdef YIRAGE_USE_CUDA
    // CUDA-specific getters
    cudaStream_t get_cuda_stream(void* stream) const;
    cublasHandle_t get_cublas_handle() const { return cublas_handle_; }
#endif

private:
    int current_device_;
    bool is_available_;
    
    // Profiling
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool profiling_active_;
    
#ifdef YIRAGE_USE_CUDA
    // CUDA events for profiling
    cudaEvent_t start_event_;
    cudaEvent_t end_event_;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_;
    
    // Stream tracking
    std::unordered_map<void*, cudaStream_t> stream_map_;
    std::mutex stream_mutex_;
#endif
    
    void initialize_cuda();
    void cleanup_cuda();
    size_t get_available_memory();
};

} // namespace backend
} // namespace yirage
