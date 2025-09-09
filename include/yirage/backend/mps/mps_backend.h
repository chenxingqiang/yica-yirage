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

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace yirage {
namespace backend {

#ifdef __APPLE__
// Forward declarations for Objective-C++ types
@class MTLDevice;
@class MTLCommandQueue;
@class MTLCommandBuffer;
@class MTLBuffer;
#endif

class MpsBackend : public BackendInterface {
public:
    MpsBackend();
    virtual ~MpsBackend();
    
    // Backend identification
    BackendType get_backend_type() const override { return BackendType::MPS; }
    std::string get_backend_name() const override { return "MPS"; }
    
    // Device management
    int get_device_count() override;
    void set_device(int device_id) override;
    int get_current_device() override { return current_device_; }
    DeviceInfo get_device_info(int device_id) override;
    
    // Memory management
    void* allocate(size_t size, MemoryType mem_type = MemoryType::DEVICE) override;
    void deallocate(void* ptr) override;
    void memcpy(void* dst, const void* src, size_t size, 
               MemoryType dst_type = MemoryType::DEVICE,
               MemoryType src_type = MemoryType::DEVICE) override;
    void memset(void* ptr, int value, size_t size) override;
    
    // Stream management (Metal command queues)
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

#ifdef __APPLE__
    // MPS-specific methods
    id<MTLDevice> get_metal_device() const { return metal_device_; }
    id<MTLCommandQueue> get_default_command_queue() const { return default_command_queue_; }
#endif

private:
    int current_device_;
    bool is_available_;
    
    // Profiling
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool profiling_active_;
    
#ifdef __APPLE__
    id<MTLDevice> metal_device_;
    id<MTLCommandQueue> default_command_queue_;
    
    // Buffer tracking
    std::unordered_map<void*, id<MTLBuffer>> buffer_map_;
    std::mutex buffer_mutex_;
#endif
    
    void initialize_metal();
    size_t get_available_memory();
    
#ifdef __APPLE__
    // MPS kernel implementations
    void launch_mps_matmul(id<MTLCommandBuffer> commandBuffer,
                          void** inputs, void** outputs,
                          const KernelConfig& config,
                          const std::vector<size_t>& input_sizes,
                          const std::vector<size_t>& output_sizes);
    
    void launch_mps_rms_norm(id<MTLCommandBuffer> commandBuffer,
                            void** inputs, void** outputs,
                            const KernelConfig& config,
                            const std::vector<size_t>& input_sizes,
                            const std::vector<size_t>& output_sizes);
    
    void launch_mps_element_unary(id<MTLCommandBuffer> commandBuffer,
                                 void** inputs, void** outputs,
                                 const KernelConfig& config,
                                 const std::vector<size_t>& input_sizes,
                                 const std::vector<size_t>& output_sizes);
#endif
};

} // namespace backend
} // namespace yirage
