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

#include "yirage/type.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace yirage {
namespace backend {

enum class BackendType {
    CUDA,
    CPU,
    MPS,
    AUTO  // Automatically select the best available backend
};

enum class MemoryType {
    HOST,
    DEVICE,
    UNIFIED
};

struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t available_memory;
    int compute_units;
    BackendType backend_type;
};

struct KernelConfig {
    std::array<int, 3> grid_dim = {1, 1, 1};
    std::array<int, 3> block_dim = {1, 1, 1};
    size_t shared_memory_size = 0;
    void* stream = nullptr;
};

// Abstract base class for all backend implementations
class BackendInterface {
public:
    virtual ~BackendInterface() = default;
    
    // Backend identification
    virtual BackendType get_backend_type() const = 0;
    virtual std::string get_backend_name() const = 0;
    
    // Device management
    virtual int get_device_count() = 0;
    virtual void set_device(int device_id) = 0;
    virtual int get_current_device() = 0;
    virtual DeviceInfo get_device_info(int device_id) = 0;
    
    // Memory management
    virtual void* allocate(size_t size, MemoryType mem_type = MemoryType::DEVICE) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void memcpy(void* dst, const void* src, size_t size, 
                       MemoryType dst_type = MemoryType::DEVICE,
                       MemoryType src_type = MemoryType::DEVICE) = 0;
    virtual void memset(void* ptr, int value, size_t size) = 0;
    
    // Stream/queue management
    virtual void synchronize() = 0;
    virtual void* create_stream() = 0;
    virtual void destroy_stream(void* stream) = 0;
    virtual void synchronize_stream(void* stream) = 0;
    
    // Performance profiling
    virtual void start_profiling() = 0;
    virtual void stop_profiling() = 0;
    virtual double get_elapsed_time_ms() = 0;
    
    // Kernel execution (will be implemented by specific kernels)
    virtual void launch_kernel(const std::string& kernel_name,
                             void** inputs, 
                             void** outputs,
                             const KernelConfig& config,
                             const std::vector<size_t>& input_sizes,
                             const std::vector<size_t>& output_sizes) = 0;
};

// Factory for creating backend instances
class BackendFactory {
public:
    static std::unique_ptr<BackendInterface> create(BackendType type);
    static BackendType detect_best_backend();
    static std::vector<BackendType> get_available_backends();
    static bool is_backend_available(BackendType type);
};

} // namespace backend
} // namespace yirage
