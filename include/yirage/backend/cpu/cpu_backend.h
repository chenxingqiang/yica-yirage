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
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

namespace yirage {
namespace backend {

// CPU stream implementation using thread pool
class CpuStream {
public:
    CpuStream();
    ~CpuStream();
    
    void enqueue(std::function<void()> task);
    void synchronize();
    bool is_idle() const;
    
private:
    std::thread worker_thread_;
    std::queue<std::function<void()>> task_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> idle_{true};
    
    void worker_loop();
};

class CpuBackend : public BackendInterface {
public:
    CpuBackend();
    virtual ~CpuBackend();
    
    // Backend identification
    BackendType get_backend_type() const override { return BackendType::CPU; }
    std::string get_backend_name() const override { return "CPU"; }
    
    // Device management (CPU treats cores as "devices")
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

private:
    int current_device_;
    int num_cores_;
    size_t total_memory_;
    size_t allocated_memory_;
    
    // Profiling
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool profiling_active_;
    
    // Stream management
    std::unordered_map<void*, std::unique_ptr<CpuStream>> streams_;
    std::mutex streams_mutex_;
    
    // Memory tracking
    std::unordered_map<void*, size_t> allocated_blocks_;
    std::mutex memory_mutex_;
    
    // Helper functions
    void initialize_cpu_info();
    size_t get_available_memory();
};

} // namespace backend
} // namespace yirage
