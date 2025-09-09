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

#include "yirage/backend/cpu/cpu_backend.h"
#include "yirage/backend/cpu/cpu_kernels.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

#ifdef __linux__
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace yirage {
namespace backend {

// CpuStream implementation
CpuStream::CpuStream() : worker_thread_(&CpuStream::worker_loop, this) {}

CpuStream::~CpuStream() {
    stop_flag_ = true;
    condition_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void CpuStream::enqueue(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
        idle_ = false;
    }
    condition_.notify_one();
}

void CpuStream::synchronize() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    condition_.wait(lock, [this] { return task_queue_.empty() && idle_; });
}

bool CpuStream::is_idle() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.empty() && idle_;
}

void CpuStream::worker_loop() {
    while (!stop_flag_) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return !task_queue_.empty() || stop_flag_; });
            
            if (stop_flag_) break;
            
            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
        }
        
        if (task) {
            task();
            
            // Check if queue is empty to set idle flag
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                if (task_queue_.empty()) {
                    idle_ = true;
                    condition_.notify_all();
                }
            }
        }
    }
}

// CpuBackend implementation
CpuBackend::CpuBackend() 
    : current_device_(0), allocated_memory_(0), profiling_active_(false) {
    initialize_cpu_info();
}

CpuBackend::~CpuBackend() {
    // Clean up streams
    std::lock_guard<std::mutex> lock(streams_mutex_);
    streams_.clear();
}

void CpuBackend::initialize_cpu_info() {
    num_cores_ = std::thread::hardware_concurrency();
    if (num_cores_ == 0) {
        num_cores_ = 1; // Fallback
    }
    
    // Get total system memory
#ifdef __linux__
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        total_memory_ = info.totalram * info.mem_unit;
    } else {
        total_memory_ = 8ULL * 1024 * 1024 * 1024; // 8GB fallback
    }
#elif defined(__APPLE__)
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t memsize;
    size_t length = sizeof(memsize);
    if (sysctl(mib, 2, &memsize, &length, NULL, 0) == 0) {
        total_memory_ = memsize;
    } else {
        total_memory_ = 8ULL * 1024 * 1024 * 1024; // 8GB fallback
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        total_memory_ = statex.ullTotalPhys;
    } else {
        total_memory_ = 8ULL * 1024 * 1024 * 1024; // 8GB fallback
    }
#else
    total_memory_ = 8ULL * 1024 * 1024 * 1024; // 8GB fallback
#endif
}

int CpuBackend::get_device_count() {
    return 1; // CPU backend treats the entire system as one device
}

void CpuBackend::set_device(int device_id) {
    if (device_id != 0) {
        throw std::invalid_argument("CPU backend only supports device_id = 0");
    }
    current_device_ = device_id;
}

DeviceInfo CpuBackend::get_device_info(int device_id) {
    if (device_id != 0) {
        throw std::invalid_argument("CPU backend only supports device_id = 0");
    }
    
    DeviceInfo info;
    info.device_id = 0;
    info.name = "CPU";
    info.total_memory = total_memory_;
    info.available_memory = get_available_memory();
    info.compute_units = num_cores_;
    info.backend_type = BackendType::CPU;
    
    return info;
}

void* CpuBackend::allocate(size_t size, MemoryType mem_type) {
    // For CPU backend, all memory types are treated as host memory
    void* ptr = std::aligned_alloc(64, size); // 64-byte aligned for SIMD
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        allocated_blocks_[ptr] = size;
        allocated_memory_ += size;
    }
    
    return ptr;
}

void CpuBackend::deallocate(void* ptr) {
    if (!ptr) return;
    
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        auto it = allocated_blocks_.find(ptr);
        if (it != allocated_blocks_.end()) {
            allocated_memory_ -= it->second;
            allocated_blocks_.erase(it);
        }
    }
    
    std::free(ptr);
}

void CpuBackend::memcpy(void* dst, const void* src, size_t size,
                       MemoryType dst_type, MemoryType src_type) {
    // For CPU backend, all memory is host memory
    std::memcpy(dst, src, size);
}

void CpuBackend::memset(void* ptr, int value, size_t size) {
    std::memset(ptr, value, size);
}

void CpuBackend::synchronize() {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    for (auto& [stream_ptr, stream] : streams_) {
        stream->synchronize();
    }
}

void* CpuBackend::create_stream() {
    auto stream = std::make_unique<CpuStream>();
    void* stream_ptr = stream.get();
    
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        streams_[stream_ptr] = std::move(stream);
    }
    
    return stream_ptr;
}

void CpuBackend::destroy_stream(void* stream) {
    if (!stream) return;
    
    std::lock_guard<std::mutex> lock(streams_mutex_);
    streams_.erase(stream);
}

void CpuBackend::synchronize_stream(void* stream) {
    if (!stream) return;
    
    std::lock_guard<std::mutex> lock(streams_mutex_);
    auto it = streams_.find(stream);
    if (it != streams_.end()) {
        it->second->synchronize();
    }
}

void CpuBackend::start_profiling() {
    start_time_ = std::chrono::high_resolution_clock::now();
    profiling_active_ = true;
}

void CpuBackend::stop_profiling() {
    end_time_ = std::chrono::high_resolution_clock::now();
    profiling_active_ = false;
}

double CpuBackend::get_elapsed_time_ms() {
    if (profiling_active_) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            current_time - start_time_);
        return duration.count() / 1000.0;
    } else {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time_ - start_time_);
        return duration.count() / 1000.0;
    }
}

void CpuBackend::launch_kernel(const std::string& kernel_name,
                             void** inputs, 
                             void** outputs,
                             const KernelConfig& config,
                             const std::vector<size_t>& input_sizes,
                             const std::vector<size_t>& output_sizes) {
    
    // Create a task to execute the kernel
    auto task = [=]() {
        CpuKernelDispatcher::execute(kernel_name, inputs, outputs, 
                                   config, input_sizes, output_sizes);
    };
    
    if (config.stream) {
        // Execute on specified stream
        std::lock_guard<std::mutex> lock(streams_mutex_);
        auto it = streams_.find(config.stream);
        if (it != streams_.end()) {
            it->second->enqueue(task);
        } else {
            throw std::invalid_argument("Invalid stream");
        }
    } else {
        // Execute synchronously
        task();
    }
}

size_t CpuBackend::get_available_memory() {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    return total_memory_ - allocated_memory_;
}

} // namespace backend
} // namespace yirage
