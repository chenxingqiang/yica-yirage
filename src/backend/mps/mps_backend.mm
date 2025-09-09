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

#include "yirage/backend/mps/mps_backend.h"
#include <stdexcept>
#include <iostream>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#endif

namespace yirage {
namespace backend {

#ifdef __APPLE__

MpsBackend::MpsBackend() 
    : current_device_(0), is_available_(false), profiling_active_(false) {
    initialize_metal();
}

MpsBackend::~MpsBackend() {
    // Clean up Metal resources
    @autoreleasepool {
        if (default_command_queue_) {
            [default_command_queue_ release];
            default_command_queue_ = nil;
        }
        
        if (metal_device_) {
            [metal_device_ release];
            metal_device_ = nil;
        }
        
        // Clean up buffers
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        for (auto& [ptr, buffer] : buffer_map_) {
            [buffer release];
        }
        buffer_map_.clear();
    }
}

void MpsBackend::initialize_metal() {
    @autoreleasepool {
        // Get the default Metal device
        metal_device_ = MTLCreateSystemDefaultDevice();
        if (!metal_device_) {
            std::cerr << "Metal is not supported on this device" << std::endl;
            return;
        }
        
        // Retain the device
        [metal_device_ retain];
        
        // Create command queue
        default_command_queue_ = [metal_device_ newCommandQueue];
        if (!default_command_queue_) {
            std::cerr << "Failed to create Metal command queue" << std::endl;
            return;
        }
        
        [default_command_queue_ retain];
        
        is_available_ = true;
        
        NSString* deviceName = [metal_device_ name];
        std::cout << "Initialized MPS backend with device: " 
                  << [deviceName UTF8String] << std::endl;
    }
}

int MpsBackend::get_device_count() {
    return is_available_ ? 1 : 0;
}

void MpsBackend::set_device(int device_id) {
    if (device_id != 0) {
        throw std::invalid_argument("MPS backend only supports device_id = 0");
    }
    current_device_ = device_id;
}

DeviceInfo MpsBackend::get_device_info(int device_id) {
    if (device_id != 0 || !is_available_) {
        throw std::invalid_argument("MPS backend only supports device_id = 0");
    }
    
    DeviceInfo info;
    info.device_id = 0;
    info.backend_type = BackendType::MPS;
    
    @autoreleasepool {
        NSString* deviceName = [metal_device_ name];
        info.name = std::string([deviceName UTF8String]);
        
        // Get memory information
        info.total_memory = [metal_device_ recommendedMaxWorkingSetSize];
        info.available_memory = get_available_memory();
        
        // Get compute units (shader cores)
        info.compute_units = 0; // Metal doesn't expose this directly
        
        // Try to get more detailed info for Apple Silicon
        if ([metal_device_ supportsFamily:MTLGPUFamilyApple7]) {
            info.compute_units = 1024; // Estimate for M1/M2
        } else if ([metal_device_ supportsFamily:MTLGPUFamilyApple6]) {
            info.compute_units = 512; // Estimate for older Apple GPUs
        }
    }
    
    return info;
}

void* MpsBackend::allocate(size_t size, MemoryType mem_type) {
    if (!is_available_) {
        throw std::runtime_error("MPS backend is not available");
    }
    
    @autoreleasepool {
        // Create Metal buffer
        id<MTLBuffer> buffer = [metal_device_ newBufferWithLength:size 
                                                         options:MTLResourceStorageModeShared];
        
        if (!buffer) {
            throw std::bad_alloc();
        }
        
        [buffer retain];
        
        void* ptr = [buffer contents];
        
        // Store buffer mapping
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            buffer_map_[ptr] = buffer;
        }
        
        return ptr;
    }
}

void MpsBackend::deallocate(void* ptr) {
    if (!ptr || !is_available_) return;
    
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        auto it = buffer_map_.find(ptr);
        if (it != buffer_map_.end()) {
            [it->second release];
            buffer_map_.erase(it);
        }
    }
}

void MpsBackend::memcpy(void* dst, const void* src, size_t size,
                       MemoryType dst_type, MemoryType src_type) {
    // For MPS, we use shared memory, so regular memcpy works
    std::memcpy(dst, src, size);
}

void MpsBackend::memset(void* ptr, int value, size_t size) {
    std::memset(ptr, value, size);
}

void MpsBackend::synchronize() {
    if (!is_available_) return;
    
    @autoreleasepool {
        // Create a command buffer and commit it to ensure synchronization
        id<MTLCommandBuffer> commandBuffer = [default_command_queue_ commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void* MpsBackend::create_stream() {
    if (!is_available_) {
        throw std::runtime_error("MPS backend is not available");
    }
    
    @autoreleasepool {
        // Create a new command queue for this stream
        id<MTLCommandQueue> commandQueue = [metal_device_ newCommandQueue];
        if (!commandQueue) {
            throw std::runtime_error("Failed to create MPS command queue");
        }
        
        [commandQueue retain];
        return (__bridge_retained void*)commandQueue;
    }
}

void MpsBackend::destroy_stream(void* stream) {
    if (!stream) return;
    
    @autoreleasepool {
        id<MTLCommandQueue> commandQueue = (__bridge_transfer id<MTLCommandQueue>)stream;
        // commandQueue will be released automatically by ARC
    }
}

void MpsBackend::synchronize_stream(void* stream) {
    if (!stream || !is_available_) return;
    
    @autoreleasepool {
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)stream;
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void MpsBackend::start_profiling() {
    start_time_ = std::chrono::high_resolution_clock::now();
    profiling_active_ = true;
}

void MpsBackend::stop_profiling() {
    end_time_ = std::chrono::high_resolution_clock::now();
    profiling_active_ = false;
}

double MpsBackend::get_elapsed_time_ms() {
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

void MpsBackend::launch_kernel(const std::string& kernel_name,
                             void** inputs, 
                             void** outputs,
                             const KernelConfig& config,
                             const std::vector<size_t>& input_sizes,
                             const std::vector<size_t>& output_sizes) {
    if (!is_available_) {
        throw std::runtime_error("MPS backend is not available");
    }
    
    @autoreleasepool {
        // Get command queue
        id<MTLCommandQueue> commandQueue = config.stream ? 
            (__bridge id<MTLCommandQueue>)config.stream : default_command_queue_;
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        // Dispatch to MPS kernel implementations based on kernel_name
        if (kernel_name == "matmul") {
            launch_mps_matmul(commandBuffer, inputs, outputs, config, input_sizes, output_sizes);
        } else if (kernel_name == "rms_norm") {
            launch_mps_rms_norm(commandBuffer, inputs, outputs, config, input_sizes, output_sizes);
        } else if (kernel_name == "element_unary") {
            launch_mps_element_unary(commandBuffer, inputs, outputs, config, input_sizes, output_sizes);
        } else {
            throw std::runtime_error("Unsupported MPS kernel: " + kernel_name);
        }
        
        [commandBuffer commit];
        
        // Synchronize if no stream specified
        if (!config.stream) {
            [commandBuffer waitUntilCompleted];
        }
    }
}

size_t MpsBackend::get_available_memory() {
    if (!is_available_) return 0;
    
    @autoreleasepool {
        // Get current memory usage
        size_t recommended = [metal_device_ recommendedMaxWorkingSetSize];
        size_t current = [metal_device_ currentAllocatedSize];
        
        return recommended > current ? recommended - current : 0;
    }
}

// MPS kernel implementations (placeholder)
void MpsBackend::launch_mps_matmul(id<MTLCommandBuffer> commandBuffer,
                                  void** inputs, void** outputs,
                                  const KernelConfig& config,
                                  const std::vector<size_t>& input_sizes,
                                  const std::vector<size_t>& output_sizes) {
    // TODO: Implement MPS matrix multiplication using MPSMatrixMultiplication
    throw std::runtime_error("MPS matmul not yet implemented");
}

void MpsBackend::launch_mps_rms_norm(id<MTLCommandBuffer> commandBuffer,
                                    void** inputs, void** outputs,
                                    const KernelConfig& config,
                                    const std::vector<size_t>& input_sizes,
                                    const std::vector<size_t>& output_sizes) {
    // TODO: Implement MPS RMS normalization
    throw std::runtime_error("MPS RMS norm not yet implemented");
}

void MpsBackend::launch_mps_element_unary(id<MTLCommandBuffer> commandBuffer,
                                         void** inputs, void** outputs,
                                         const KernelConfig& config,
                                         const std::vector<size_t>& input_sizes,
                                         const std::vector<size_t>& output_sizes) {
    // TODO: Implement MPS element-wise unary operations
    throw std::runtime_error("MPS element unary not yet implemented");
}

#else // !__APPLE__

// Stub implementation for non-Apple platforms
MpsBackend::MpsBackend() : current_device_(-1), is_available_(false), profiling_active_(false) {
    throw std::runtime_error("MPS backend is only available on Apple platforms");
}

MpsBackend::~MpsBackend() {}

void MpsBackend::initialize_metal() {}

int MpsBackend::get_device_count() { return 0; }

void MpsBackend::set_device(int device_id) {
    throw std::runtime_error("MPS backend is not available on this platform");
}

DeviceInfo MpsBackend::get_device_info(int device_id) {
    throw std::runtime_error("MPS backend is not available on this platform");
}

void* MpsBackend::allocate(size_t size, MemoryType mem_type) {
    throw std::runtime_error("MPS backend is not available on this platform");
}

void MpsBackend::deallocate(void* ptr) {}

void MpsBackend::memcpy(void* dst, const void* src, size_t size,
                       MemoryType dst_type, MemoryType src_type) {
    throw std::runtime_error("MPS backend is not available on this platform");
}

void MpsBackend::memset(void* ptr, int value, size_t size) {
    throw std::runtime_error("MPS backend is not available on this platform");
}

void MpsBackend::synchronize() {}
void* MpsBackend::create_stream() { return nullptr; }
void MpsBackend::destroy_stream(void* stream) {}
void MpsBackend::synchronize_stream(void* stream) {}

void MpsBackend::start_profiling() {}
void MpsBackend::stop_profiling() {}
double MpsBackend::get_elapsed_time_ms() { return 0.0; }

void MpsBackend::launch_kernel(const std::string& kernel_name,
                             void** inputs, 
                             void** outputs,
                             const KernelConfig& config,
                             const std::vector<size_t>& input_sizes,
                             const std::vector<size_t>& output_sizes) {
    throw std::runtime_error("MPS backend is not available on this platform");
}

size_t MpsBackend::get_available_memory() { return 0; }

#endif // __APPLE__

} // namespace backend
} // namespace yirage
