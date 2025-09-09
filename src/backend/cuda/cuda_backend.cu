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

#include "yirage/backend/cuda/cuda_backend.h"
#include <stdexcept>
#include <iostream>
#include <sstream>

#ifdef YIRAGE_USE_CUDA
#include "yirage/utils/cuda_helper.h"
#include "yirage/kernel/device_memory_manager.h"
#endif

namespace yirage {
namespace backend {

#ifdef YIRAGE_USE_CUDA

CudaBackend::CudaBackend() 
    : current_device_(0), is_available_(false), profiling_active_(false) {
    initialize_cuda();
}

CudaBackend::~CudaBackend() {
    cleanup_cuda();
}

void CudaBackend::initialize_cuda() {
    try {
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            std::cerr << "No CUDA devices found or CUDA not available" << std::endl;
            return;
        }
        
        // Set default device
        checkCUDA(cudaSetDevice(0));
        current_device_ = 0;
        
        // Create CUDA events for profiling
        checkCUDA(cudaEventCreate(&start_event_));
        checkCUDA(cudaEventCreate(&end_event_));
        
        // Initialize cuBLAS
        cublasStatus_t stat = cublasCreate(&cublas_handle_);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
        
        is_available_ = true;
        
        // Print device info
        cudaDeviceProp prop;
        checkCUDA(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Initialized CUDA backend with device: " << prop.name << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize CUDA backend: " << e.what() << std::endl;
        is_available_ = false;
    }
}

void CudaBackend::cleanup_cuda() {
    if (!is_available_) return;
    
    try {
        // Clean up streams
        {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            for (auto& [ptr, stream] : stream_map_) {
                cudaStreamDestroy(stream);
            }
            stream_map_.clear();
        }
        
        // Destroy cuBLAS handle
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
        
        // Destroy CUDA events
        cudaEventDestroy(start_event_);
        cudaEventDestroy(end_event_);
        
    } catch (const std::exception& e) {
        std::cerr << "Error during CUDA cleanup: " << e.what() << std::endl;
    }
}

int CudaBackend::get_device_count() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess) ? device_count : 0;
}

void CudaBackend::set_device(int device_id) {
    if (!is_available_) {
        throw std::runtime_error("CUDA backend is not available");
    }
    
    int device_count = get_device_count();
    if (device_id < 0 || device_id >= device_count) {
        throw std::invalid_argument("Invalid CUDA device ID");
    }
    
    checkCUDA(cudaSetDevice(device_id));
    current_device_ = device_id;
    
    // Update cuBLAS stream
    cudaStream_t current_stream;
    checkCUDA(cudaStreamCreate(&current_stream));
    cublasSetStream(cublas_handle_, current_stream);
}

int CudaBackend::get_current_device() {
    int device_id;
    cudaError_t error = cudaGetDevice(&device_id);
    return (error == cudaSuccess) ? device_id : -1;
}

DeviceInfo CudaBackend::get_device_info(int device_id) {
    if (!is_available_) {
        throw std::runtime_error("CUDA backend is not available");
    }
    
    DeviceInfo info;
    info.device_id = device_id;
    info.backend_type = BackendType::CUDA;
    
    cudaDeviceProp prop;
    checkCUDA(cudaGetDeviceProperties(&prop, device_id));
    
    info.name = std::string(prop.name);
    info.total_memory = prop.totalGlobalMem;
    info.compute_units = prop.multiProcessorCount;
    
    // Get available memory
    size_t free_mem, total_mem;
    int original_device = get_current_device();
    checkCUDA(cudaSetDevice(device_id));
    checkCUDA(cudaMemGetInfo(&free_mem, &total_mem));
    checkCUDA(cudaSetDevice(original_device));
    
    info.available_memory = free_mem;
    
    return info;
}

void* CudaBackend::allocate(size_t size, MemoryType mem_type) {
    if (!is_available_) {
        throw std::runtime_error("CUDA backend is not available");
    }
    
    void* ptr = nullptr;
    
    switch (mem_type) {
        case MemoryType::DEVICE:
            checkCUDA(cudaMalloc(&ptr, size));
            break;
        case MemoryType::HOST:
            checkCUDA(cudaMallocHost(&ptr, size));
            break;
        case MemoryType::UNIFIED:
            checkCUDA(cudaMallocManaged(&ptr, size));
            break;
    }
    
    return ptr;
}

void CudaBackend::deallocate(void* ptr) {
    if (!ptr || !is_available_) return;
    
    // Try to determine memory type and free accordingly
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
    
    if (error == cudaSuccess) {
        switch (attributes.type) {
            case cudaMemoryTypeDevice:
            case cudaMemoryTypeManaged:
                cudaFree(ptr);
                break;
            case cudaMemoryTypeHost:
                cudaFreeHost(ptr);
                break;
            default:
                cudaFree(ptr); // Default to device free
                break;
        }
    } else {
        // If we can't determine the type, try device free
        cudaFree(ptr);
    }
}

void CudaBackend::memcpy(void* dst, const void* src, size_t size,
                        MemoryType dst_type, MemoryType src_type) {
    if (!is_available_) {
        throw std::runtime_error("CUDA backend is not available");
    }
    
    cudaMemcpyKind kind;
    
    if (src_type == MemoryType::HOST && dst_type == MemoryType::DEVICE) {
        kind = cudaMemcpyHostToDevice;
    } else if (src_type == MemoryType::DEVICE && dst_type == MemoryType::HOST) {
        kind = cudaMemcpyDeviceToHost;
    } else if (src_type == MemoryType::DEVICE && dst_type == MemoryType::DEVICE) {
        kind = cudaMemcpyDeviceToDevice;
    } else {
        kind = cudaMemcpyDefault; // Let CUDA runtime determine
    }
    
    checkCUDA(cudaMemcpy(dst, src, size, kind));
}

void CudaBackend::memset(void* ptr, int value, size_t size) {
    if (!is_available_) {
        throw std::runtime_error("CUDA backend is not available");
    }
    
    checkCUDA(cudaMemset(ptr, value, size));
}

void CudaBackend::synchronize() {
    if (!is_available_) return;
    checkCUDA(cudaDeviceSynchronize());
}

void* CudaBackend::create_stream() {
    if (!is_available_) {
        throw std::runtime_error("CUDA backend is not available");
    }
    
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    
    void* stream_ptr = static_cast<void*>(stream);
    
    {
        std::lock_guard<std::mutex> lock(stream_mutex_);
        stream_map_[stream_ptr] = stream;
    }
    
    return stream_ptr;
}

void CudaBackend::destroy_stream(void* stream) {
    if (!stream || !is_available_) return;
    
    std::lock_guard<std::mutex> lock(stream_mutex_);
    auto it = stream_map_.find(stream);
    if (it != stream_map_.end()) {
        cudaStreamDestroy(it->second);
        stream_map_.erase(it);
    }
}

void CudaBackend::synchronize_stream(void* stream) {
    if (!stream || !is_available_) return;
    
    std::lock_guard<std::mutex> lock(stream_mutex_);
    auto it = stream_map_.find(stream);
    if (it != stream_map_.end()) {
        checkCUDA(cudaStreamSynchronize(it->second));
    }
}

void CudaBackend::start_profiling() {
    if (!is_available_) return;
    
    checkCUDA(cudaEventRecord(start_event_));
    start_time_ = std::chrono::high_resolution_clock::now();
    profiling_active_ = true;
}

void CudaBackend::stop_profiling() {
    if (!is_available_) return;
    
    checkCUDA(cudaEventRecord(end_event_));
    checkCUDA(cudaEventSynchronize(end_event_));
    end_time_ = std::chrono::high_resolution_clock::now();
    profiling_active_ = false;
}

double CudaBackend::get_elapsed_time_ms() {
    if (!is_available_) return 0.0;
    
    if (profiling_active_) {
        // Use CPU timing for active profiling
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            current_time - start_time_);
        return duration.count() / 1000.0;
    } else {
        // Use CUDA events for completed profiling (more accurate)
        float cuda_time_ms;
        cudaError_t error = cudaEventElapsedTime(&cuda_time_ms, start_event_, end_event_);
        if (error == cudaSuccess) {
            return static_cast<double>(cuda_time_ms);
        } else {
            // Fallback to CPU timing
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time_ - start_time_);
            return duration.count() / 1000.0;
        }
    }
}

void CudaBackend::launch_kernel(const std::string& kernel_name,
                               void** inputs, 
                               void** outputs,
                               const KernelConfig& config,
                               const std::vector<size_t>& input_sizes,
                               const std::vector<size_t>& output_sizes) {
    if (!is_available_) {
        throw std::runtime_error("CUDA backend is not available");
    }
    
    // Get CUDA stream
    cudaStream_t cuda_stream = 0; // Default stream
    if (config.stream) {
        cuda_stream = get_cuda_stream(config.stream);
    }
    
    // Delegate to existing CUDA kernel implementations
    // This would integrate with the existing YiRage CUDA kernels
    throw std::runtime_error("CUDA kernel integration not yet implemented: " + kernel_name);
}

cudaStream_t CudaBackend::get_cuda_stream(void* stream) const {
    if (!stream) return 0; // Default stream
    
    std::lock_guard<std::mutex> lock(stream_mutex_);
    auto it = stream_map_.find(stream);
    return (it != stream_map_.end()) ? it->second : 0;
}

size_t CudaBackend::get_available_memory() {
    if (!is_available_) return 0;
    
    size_t free_mem, total_mem;
    cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
    return (error == cudaSuccess) ? free_mem : 0;
}

#else // !YIRAGE_USE_CUDA

// Stub implementation for builds without CUDA
CudaBackend::CudaBackend() 
    : current_device_(-1), is_available_(false), profiling_active_(false) {
    throw std::runtime_error("CUDA backend is not available (not compiled with CUDA support)");
}

CudaBackend::~CudaBackend() {}

void CudaBackend::initialize_cuda() {}
void CudaBackend::cleanup_cuda() {}

int CudaBackend::get_device_count() { return 0; }

void CudaBackend::set_device(int device_id) {
    throw std::runtime_error("CUDA backend is not available");
}

int CudaBackend::get_current_device() { return -1; }

DeviceInfo CudaBackend::get_device_info(int device_id) {
    throw std::runtime_error("CUDA backend is not available");
}

void* CudaBackend::allocate(size_t size, MemoryType mem_type) {
    throw std::runtime_error("CUDA backend is not available");
}

void CudaBackend::deallocate(void* ptr) {}

void CudaBackend::memcpy(void* dst, const void* src, size_t size,
                        MemoryType dst_type, MemoryType src_type) {
    throw std::runtime_error("CUDA backend is not available");
}

void CudaBackend::memset(void* ptr, int value, size_t size) {
    throw std::runtime_error("CUDA backend is not available");
}

void CudaBackend::synchronize() {}
void* CudaBackend::create_stream() { return nullptr; }
void CudaBackend::destroy_stream(void* stream) {}
void CudaBackend::synchronize_stream(void* stream) {}

void CudaBackend::start_profiling() {}
void CudaBackend::stop_profiling() {}
double CudaBackend::get_elapsed_time_ms() { return 0.0; }

void CudaBackend::launch_kernel(const std::string& kernel_name,
                               void** inputs, 
                               void** outputs,
                               const KernelConfig& config,
                               const std::vector<size_t>& input_sizes,
                               const std::vector<size_t>& output_sizes) {
    throw std::runtime_error("CUDA backend is not available");
}

size_t CudaBackend::get_available_memory() { return 0; }

#endif // YIRAGE_USE_CUDA

} // namespace backend
} // namespace yirage
