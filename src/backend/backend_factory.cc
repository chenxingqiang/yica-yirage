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

#include "yirage/backend/backend_interface.h"
#include <stdexcept>
#include <cstdlib>

// Include backend implementations
#ifdef YIRAGE_USE_CUDA
#include "yirage/backend/cuda/cuda_backend.h"
#endif

#ifdef YIRAGE_USE_CPU
#include "yirage/backend/cpu/cpu_backend.h"
#endif

#ifdef YIRAGE_USE_MPS
#include "yirage/backend/mps/mps_backend.h"
#endif

namespace yirage {
namespace backend {

std::unique_ptr<BackendInterface> BackendFactory::create(BackendType type) {
    switch (type) {
#ifdef YIRAGE_USE_CUDA
        case BackendType::CUDA:
            return std::make_unique<CudaBackend>();
#endif
            
#ifdef YIRAGE_USE_CPU
        case BackendType::CPU:
            return std::make_unique<CpuBackend>();
#endif
            
#ifdef YIRAGE_USE_MPS
        case BackendType::MPS:
            return std::make_unique<MpsBackend>();
#endif
            
        case BackendType::AUTO:
            return create(detect_best_backend());
            
        default:
            throw std::runtime_error("Unsupported backend type");
    }
}

BackendType BackendFactory::detect_best_backend() {
    // Check environment variable first
    const char* env_backend = std::getenv("YIRAGE_BACKEND");
    if (env_backend) {
        std::string backend_str(env_backend);
        if (backend_str == "cuda" && is_backend_available(BackendType::CUDA)) {
            return BackendType::CUDA;
        } else if (backend_str == "cpu" && is_backend_available(BackendType::CPU)) {
            return BackendType::CPU;
        } else if (backend_str == "mps" && is_backend_available(BackendType::MPS)) {
            return BackendType::MPS;
        }
    }
    
    // Auto-detect in order of preference
    if (is_backend_available(BackendType::CUDA)) {
        return BackendType::CUDA;
    }
    
    if (is_backend_available(BackendType::MPS)) {
        return BackendType::MPS;
    }
    
    if (is_backend_available(BackendType::CPU)) {
        return BackendType::CPU;
    }
    
    throw std::runtime_error("No supported backend available");
}

std::vector<BackendType> BackendFactory::get_available_backends() {
    std::vector<BackendType> available;
    
#ifdef YIRAGE_USE_CUDA
    if (is_backend_available(BackendType::CUDA)) {
        available.push_back(BackendType::CUDA);
    }
#endif
    
#ifdef YIRAGE_USE_MPS
    if (is_backend_available(BackendType::MPS)) {
        available.push_back(BackendType::MPS);
    }
#endif
    
#ifdef YIRAGE_USE_CPU
    if (is_backend_available(BackendType::CPU)) {
        available.push_back(BackendType::CPU);
    }
#endif
    
    return available;
}

bool BackendFactory::is_backend_available(BackendType type) {
    switch (type) {
#ifdef YIRAGE_USE_CUDA
        case BackendType::CUDA: {
            // Check if CUDA devices are available
            try {
                auto backend = std::make_unique<CudaBackend>();
                return backend->get_device_count() > 0;
            } catch (...) {
                return false;
            }
        }
#endif
        
#ifdef YIRAGE_USE_CPU
        case BackendType::CPU:
            // CPU backend is always available when compiled
            return true;
#endif
            
#ifdef YIRAGE_USE_MPS
        case BackendType::MPS: {
            // Check if MPS is available (macOS with Metal support)
            try {
                auto backend = std::make_unique<MpsBackend>();
                return backend->get_device_count() > 0;
            } catch (...) {
                return false;
            }
        }
#endif
        
        default:
            return false;
    }
}

} // namespace backend
} // namespace yirage
