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

#include "yirage/backend/kernel_interface.h"
#include <stdexcept>

// Include backend-specific kernel implementations
#ifdef YIRAGE_USE_CPU
#include "yirage/backend/cpu/cpu_kernels.h"
#endif

#ifdef YIRAGE_USE_CUDA
#include "yirage/backend/cuda/cuda_kernels.h"
#endif

#ifdef YIRAGE_USE_MPS
#include "yirage/backend/mps/mps_kernels.h"
#endif

namespace yirage {
namespace backend {

std::unique_ptr<KernelInterface> KernelFactory::create(BackendType backend_type) {
    switch (backend_type) {
#ifdef YIRAGE_USE_CPU
        case BackendType::CPU:
            return std::make_unique<CpuKernelInterface>();
#endif
            
#ifdef YIRAGE_USE_CUDA
        case BackendType::CUDA:
            return std::make_unique<CudaKernelInterface>();
#endif
            
#ifdef YIRAGE_USE_MPS
        case BackendType::MPS:
            return std::make_unique<MpsKernelInterface>();
#endif
            
        default:
            throw std::runtime_error("Unsupported backend type for kernel creation");
    }
}

} // namespace backend
} // namespace yirage
