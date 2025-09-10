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

#include "yirage/cpu/cpu_compatibility.h"

#ifdef YIRAGE_CPU_ONLY

namespace yirage {
namespace cpu {
namespace detail {

// Thread-local variables for CUDA compatibility
thread_local dim3 blockIdx{0, 0, 0};
thread_local dim3 blockDim{1, 1, 1};
thread_local dim3 threadIdx{0, 0, 0};
thread_local dim3 gridDim{1, 1, 1};

} // namespace detail
} // namespace cpu
} // namespace yirage

#endif // YIRAGE_CPU_ONLY
