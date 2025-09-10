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

#include "yirage/backend/cpu/cpu_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <omp.h>

namespace yirage {
namespace backend {

// Helper functions for half precision conversion
uint16_t float_to_half(float f) {
    // Simple float to half conversion (not optimized)
    union { float f; uint32_t i; } u = { f };
    uint32_t i = u.i;
    int s = (i >> 31) & 0x1;
    int e = ((i >> 23) & 0xff) - 127 + 15;
    int m = i & 0x7fffff;

    if (e <= 0) return s << 15;
    if (e >= 31) return (s << 15) | 0x7c00;

    return (s << 15) | (e << 10) | (m >> 13);
}

float half_to_float(uint16_t h) {
    // Simple half to float conversion (not optimized)
    int s = (h >> 15) & 0x1;
    int e = (h >> 10) & 0x1f;
    int m = h & 0x3ff;

    if (e == 0) {
        if (m == 0) return s ? -0.0f : 0.0f;
        // Denormalized
        float f = m / 1024.0f;
        return s ? -f : f;
    }

    if (e == 31) {
        if (m == 0) return s ? -INFINITY : INFINITY;
        return NAN;
    }

    float f = (1.0f + m / 1024.0f) * std::pow(2.0f, e - 15);
    return s ? -f : f;
}

// Static instance for kernel dispatcher
CpuKernelInterface CpuKernelDispatcher::kernel_impl_;

CpuKernelInterface::CpuKernelInterface() {
    // Initialize OpenMP if available
    #ifdef _OPENMP
    omp_set_dynamic(0);  // Disable dynamic teams
    omp_set_num_threads(std::thread::hardware_concurrency());
    #endif
}

void CpuKernelInterface::matmul(void* output, const void* a, const void* b,
                               const MatmulConfig& config, yirage::type::DataType dtype) {
    // Simple CPU matrix multiplication implementation
    // This is a basic implementation - in production, you'd use optimized BLAS

    if (dtype == yirage::type::DT_FLOAT16) {
        // For half precision, we'll convert to float for computation
        const auto* a_ptr = static_cast<const uint16_t*>(a);
        const auto* b_ptr = static_cast<const uint16_t*>(b);
        auto* out_ptr = static_cast<uint16_t*>(output);

        int m = config.m, n = config.n, k = config.k;

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    // Convert half to float for computation
                    float a_val = half_to_float(a_ptr[i * k + l]);
                    float b_val = half_to_float(b_ptr[l * n + j]);
                    sum += a_val * b_val;
                }
                // Convert back to half
                out_ptr[i * n + j] = float_to_half(sum * config.alpha);
            }
        }
    } else if (dtype == yirage::type::DT_FLOAT32) {
        const auto* a_ptr = static_cast<const float*>(a);
        const auto* b_ptr = static_cast<const float*>(b);
        auto* out_ptr = static_cast<float*>(output);

        int m = config.m, n = config.n, k = config.k;

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    sum += a_ptr[i * k + l] * b_ptr[l * n + j];
                }
                out_ptr[i * n + j] = sum * config.alpha;
            }
        }
    } else {
        throw std::runtime_error("Unsupported data type for CPU matmul");
    }
}

void CpuKernelInterface::element_wise_unary(void* output, const void* input,
                                          yirage::type::KNOperatorType op_type,
                                          int num_elements,
                                          yirage::type::DataType dtype,
                                          void* stream) {
    if (dtype == yirage::type::DT_FLOAT16) {
        const auto* in_ptr = static_cast<const uint16_t*>(input);
        auto* out_ptr = static_cast<uint16_t*>(output);

        #pragma omp parallel for
        for (int i = 0; i < num_elements; ++i) {
            float val = half_to_float(in_ptr[i]);
            float result;

            switch (op_type) {
                case yirage::type::KN_EXP_OP:
                    result = std::exp(val);
                    break;
                case yirage::type::KN_SQRT_OP:
                    result = std::sqrt(val);
                    break;
                case yirage::type::KN_SILU_OP:
                    result = val / (1.0f + std::exp(-val));
                    break;
                case yirage::type::KN_GELU_OP:
                    result = 0.5f * val * (1.0f + std::erf(val / std::sqrt(2.0f)));
                    break;
                case yirage::type::KN_RELU_OP:
                    result = std::max(0.0f, val);
                    break;
                default:
                    throw std::runtime_error("Unsupported unary operation");
            }

            out_ptr[i] = float_to_half(result);
        }
    } else if (dtype == yirage::type::DT_FLOAT32) {
        const auto* in_ptr = static_cast<const float*>(input);
        auto* out_ptr = static_cast<float*>(output);

        #pragma omp parallel for
        for (int i = 0; i < num_elements; ++i) {
            float val = in_ptr[i];
            float result;

            switch (op_type) {
                case yirage::type::KN_EXP_OP:
                    result = std::exp(val);
                    break;
                case yirage::type::KN_SQRT_OP:
                    result = std::sqrt(val);
                    break;
                case yirage::type::KN_SILU_OP:
                    result = val / (1.0f + std::exp(-val));
                    break;
                case yirage::type::KN_GELU_OP:
                    result = 0.5f * val * (1.0f + std::erf(val / std::sqrt(2.0f)));
                    break;
                case yirage::type::KN_RELU_OP:
                    result = std::max(0.0f, val);
                    break;
                default:
                    throw std::runtime_error("Unsupported unary operation");
            }

            out_ptr[i] = result;
        }
    } else {
        throw std::runtime_error("Unsupported data type for CPU unary operation");
    }
}

void CpuKernelInterface::rms_norm(void* output, const void* input, const void* weight,
                                 int batch_size, int hidden_size, float eps,
                                 yirage::type::DataType dtype,
                                 void* stream) {
    if (dtype == yirage::type::DT_FLOAT16) {
        const auto* in_ptr = static_cast<const uint16_t*>(input);
        const auto* w_ptr = static_cast<const uint16_t*>(weight);
        auto* out_ptr = static_cast<uint16_t*>(output);

        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            // Compute RMS
            float sum_squares = 0.0f;
            for (int h = 0; h < hidden_size; ++h) {
                float val = half_to_float(in_ptr[b * hidden_size + h]);
                sum_squares += val * val;
            }
            float rms = std::sqrt(sum_squares / hidden_size + eps);

            // Apply normalization and weight
            for (int h = 0; h < hidden_size; ++h) {
                float val = half_to_float(in_ptr[b * hidden_size + h]);
                float w_val = half_to_float(w_ptr[h]);
                float result = (val / rms) * w_val;
                out_ptr[b * hidden_size + h] = float_to_half(result);
            }
        }
    } else if (dtype == yirage::type::DT_FLOAT32) {
        const auto* in_ptr = static_cast<const float*>(input);
        const auto* w_ptr = static_cast<const float*>(weight);
        auto* out_ptr = static_cast<float*>(output);

        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            // Compute RMS
            float sum_squares = 0.0f;
            for (int h = 0; h < hidden_size; ++h) {
                float val = in_ptr[b * hidden_size + h];
                sum_squares += val * val;
            }
            float rms = std::sqrt(sum_squares / hidden_size + eps);

            // Apply normalization and weight
            for (int h = 0; h < hidden_size; ++h) {
                float val = in_ptr[b * hidden_size + h];
                float w_val = w_ptr[h];
                out_ptr[b * hidden_size + h] = (val / rms) * w_val;
            }
        }
    } else {
        throw std::runtime_error("Unsupported data type for CPU RMS norm");
    }
}

// Placeholder implementations for other methods
void CpuKernelInterface::element_wise_binary(void* output, const void* a, const void* b,
                                           yirage::type::KNOperatorType op_type,
                                           int num_elements,
                                           yirage::type::DataType dtype,
                                           void* stream) {
    if (dtype == yirage::type::DT_FLOAT16) {
        const auto* a_ptr = static_cast<const uint16_t*>(a);
        const auto* b_ptr = static_cast<const uint16_t*>(b);
        auto* out_ptr = static_cast<uint16_t*>(output);

        #pragma omp parallel for
        for (int i = 0; i < num_elements; ++i) {
            float a_val = half_to_float(a_ptr[i]);
            float b_val = half_to_float(b_ptr[i]);
            float result;

            switch (op_type) {
                case yirage::type::KN_ADD_OP:
                    result = a_val + b_val;
                    break;
                case yirage::type::KN_MUL_OP:
                    result = a_val * b_val;
                    break;
                case yirage::type::KN_SUB_OP:
                    result = a_val - b_val;
                    break;
                case yirage::type::KN_DIV_OP:
                    result = a_val / b_val;
                    break;
                default:
                    throw std::runtime_error("Unsupported binary operation");
            }

            out_ptr[i] = float_to_half(result);
        }
    } else if (dtype == yirage::type::DT_FLOAT32) {
        const auto* a_ptr = static_cast<const float*>(a);
        const auto* b_ptr = static_cast<const float*>(b);
        auto* out_ptr = static_cast<float*>(output);

        #pragma omp parallel for
        for (int i = 0; i < num_elements; ++i) {
            float a_val = a_ptr[i];
            float b_val = b_ptr[i];
            float result;

            switch (op_type) {
                case yirage::type::KN_ADD_OP:
                    result = a_val + b_val;
                    break;
                case yirage::type::KN_MUL_OP:
                    result = a_val * b_val;
                    break;
                case yirage::type::KN_SUB_OP:
                    result = a_val - b_val;
                    break;
                case yirage::type::KN_DIV_OP:
                    result = a_val / b_val;
                    break;
                default:
                    throw std::runtime_error("Unsupported binary operation");
            }

            out_ptr[i] = result;
        }
    } else {
        throw std::runtime_error("Unsupported data type for CPU binary operation");
    }
}

void CpuKernelInterface::reduction(void* output, const void* input,
                                 yirage::type::KNOperatorType reduction_type,
                                 const std::vector<int>& input_dims,
                                 const std::vector<int>& reduce_dims,
                                 yirage::type::DataType dtype,
                                 void* stream) {
    // TODO: Implement reduction operations
    throw std::runtime_error("CPU reduction operations not yet implemented");
}

void CpuKernelInterface::layer_norm(void* output, const void* input,
                                  const void* weight, const void* bias,
                                  int batch_size, int hidden_size, float eps,
                                  yirage::type::DataType dtype,
                                  void* stream) {
    // TODO: Implement layer normalization
    throw std::runtime_error("CPU layer norm not yet implemented");
}

void CpuKernelInterface::attention(void* output, const void* query, const void* key,
                                 const void* value, void* k_cache, void* v_cache,
                                 const AttentionConfig& config,
                                 yirage::type::DataType dtype) {
    // TODO: Implement attention mechanism
    throw std::runtime_error("CPU attention not yet implemented");
}

void CpuKernelInterface::silu(void* output, const void* input, int num_elements,
                             yirage::type::DataType dtype, void* stream) {
    element_wise_unary(output, input, yirage::type::KN_SILU_OP, num_elements, dtype, stream);
}

void CpuKernelInterface::gelu(void* output, const void* input, int num_elements,
                             yirage::type::DataType dtype, void* stream) {
    element_wise_unary(output, input, yirage::type::KN_GELU_OP, num_elements, dtype, stream);
}

void CpuKernelInterface::relu(void* output, const void* input, int num_elements,
                             yirage::type::DataType dtype, void* stream) {
    element_wise_unary(output, input, yirage::type::KN_RELU_OP, num_elements, dtype, stream);
}

void CpuKernelInterface::copy(void* dst, const void* src, size_t size, void* stream) {
    std::memcpy(dst, src, size);
}

void CpuKernelInterface::fill(void* ptr, int value, size_t size, void* stream) {
    std::memset(ptr, value, size);
}

void CpuKernelInterface::embedding_lookup(void* output, const void* input,
                                        const void* weight,
                                        int batch_size, int seq_length,
                                        int vocab_size, int hidden_size,
                                        yirage::type::DataType dtype,
                                        void* stream) {
    // TODO: Implement embedding lookup
    throw std::runtime_error("CPU embedding lookup not yet implemented");
}

void CpuKernelInterface::argmax(void* output, const void* input,
                               int batch_size, int vocab_size,
                               yirage::type::DataType input_dtype,
                               void* stream) {
    auto* out_ptr = static_cast<int64_t*>(output);

    if (input_dtype == yirage::type::DT_FLOAT16) {
        const auto* in_ptr = static_cast<const uint16_t*>(input);

        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            int64_t max_idx = 0;
            float max_val = half_to_float(in_ptr[b * vocab_size]);

            for (int v = 1; v < vocab_size; ++v) {
                float val = half_to_float(in_ptr[b * vocab_size + v]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = v;
                }
            }

            out_ptr[b] = max_idx;
        }
    } else if (input_dtype == yirage::type::DT_FLOAT32) {
        const auto* in_ptr = static_cast<const float*>(input);

        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            int64_t max_idx = 0;
            float max_val = in_ptr[b * vocab_size];

            for (int v = 1; v < vocab_size; ++v) {
                float val = in_ptr[b * vocab_size + v];
                if (val > max_val) {
                    max_val = val;
                    max_idx = v;
                }
            }

            out_ptr[b] = max_idx;
        }
    } else {
        throw std::runtime_error("Unsupported data type for CPU argmax");
    }
}

void CpuKernelInterface::rms_norm_linear(void* output, const void* input,
                                       const void* norm_weight, const void* linear_weight,
                                       int batch_size, int input_size, int output_size,
                                       float eps, yirage::type::DataType dtype,
                                       void* stream) {
    // TODO: Implement fused RMS norm + linear
    throw std::runtime_error("CPU RMS norm linear not yet implemented");
}

void CpuKernelInterface::silu_mul_linear(void* output, const void* gate_input,
                                       const void* up_input, const void* linear_weight,
                                       int batch_size, int hidden_size, int intermediate_size,
                                       yirage::type::DataType dtype,
                                       void* stream) {
    // TODO: Implement fused SiLU multiply linear
    throw std::runtime_error("CPU SiLU mul linear not yet implemented");
}

// (Half precision functions already defined above)

// Kernel dispatcher implementation
void CpuKernelDispatcher::execute(const std::string& kernel_name,
                                void** inputs,
                                void** outputs,
                                const KernelConfig& config,
                                const std::vector<size_t>& input_sizes,
                                const std::vector<size_t>& output_sizes) {
    // This would dispatch to specific kernel implementations based on kernel_name
    // For now, just throw an error for unsupported kernels
    throw std::runtime_error("CPU kernel dispatcher not fully implemented: " + kernel_name);
}

} // namespace backend
} // namespace yirage
