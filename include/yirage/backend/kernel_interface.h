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
#include "yirage/type.h"
#include <vector>

namespace yirage {
namespace backend {

// Configuration for attention operations
struct AttentionConfig {
    int batch_size;
    int seq_length;
    int num_heads;
    int head_dim;
    int num_kv_heads = -1;  // For GQA, -1 means same as num_heads
    bool causal_mask = true;
    float scale = -1.0f;    // -1 means use 1/sqrt(head_dim)
    void* stream = nullptr;
};

// Configuration for matrix multiplication
struct MatmulConfig {
    int m, n, k;
    bool transpose_a = false;
    bool transpose_b = false;
    float alpha = 1.0f;
    float beta = 0.0f;
    void* stream = nullptr;
};

// Abstract interface for compute kernels
class KernelInterface {
public:
    virtual ~KernelInterface() = default;
    
    // Matrix operations
    virtual void matmul(void* output, const void* a, const void* b,
                       const MatmulConfig& config, yirage::type::DataType dtype) = 0;
    
    // Element-wise operations
    virtual void element_wise_unary(void* output, const void* input,
                                  yirage::type::KNOperatorType op_type, 
                                  int num_elements,
                                  yirage::type::DataType dtype,
                                  void* stream = nullptr) = 0;
    
    virtual void element_wise_binary(void* output, const void* a, const void* b,
                                   yirage::type::KNOperatorType op_type, 
                                   int num_elements,
                                   yirage::type::DataType dtype,
                                   void* stream = nullptr) = 0;
    
    // Reduction operations
    virtual void reduction(void* output, const void* input,
                         yirage::type::KNOperatorType reduction_type, 
                         const std::vector<int>& input_dims,
                         const std::vector<int>& reduce_dims,
                         yirage::type::DataType dtype,
                         void* stream = nullptr) = 0;
    
    // Normalization operations
    virtual void rms_norm(void* output, const void* input, const void* weight,
                         int batch_size, int hidden_size, float eps,
                         yirage::type::DataType dtype,
                         void* stream = nullptr) = 0;
    
    virtual void layer_norm(void* output, const void* input, 
                          const void* weight, const void* bias,
                          int batch_size, int hidden_size, float eps,
                          yirage::type::DataType dtype,
                          void* stream = nullptr) = 0;
    
    // Attention operations
    virtual void attention(void* output, const void* query, const void* key,
                         const void* value, void* k_cache, void* v_cache,
                         const AttentionConfig& config,
                         yirage::type::DataType dtype) = 0;
    
    // Activation functions
    virtual void silu(void* output, const void* input, int num_elements,
                     yirage::type::DataType dtype, void* stream = nullptr) = 0;
    
    virtual void gelu(void* output, const void* input, int num_elements,
                     yirage::type::DataType dtype, void* stream = nullptr) = 0;
    
    virtual void relu(void* output, const void* input, int num_elements,
                     yirage::type::DataType dtype, void* stream = nullptr) = 0;
    
    // Memory operations
    virtual void copy(void* dst, const void* src, size_t size,
                     void* stream = nullptr) = 0;
    
    virtual void fill(void* ptr, int value, size_t size,
                     void* stream = nullptr) = 0;
    
    // Embedding operations
    virtual void embedding_lookup(void* output, const void* input, 
                                const void* weight,
                                int batch_size, int seq_length, 
                                int vocab_size, int hidden_size,
                                yirage::type::DataType dtype,
                                void* stream = nullptr) = 0;
    
    // Argmax operations
    virtual void argmax(void* output, const void* input,
                       int batch_size, int vocab_size,
                       yirage::type::DataType input_dtype,
                       void* stream = nullptr) = 0;
    
    // Fused operations (commonly used combinations)
    virtual void rms_norm_linear(void* output, const void* input,
                               const void* norm_weight, const void* linear_weight,
                               int batch_size, int input_size, int output_size,
                               float eps, yirage::type::DataType dtype,
                               void* stream = nullptr) = 0;
    
    virtual void silu_mul_linear(void* output, const void* gate_input, 
                               const void* up_input, const void* linear_weight,
                               int batch_size, int hidden_size, int intermediate_size,
                               yirage::type::DataType dtype,
                               void* stream = nullptr) = 0;
};

// Factory for creating kernel implementations
class KernelFactory {
public:
    static std::unique_ptr<KernelInterface> create(BackendType backend_type);
};

} // namespace backend
} // namespace yirage
