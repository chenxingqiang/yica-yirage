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

#include "yirage/backend/kernel_interface.h"
#include <string>
#include <vector>

namespace yirage {
namespace backend {

class CpuKernelInterface : public KernelInterface {
public:
    CpuKernelInterface();
    virtual ~CpuKernelInterface() = default;
    
    // Matrix operations
    void matmul(void* output, const void* a, const void* b,
               const MatmulConfig& config, yirage::type::DataType dtype) override;
    
    // Element-wise operations
    void element_wise_unary(void* output, const void* input,
                          yirage::type::KNOperatorType op_type, 
                          int num_elements,
                          yirage::type::DataType dtype,
                          void* stream = nullptr) override;
    
    void element_wise_binary(void* output, const void* a, const void* b,
                           yirage::type::KNOperatorType op_type, 
                           int num_elements,
                           yirage::type::DataType dtype,
                           void* stream = nullptr) override;
    
    // Reduction operations
    void reduction(void* output, const void* input,
                 yirage::type::KNOperatorType reduction_type, 
                 const std::vector<int>& input_dims,
                 const std::vector<int>& reduce_dims,
                 yirage::type::DataType dtype,
                 void* stream = nullptr) override;
    
    // Normalization operations
    void rms_norm(void* output, const void* input, const void* weight,
                 int batch_size, int hidden_size, float eps,
                 yirage::type::DataType dtype,
                 void* stream = nullptr) override;
    
    void layer_norm(void* output, const void* input, 
                  const void* weight, const void* bias,
                  int batch_size, int hidden_size, float eps,
                  yirage::type::DataType dtype,
                  void* stream = nullptr) override;
    
    // Attention operations
    void attention(void* output, const void* query, const void* key,
                 const void* value, void* k_cache, void* v_cache,
                 const AttentionConfig& config,
                 yirage::type::DataType dtype) override;
    
    // Activation functions
    void silu(void* output, const void* input, int num_elements,
             yirage::type::DataType dtype, void* stream = nullptr) override;
    
    void gelu(void* output, const void* input, int num_elements,
             yirage::type::DataType dtype, void* stream = nullptr) override;
    
    void relu(void* output, const void* input, int num_elements,
             yirage::type::DataType dtype, void* stream = nullptr) override;
    
    // Memory operations
    void copy(void* dst, const void* src, size_t size,
             void* stream = nullptr) override;
    
    void fill(void* ptr, int value, size_t size,
             void* stream = nullptr) override;
    
    // Embedding operations
    void embedding_lookup(void* output, const void* input, 
                        const void* weight,
                        int batch_size, int seq_length, 
                        int vocab_size, int hidden_size,
                        yirage::type::DataType dtype,
                        void* stream = nullptr) override;
    
    // Argmax operations
    void argmax(void* output, const void* input,
               int batch_size, int vocab_size,
               yirage::type::DataType input_dtype,
               void* stream = nullptr) override;
    
    // Fused operations
    void rms_norm_linear(void* output, const void* input,
                       const void* norm_weight, const void* linear_weight,
                       int batch_size, int input_size, int output_size,
                       float eps, yirage::type::DataType dtype,
                       void* stream = nullptr) override;
    
    void silu_mul_linear(void* output, const void* gate_input, 
                       const void* up_input, const void* linear_weight,
                       int batch_size, int hidden_size, int intermediate_size,
                       yirage::type::DataType dtype,
                       void* stream = nullptr) override;
};

// Kernel dispatcher for string-based kernel execution
class CpuKernelDispatcher {
public:
    static void execute(const std::string& kernel_name,
                       void** inputs, 
                       void** outputs,
                       const KernelConfig& config,
                       const std::vector<size_t>& input_sizes,
                       const std::vector<size_t>& output_sizes);
                       
private:
    static CpuKernelInterface kernel_impl_;
};

} // namespace backend
} // namespace yirage
