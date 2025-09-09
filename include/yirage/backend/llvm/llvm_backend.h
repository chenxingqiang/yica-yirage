// Copyright 2025-2026 YICA TEAM
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef YIRAGE_USE_LLVM

#include "yirage/backend/backend_interface.h"
#include "yirage/backend/kernel_interface.h"

// LLVM headers
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace yirage {
namespace backend {
namespace llvm_backend {

/**
 * Target information for LLVM backend
 */
struct TargetInfo {
    std::string triple;        // e.g., "x86_64-unknown-linux-gnu"
    std::string cpu;           // e.g., "skylake", "cortex-a77"
    std::string features;      // e.g., "+avx2,+fma", "+neon"
    llvm::CodeGenOpt::Level opt_level = llvm::CodeGenOpt::Aggressive;
    
    // Auto-detect host target
    static TargetInfo detect_host();
    
    // Check if target is available
    bool is_available() const;
    
    std::string to_string() const;
};

/**
 * LLVM Backend Manager
 * Handles LLVM initialization and backend creation
 */
class LLVMBackendManager {
public:
    static std::unique_ptr<class LLVMBackend> create_backend(const TargetInfo& target);
    static std::vector<TargetInfo> get_available_targets();
    static bool initialize_llvm();
    
private:
    static bool llvm_initialized_;
    static void register_all_targets();
};

/**
 * LLVM Kernel Interface
 * Implements kernel operations using LLVM JIT compilation
 */
class LLVMKernelInterface : public KernelInterface {
public:
    explicit LLVMKernelInterface(class LLVMBackend* backend);
    ~LLVMKernelInterface() override;
    
    // Kernel operations
    void matmul(void* output, const void* a, const void* b,
                int m, int n, int k,
                yirage::type::DataType dtype,
                void* stream) override;
    
    void rms_norm(void* output, const void* input, const void* weight,
                  int batch_size, int hidden_size, float eps,
                  yirage::type::DataType dtype,
                  void* stream) override;
    
    void element_wise_unary(void* output, const void* input,
                           yirage::type::KNOperatorType op_type,
                           int num_elements,
                           yirage::type::DataType dtype,
                           void* stream) override;
    
    void element_wise_binary(void* output, const void* a, const void* b,
                            yirage::type::KNOperatorType op_type,
                            int num_elements,
                            yirage::type::DataType dtype,
                            void* stream) override;
    
    void argmax(void* output, const void* input,
                int batch_size, int vocab_size,
                yirage::type::DataType input_dtype,
                void* stream) override;

private:
    class LLVMBackend* backend_;
    
    // Compiled function cache
    using CompiledFunction = void(*)(void**);
    std::unordered_map<std::string, CompiledFunction> function_cache_;
    
    CompiledFunction get_or_compile_function(const std::string& kernel_name,
                                           std::function<llvm::Function*()> generator);
    std::string generate_kernel_key(const std::string& base_name,
                                   const std::vector<int>& shape_params,
                                   yirage::type::DataType dtype);
};

/**
 * LLVM Backend
 * Main backend implementation using LLVM JIT compilation
 */
class LLVMBackend : public BackendInterface {
public:
    explicit LLVMBackend(const TargetInfo& target);
    ~LLVMBackend() override;
    
    // BackendInterface implementation
    void* allocate_memory(size_t size, size_t alignment = 32) override;
    void free_memory(void* ptr) override;
    std::unique_ptr<KernelInterface> get_kernel_interface() override;
    size_t get_available_memory() const override;
    BackendType get_backend_type() const override { return BackendType::LLVM; }
    
    // LLVM-specific methods
    llvm::LLVMContext& get_context() { return *context_; }
    llvm::Module& get_module() { return *module_; }
    llvm::IRBuilder<>& get_builder() { return *builder_; }
    
    // Compilation and execution
    using CompiledFunction = void(*)(void**);
    CompiledFunction compile_function(llvm::Function* func);
    
    // IR generation helpers
    llvm::Type* get_llvm_type(yirage::type::DataType dt);
    llvm::Value* create_loop(llvm::Value* start, llvm::Value* end, llvm::Value* step,
                            std::function<void(llvm::Value*)> body);
    
    const TargetInfo& get_target_info() const { return target_info_; }

private:
    TargetInfo target_info_;
    
    // LLVM components
    std::unique_ptr<llvm::LLVMContext> context_;
    std::unique_ptr<llvm::Module> module_;
    std::unique_ptr<llvm::IRBuilder<>> builder_;
    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::unique_ptr<llvm::TargetMachine> target_machine_;
    
    // Memory management
    std::vector<void*> allocated_memory_;
    
    // Function compilation cache
    std::unordered_map<std::string, CompiledFunction> compiled_functions_;
    
    void initialize_llvm_components();
    void setup_target_machine();
    void setup_optimization_pipeline();
    
    std::string generate_function_hash(llvm::Function* func);
    
    friend class LLVMKernelInterface;
};

/**
 * IR Translation utilities
 */
class IRTranslator {
public:
    explicit IRTranslator(LLVMBackend& backend);
    
    // Generate matrix multiplication kernel
    llvm::Function* generate_matmul_kernel(int m, int n, int k, yirage::type::DataType dtype);
    
    // Generate RMS normalization kernel
    llvm::Function* generate_rms_norm_kernel(int batch_size, int hidden_size, 
                                           yirage::type::DataType dtype);
    
    // Generate element-wise operation kernel
    llvm::Function* generate_element_wise_kernel(yirage::type::KNOperatorType op_type,
                                               int num_elements,
                                               yirage::type::DataType dtype,
                                               bool is_binary = false);
    
    // Generate argmax kernel
    llvm::Function* generate_argmax_kernel(int batch_size, int vocab_size,
                                         yirage::type::DataType dtype);

private:
    LLVMBackend& backend_;
    
    struct LoopNest {
        std::vector<llvm::Value*> indices;
        std::vector<llvm::BasicBlock*> headers;
        std::vector<llvm::BasicBlock*> bodies;
        std::vector<llvm::BasicBlock*> exits;
    };
    
    LoopNest create_loop_nest(const std::vector<llvm::Value*>& bounds);
    llvm::Value* create_tensor_access(llvm::Value* base_ptr,
                                     const std::vector<llvm::Value*>& indices,
                                     const std::vector<int>& strides,
                                     llvm::Type* element_type);
    
    llvm::Value* generate_unary_op(llvm::Value* input, yirage::type::KNOperatorType op_type);
    llvm::Value* generate_binary_op(llvm::Value* a, llvm::Value* b, 
                                   yirage::type::KNOperatorType op_type);
};

/**
 * Hardware-specific optimizers
 */
class HardwareOptimizer {
public:
    virtual ~HardwareOptimizer() = default;
    virtual void optimize_module(llvm::Module& module) = 0;
    virtual void setup_passes(llvm::legacy::PassManager& pm) = 0;
    
    static std::unique_ptr<HardwareOptimizer> create_for_target(const TargetInfo& target);
};

class X86Optimizer : public HardwareOptimizer {
public:
    void optimize_module(llvm::Module& module) override;
    void setup_passes(llvm::legacy::PassManager& pm) override;
    
private:
    void enable_vectorization(llvm::Module& module);
    void optimize_cache_behavior(llvm::Module& module);
};

class ARMOptimizer : public HardwareOptimizer {
public:
    void optimize_module(llvm::Module& module) override;
    void setup_passes(llvm::legacy::PassManager& pm) override;
    
private:
    void enable_neon_vectorization(llvm::Module& module);
    void optimize_for_mobile(llvm::Module& module);
};

class RISCVOptimizer : public HardwareOptimizer {
public:
    void optimize_module(llvm::Module& module) override;
    void setup_passes(llvm::legacy::PassManager& pm) override;
    
private:
    void enable_rvv_vectorization(llvm::Module& module);
    void optimize_riscv_specific(llvm::Module& module);
};

} // namespace llvm_backend

// Convenience typedefs
using LLVMBackend = llvm_backend::LLVMBackend;
using LLVMTargetInfo = llvm_backend::TargetInfo;

} // namespace backend
} // namespace yirage

#endif // YIRAGE_USE_LLVM
