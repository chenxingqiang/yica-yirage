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

#ifdef YIRAGE_USE_LLVM

#include "yirage/backend/llvm/llvm_backend.h"
#include "yirage/base/data_type.h"

#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Host.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutorProcessControl.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>

#include <iostream>
#include <cstring>
#include <cstdlib>

namespace yirage {
namespace backend {
namespace llvm_backend {

// Static member initialization
bool LLVMBackendManager::llvm_initialized_ = false;

//===----------------------------------------------------------------------===//
// TargetInfo Implementation
//===----------------------------------------------------------------------===//

TargetInfo TargetInfo::detect_host() {
    TargetInfo info;
    info.triple = llvm::sys::getDefaultTargetTriple();
    info.cpu = llvm::sys::getHostCPUName().str();
    
    // Detect host features
    llvm::StringMap<bool> host_features;
    if (llvm::sys::getHostCPUFeatures(host_features)) {
        std::vector<std::string> features;
        for (const auto& feature : host_features) {
            if (feature.second) {
                features.push_back("+" + feature.first().str());
            }
        }
        
        // Join features with commas
        std::string feature_str;
        for (size_t i = 0; i < features.size(); ++i) {
            if (i > 0) feature_str += ",";
            feature_str += features[i];
        }
        info.features = feature_str;
    }
    
    return info;
}

bool TargetInfo::is_available() const {
    std::string error;
    const auto* target = llvm::TargetRegistry::lookupTarget(triple, error);
    return target != nullptr;
}

std::string TargetInfo::to_string() const {
    return "TargetInfo{triple=" + triple + ", cpu=" + cpu + ", features=" + features + "}";
}

//===----------------------------------------------------------------------===//
// LLVMBackendManager Implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<LLVMBackend> LLVMBackendManager::create_backend(const TargetInfo& target) {
    if (!initialize_llvm()) {
        throw std::runtime_error("Failed to initialize LLVM");
    }
    
    if (!target.is_available()) {
        throw std::runtime_error("Target not available: " + target.triple);
    }
    
    return std::make_unique<LLVMBackend>(target);
}

std::vector<TargetInfo> LLVMBackendManager::get_available_targets() {
    if (!initialize_llvm()) {
        return {};
    }
    
    std::vector<TargetInfo> targets;
    
    // Add common targets that are likely to be available
    std::vector<std::tuple<std::string, std::string, std::string>> common_targets = {
        {"x86_64-unknown-linux-gnu", "skylake", "+avx2,+fma"},
        {"x86_64-unknown-linux-gnu", "haswell", "+avx2"},
        {"x86_64-pc-windows-msvc", "skylake", "+avx2,+fma"},
        {"x86_64-apple-darwin", "skylake", "+avx2,+fma"},
        {"aarch64-unknown-linux-gnu", "cortex-a77", "+neon"},
        {"aarch64-apple-darwin", "apple-a14", "+neon"},
        {"riscv64-unknown-linux-gnu", "generic", ""},
    };
    
    for (const auto& [triple, cpu, features] : common_targets) {
        TargetInfo info;
        info.triple = triple;
        info.cpu = cpu;
        info.features = features;
        
        if (info.is_available()) {
            targets.push_back(info);
        }
    }
    
    return targets;
}

bool LLVMBackendManager::initialize_llvm() {
    if (llvm_initialized_) {
        return true;
    }
    
    try {
        register_all_targets();
        llvm_initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize LLVM: " << e.what() << std::endl;
        return false;
    }
}

void LLVMBackendManager::register_all_targets() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
}

//===----------------------------------------------------------------------===//
// LLVMBackend Implementation
//===----------------------------------------------------------------------===//

LLVMBackend::LLVMBackend(const TargetInfo& target) : target_info_(target) {
    initialize_llvm_components();
    setup_target_machine();
    setup_optimization_pipeline();
}

LLVMBackend::~LLVMBackend() {
    // Free allocated memory
    for (void* ptr : allocated_memory_) {
        std::free(ptr);
    }
}

void* LLVMBackend::allocate_memory(size_t size, size_t alignment) {
    void* ptr = std::aligned_alloc(alignment, size);
    if (!ptr) {
        throw std::bad_alloc();
    }
    allocated_memory_.push_back(ptr);
    return ptr;
}

void LLVMBackend::free_memory(void* ptr) {
    auto it = std::find(allocated_memory_.begin(), allocated_memory_.end(), ptr);
    if (it != allocated_memory_.end()) {
        std::free(ptr);
        allocated_memory_.erase(it);
    }
}

std::unique_ptr<KernelInterface> LLVMBackend::get_kernel_interface() {
    return std::make_unique<LLVMKernelInterface>(this);
}

size_t LLVMBackend::get_available_memory() const {
    // Return a large value for LLVM backend as it uses system memory
    return 1ULL << 40; // 1TB
}

void LLVMBackend::initialize_llvm_components() {
    context_ = std::make_unique<llvm::LLVMContext>();
    module_ = std::make_unique<llvm::Module>("yirage_llvm_module", *context_);
    builder_ = std::make_unique<llvm::IRBuilder<>>(*context_);
    
    // Set target triple for the module
    module_->setTargetTriple(target_info_.triple);
}

void LLVMBackend::setup_target_machine() {
    std::string error;
    const auto* target = llvm::TargetRegistry::lookupTarget(target_info_.triple, error);
    if (!target) {
        throw std::runtime_error("Failed to lookup target: " + error);
    }
    
    llvm::TargetOptions target_options;
    target_options.UnsafeFPMath = true; // Enable fast math for performance
    
    target_machine_ = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(
            target_info_.triple,
            target_info_.cpu,
            target_info_.features,
            target_options,
            llvm::Reloc::PIC_,
            llvm::CodeModel::Small,
            target_info_.opt_level
        )
    );
    
    if (!target_machine_) {
        throw std::runtime_error("Failed to create target machine");
    }
    
    // Set data layout for the module
    module_->setDataLayout(target_machine_->createDataLayout());
}

void LLVMBackend::setup_optimization_pipeline() {
    // Create JIT with target machine
    auto jtmb = llvm::orc::JITTargetMachineBuilder(*target_machine_);
    auto jit_or_err = llvm::orc::LLJITBuilder().setJITTargetMachineBuilder(std::move(jtmb)).create();
    
    if (!jit_or_err) {
        throw std::runtime_error("Failed to create LLJIT: " + 
                                llvm::toString(jit_or_err.takeError()));
    }
    
    jit_ = std::move(jit_or_err.get());
}

LLVMBackend::CompiledFunction LLVMBackend::compile_function(llvm::Function* func) {
    if (!func) {
        throw std::invalid_argument("Function is null");
    }
    
    // Generate hash for caching
    std::string func_hash = generate_function_hash(func);
    
    // Check cache
    auto it = compiled_functions_.find(func_hash);
    if (it != compiled_functions_.end()) {
        return it->second;
    }
    
    // Verify function
    if (llvm::verifyFunction(*func, &llvm::errs())) {
        throw std::runtime_error("Function verification failed");
    }
    
    // Create thread-safe module
    auto tsm = llvm::orc::ThreadSafeModule(
        llvm::CloneModule(*module_), 
        std::make_shared<llvm::orc::ThreadSafeContext>(std::move(*context_))
    );
    
    // Add module to JIT
    auto err = jit_->addIRModule(std::move(tsm));
    if (err) {
        throw std::runtime_error("Failed to add module to JIT: " + 
                               llvm::toString(std::move(err)));
    }
    
    // Lookup function
    auto sym = jit_->lookup(func->getName());
    if (!sym) {
        throw std::runtime_error("Failed to lookup function: " + 
                               llvm::toString(sym.takeError()));
    }
    
    // Cast to function pointer
    auto compiled_func = reinterpret_cast<CompiledFunction>(sym->getAddress());
    
    // Cache the result
    compiled_functions_[func_hash] = compiled_func;
    
    return compiled_func;
}

llvm::Type* LLVMBackend::get_llvm_type(yirage::type::DataType dt) {
    switch (dt) {
        case yirage::type::DT_FLOAT32:
            return llvm::Type::getFloatTy(*context_);
        case yirage::type::DT_FLOAT16:
            return llvm::Type::getHalfTy(*context_);
        case yirage::type::DT_INT32:
            return llvm::Type::getInt32Ty(*context_);
        case yirage::type::DT_INT8:
            return llvm::Type::getInt8Ty(*context_);
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

std::string LLVMBackend::generate_function_hash(llvm::Function* func) {
    // Simple hash based on function name and module
    // In a real implementation, you'd want a more robust hash
    std::hash<std::string> hasher;
    return std::to_string(hasher(func->getName().str()));
}

//===----------------------------------------------------------------------===//
// LLVMKernelInterface Implementation
//===----------------------------------------------------------------------===//

LLVMKernelInterface::LLVMKernelInterface(LLVMBackend* backend) : backend_(backend) {}

LLVMKernelInterface::~LLVMKernelInterface() = default;

void LLVMKernelInterface::matmul(void* output, const void* a, const void* b,
                                int m, int n, int k,
                                yirage::type::DataType dtype,
                                void* stream) {
    std::string kernel_key = generate_kernel_key("matmul", {m, n, k}, dtype);
    
    auto compiled_func = get_or_compile_function(kernel_key, [=]() {
        IRTranslator translator(*backend_);
        return translator.generate_matmul_kernel(m, n, k, dtype);
    });
    
    void* args[] = {const_cast<void*>(a), const_cast<void*>(b), output};
    compiled_func(args);
}

void LLVMKernelInterface::rms_norm(void* output, const void* input, const void* weight,
                                  int batch_size, int hidden_size, float eps,
                                  yirage::type::DataType dtype,
                                  void* stream) {
    std::string kernel_key = generate_kernel_key("rms_norm", {batch_size, hidden_size}, dtype);
    
    auto compiled_func = get_or_compile_function(kernel_key, [=]() {
        IRTranslator translator(*backend_);
        return translator.generate_rms_norm_kernel(batch_size, hidden_size, dtype);
    });
    
    void* args[] = {const_cast<void*>(input), const_cast<void*>(weight), output, &eps};
    compiled_func(args);
}

void LLVMKernelInterface::element_wise_unary(void* output, const void* input,
                                           yirage::type::KNOperatorType op_type,
                                           int num_elements,
                                           yirage::type::DataType dtype,
                                           void* stream) {
    std::string kernel_key = generate_kernel_key("unary_" + std::to_string(static_cast<int>(op_type)), 
                                                {num_elements}, dtype);
    
    auto compiled_func = get_or_compile_function(kernel_key, [=]() {
        IRTranslator translator(*backend_);
        return translator.generate_element_wise_kernel(op_type, num_elements, dtype, false);
    });
    
    void* args[] = {const_cast<void*>(input), output};
    compiled_func(args);
}

void LLVMKernelInterface::element_wise_binary(void* output, const void* a, const void* b,
                                            yirage::type::KNOperatorType op_type,
                                            int num_elements,
                                            yirage::type::DataType dtype,
                                            void* stream) {
    std::string kernel_key = generate_kernel_key("binary_" + std::to_string(static_cast<int>(op_type)), 
                                                {num_elements}, dtype);
    
    auto compiled_func = get_or_compile_function(kernel_key, [=]() {
        IRTranslator translator(*backend_);
        return translator.generate_element_wise_kernel(op_type, num_elements, dtype, true);
    });
    
    void* args[] = {const_cast<void*>(a), const_cast<void*>(b), output};
    compiled_func(args);
}

void LLVMKernelInterface::argmax(void* output, const void* input,
                                int batch_size, int vocab_size,
                                yirage::type::DataType input_dtype,
                                void* stream) {
    std::string kernel_key = generate_kernel_key("argmax", {batch_size, vocab_size}, input_dtype);
    
    auto compiled_func = get_or_compile_function(kernel_key, [=]() {
        IRTranslator translator(*backend_);
        return translator.generate_argmax_kernel(batch_size, vocab_size, input_dtype);
    });
    
    void* args[] = {const_cast<void*>(input), output};
    compiled_func(args);
}

LLVMKernelInterface::CompiledFunction 
LLVMKernelInterface::get_or_compile_function(const std::string& kernel_name,
                                           std::function<llvm::Function*()> generator) {
    auto it = function_cache_.find(kernel_name);
    if (it != function_cache_.end()) {
        return it->second;
    }
    
    auto func = generator();
    if (!func) {
        throw std::runtime_error("Failed to generate function: " + kernel_name);
    }
    
    auto compiled_func = backend_->compile_function(func);
    function_cache_[kernel_name] = compiled_func;
    
    return compiled_func;
}

std::string LLVMKernelInterface::generate_kernel_key(const std::string& base_name,
                                                   const std::vector<int>& shape_params,
                                                   yirage::type::DataType dtype) {
    std::string key = base_name + "_" + std::to_string(static_cast<int>(dtype));
    for (int param : shape_params) {
        key += "_" + std::to_string(param);
    }
    return key;
}

//===----------------------------------------------------------------------===//
// IRTranslator Implementation (Basic)
//===----------------------------------------------------------------------===//

IRTranslator::IRTranslator(LLVMBackend& backend) : backend_(backend) {}

llvm::Function* IRTranslator::generate_matmul_kernel(int m, int n, int k, 
                                                    yirage::type::DataType dtype) {
    auto& context = backend_.get_context();
    auto& module = backend_.get_module();
    auto& builder = backend_.get_builder();
    
    // Create function signature: void matmul(float* a, float* b, float* c)
    auto element_type = backend_.get_llvm_type(dtype);
    auto ptr_type = llvm::PointerType::get(element_type, 0);
    
    auto func_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context),
        {ptr_type, ptr_type, ptr_type},
        false
    );
    
    auto func = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        "yirage_matmul_" + std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k),
        module
    );
    
    // Set function arguments
    auto args = func->arg_begin();
    auto a_ptr = &*args++;
    auto b_ptr = &*args++;
    auto c_ptr = &*args++;
    
    a_ptr->setName("a");
    b_ptr->setName("b");
    c_ptr->setName("c");
    
    // Create basic block
    auto entry_bb = llvm::BasicBlock::Create(context, "entry", func);
    builder.SetInsertPoint(entry_bb);
    
    // Generate simple triple loop for matrix multiplication
    // This is a basic implementation - real implementation would include optimizations
    auto i32_type = llvm::Type::getInt32Ty(context);
    auto zero = llvm::ConstantInt::get(i32_type, 0);
    auto one = llvm::ConstantInt::get(i32_type, 1);
    auto m_const = llvm::ConstantInt::get(i32_type, m);
    auto n_const = llvm::ConstantInt::get(i32_type, n);
    auto k_const = llvm::ConstantInt::get(i32_type, k);
    
    // TODO: Implement the actual loop generation
    // For now, just return void
    builder.CreateRetVoid();
    
    return func;
}

llvm::Function* IRTranslator::generate_rms_norm_kernel(int batch_size, int hidden_size, 
                                                     yirage::type::DataType dtype) {
    // Placeholder implementation
    auto& context = backend_.get_context();
    auto& module = backend_.get_module();
    
    auto element_type = backend_.get_llvm_type(dtype);
    auto ptr_type = llvm::PointerType::get(element_type, 0);
    auto float_type = llvm::Type::getFloatTy(context);
    auto float_ptr_type = llvm::PointerType::get(float_type, 0);
    
    auto func_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context),
        {ptr_type, ptr_type, ptr_type, float_ptr_type}, // input, weight, output, eps
        false
    );
    
    auto func = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        "yirage_rms_norm_" + std::to_string(batch_size) + "_" + std::to_string(hidden_size),
        module
    );
    
    auto entry_bb = llvm::BasicBlock::Create(context, "entry", func);
    auto& builder = backend_.get_builder();
    builder.SetInsertPoint(entry_bb);
    builder.CreateRetVoid();
    
    return func;
}

llvm::Function* IRTranslator::generate_element_wise_kernel(yirage::type::KNOperatorType op_type,
                                                         int num_elements,
                                                         yirage::type::DataType dtype,
                                                         bool is_binary) {
    // Placeholder implementation
    auto& context = backend_.get_context();
    auto& module = backend_.get_module();
    
    auto element_type = backend_.get_llvm_type(dtype);
    auto ptr_type = llvm::PointerType::get(element_type, 0);
    
    std::vector<llvm::Type*> arg_types = {ptr_type}; // input
    if (is_binary) {
        arg_types.push_back(ptr_type); // second input
    }
    arg_types.push_back(ptr_type); // output
    
    auto func_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context),
        arg_types,
        false
    );
    
    std::string func_name = "yirage_element_wise_" + 
                           std::to_string(static_cast<int>(op_type)) + "_" +
                           std::to_string(num_elements);
    
    auto func = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        func_name,
        module
    );
    
    auto entry_bb = llvm::BasicBlock::Create(context, "entry", func);
    auto& builder = backend_.get_builder();
    builder.SetInsertPoint(entry_bb);
    builder.CreateRetVoid();
    
    return func;
}

llvm::Function* IRTranslator::generate_argmax_kernel(int batch_size, int vocab_size,
                                                   yirage::type::DataType dtype) {
    // Placeholder implementation
    auto& context = backend_.get_context();
    auto& module = backend_.get_module();
    
    auto input_type = backend_.get_llvm_type(dtype);
    auto input_ptr_type = llvm::PointerType::get(input_type, 0);
    auto output_ptr_type = llvm::PointerType::get(llvm::Type::getInt32Ty(context), 0);
    
    auto func_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(context),
        {input_ptr_type, output_ptr_type},
        false
    );
    
    auto func = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        "yirage_argmax_" + std::to_string(batch_size) + "_" + std::to_string(vocab_size),
        module
    );
    
    auto entry_bb = llvm::BasicBlock::Create(context, "entry", func);
    auto& builder = backend_.get_builder();
    builder.SetInsertPoint(entry_bb);
    builder.CreateRetVoid();
    
    return func;
}

} // namespace llvm_backend
} // namespace backend
} // namespace yirage

#endif // YIRAGE_USE_LLVM
