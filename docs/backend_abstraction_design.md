# YiRage 后端抽象层设计文档

## 概述

本文档描述了YiRage项目的后端抽象层设计，目标是将CUDA特定的实现解耦，支持多种计算后端：
- CPU后端（使用OpenMP/BLAS优化）
- MPS后端（Apple Metal Performance Shaders）
- 保持原有CUDA后端兼容性

## 当前架构分析

### CUDA耦合点分析

1. **内存管理**
   - `DeviceMemoryManager` 直接使用CUDA API
   - `cudaMalloc`, `cudaFree`, `cudaMemcpy`
   - CUDA流管理

2. **内核实现**
   - `src/kernel/cuda/` 目录下的所有CUDA内核
   - CUTLASS库依赖
   - `__global__`, `__device__` 函数

3. **编译器后端**
   - 生成CUDA代码的transpiler
   - CUDA特定的优化和布局

4. **运行时系统**
   - `runtime.h` 中的CUDA流处理
   - 持久化内核的CUDA实现

## 新架构设计

### 1. 后端抽象层 (Backend Abstraction Layer)

```cpp
// include/yirage/backend/backend_interface.h
namespace yirage {
namespace backend {

enum class BackendType {
    CUDA,
    CPU,
    MPS
};

class BackendInterface {
public:
    virtual ~BackendInterface() = default;
    
    // 内存管理
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void memcpy(void* dst, const void* src, size_t size) = 0;
    virtual void memset(void* ptr, int value, size_t size) = 0;
    
    // 同步和流管理
    virtual void synchronize() = 0;
    virtual void* create_stream() = 0;
    virtual void destroy_stream(void* stream) = 0;
    
    // 设备信息
    virtual int get_device_count() = 0;
    virtual void set_device(int device_id) = 0;
    virtual size_t get_available_memory() = 0;
    
    // 内核执行
    virtual void launch_kernel(const std::string& kernel_name,
                             void** inputs, void** outputs,
                             const KernelConfig& config) = 0;
};

}}
```

### 2. 设备内存管理抽象

```cpp
// include/yirage/backend/memory_manager.h
namespace yirage {
namespace backend {

class DeviceMemoryManager {
private:
    std::unique_ptr<BackendInterface> backend_;
    BackendType backend_type_;
    
public:
    DeviceMemoryManager(BackendType type);
    
    template<typename T>
    T* allocate(size_t count);
    
    void deallocate(void* ptr);
    
    template<typename T>
    void copy_to_device(T* dst, const T* src, size_t count);
    
    template<typename T>
    void copy_to_host(T* dst, const T* src, size_t count);
};

}}
```

### 3. 内核抽象层

```cpp
// include/yirage/backend/kernel_interface.h
namespace yirage {
namespace backend {

struct KernelConfig {
    std::array<int, 3> grid_dim;
    std::array<int, 3> block_dim;
    size_t shared_memory_size = 0;
    void* stream = nullptr;
};

class KernelInterface {
public:
    virtual ~KernelInterface() = default;
    
    // 基础数学操作
    virtual void matmul(void* output, const void* a, const void* b,
                       int m, int n, int k, DataType dtype) = 0;
    
    virtual void element_wise_unary(void* output, const void* input,
                                  UnaryOpType op_type, int num_elements,
                                  DataType dtype) = 0;
    
    virtual void element_wise_binary(void* output, const void* a, const void* b,
                                   BinaryOpType op_type, int num_elements,
                                   DataType dtype) = 0;
    
    virtual void reduction(void* output, const void* input,
                         ReductionType reduction_type, 
                         const std::vector<int>& dims,
                         DataType dtype) = 0;
    
    virtual void rms_norm(void* output, const void* input, const void* weight,
                         int batch_size, int hidden_size, float eps,
                         DataType dtype) = 0;
    
    virtual void attention(void* output, const void* query, const void* key,
                         const void* value, const AttentionConfig& config) = 0;
};

}}
```

### 4. 编译器后端抽象

```cpp
// include/yirage/backend/transpiler_interface.h
namespace yirage {
namespace backend {

class TranspilerInterface {
public:
    virtual ~TranspilerInterface() = default;
    
    virtual std::string transpile_graph(const kernel::Graph& graph,
                                       const TranspilerConfig& config) = 0;
    
    virtual CompilationResult compile_code(const std::string& code,
                                         const CompilationConfig& config) = 0;
    
    virtual BackendType get_target_backend() const = 0;
};

// 具体实现类
class CudaTranspiler : public TranspilerInterface { /* ... */ };
class CpuTranspiler : public TranspilerInterface { /* ... */ };
class MpsTranspiler : public TranspilerInterface { /* ... */ };

}}
```

## 实现策略

### 阶段1：后端接口定义和CUDA适配
1. 创建抽象接口
2. 将现有CUDA实现包装到新接口中
3. 确保向后兼容性

### 阶段2：CPU后端实现
1. 使用OpenMP进行并行化
2. 集成BLAS库（OpenBLAS/MKL）
3. 实现CPU特定的内核

### 阶段3：MPS后端实现
1. 使用Metal Performance Shaders
2. 实现Metal内核
3. 优化内存传输

### 阶段4：编译器重构
1. 抽象代码生成
2. 支持多目标编译
3. 运行时后端选择

## 目录结构重组

```
include/yirage/backend/
├── backend_interface.h
├── memory_manager.h
├── kernel_interface.h
├── transpiler_interface.h
├── cuda/
│   ├── cuda_backend.h
│   └── cuda_kernels.h
├── cpu/
│   ├── cpu_backend.h
│   └── cpu_kernels.h
└── mps/
    ├── mps_backend.h
    └── mps_kernels.h

src/backend/
├── backend_factory.cc
├── memory_manager.cc
├── cuda/
│   ├── cuda_backend.cc
│   ├── cuda_memory.cc
│   └── cuda_kernels.cu
├── cpu/
│   ├── cpu_backend.cc
│   ├── cpu_memory.cc
│   └── cpu_kernels.cc
└── mps/
    ├── mps_backend.cc
    ├── mps_memory.cc
    └── mps_kernels.metal
```

## API变更

### Python API增强

```python
import yirage as yr

# 后端选择
yr.set_backend('cuda')  # 或 'cpu', 'mps'

# 或者在图创建时指定
graph = yr.new_kernel_graph(backend='cpu')

# 持久化内核支持多后端
mpk = yr.PersistentKernel(
    backend='mps',  # 新参数
    world_size=1,
    # ... 其他参数
)
```

### C++ API变更

```cpp
// 后端工厂
auto backend = BackendFactory::create(BackendType::CPU);
auto memory_manager = std::make_unique<DeviceMemoryManager>(backend.get());

// 图编译时指定后端
TranspilerConfig config;
config.target_backend = BackendType::MPS;
auto transpiler = TranspilerFactory::create(config.target_backend);
```

## 性能考虑

### CPU后端优化
- 使用SIMD指令（AVX, AVX-512）
- OpenMP并行化
- 高效的BLAS库集成
- 内存布局优化

### MPS后端优化
- Metal内核优化
- 统一内存使用
- GPU/CPU协同计算
- Apple Silicon特定优化

### 通用优化
- 零拷贝内存传输（当可能时）
- 异步执行支持
- 内存池管理
- 动态后端切换

## 测试策略

1. **功能测试**：确保所有后端产生相同结果
2. **性能测试**：基准测试各后端性能
3. **兼容性测试**：验证现有代码的兼容性
4. **集成测试**：端到端的多后端测试

## 迁移路径

1. **向后兼容**：现有CUDA代码无需修改即可运行
2. **渐进式迁移**：用户可以逐步采用新API
3. **配置驱动**：通过环境变量或配置文件选择后端
4. **运行时检测**：自动检测可用后端并选择最佳选项

## 结论

这个设计提供了一个清晰的抽象层，允许YiRage支持多种计算后端，同时保持高性能和易用性。通过分阶段实现，可以确保项目的稳定性和向后兼容性。
