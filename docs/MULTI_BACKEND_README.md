# YiRage 多后端支持

YiRage现在支持多种计算后端，为不同硬件环境提供优化的性能。

## 支持的后端

### 🖥️ CPU 后端
- **适用场景**: 无GPU环境、开发调试、小模型推理
- **优化特性**: 
  - OpenMP并行化
  - SIMD指令优化
  - 高效内存管理
  - BLAS库集成

### 🚀 CUDA 后端
- **适用场景**: NVIDIA GPU加速
- **优化特性**:
  - 原生CUDA内核
  - cuBLAS优化
  - 多GPU支持
  - 内存池管理

### 🍎 MPS 后端
- **适用场景**: Apple Silicon (M1/M2/M3) 优化
- **优化特性**:
  - Metal Performance Shaders
  - 统一内存架构
  - Apple GPU优化
  - 低功耗推理

## 快速开始

### 基本使用

```python
import yirage as yr

# 查看可用后端
print("Available backends:", [b.value for b in yr.get_available_backends()])

# 自动选择最佳后端
yr.set_backend('auto')
print("Current backend:", yr.get_backend().value)

# 手动选择后端
yr.set_backend('cuda')  # 或 'cpu', 'mps'
```

### 创建后端特定的图

```python
# 为不同后端创建图
cuda_graph = yr.new_kernel_graph(backend='cuda')
cpu_graph = yr.new_kernel_graph(backend='cpu')
mps_graph = yr.new_kernel_graph(backend='mps')
```

### PersistentKernel多后端支持

```python
mpk = yr.PersistentKernel(
    world_size=1,
    mpi_rank=0,
    num_workers=32,
    num_local_schedulers=16,
    num_remote_schedulers=0,
    max_seq_length=1024,
    eos_token_id=2,
    meta_tensors=[step, tokens],
    profiler_tensor=None,
    spec_decode_config=None,
    backend='mps'  # 指定后端
)
```

## 命令行工具

### 后端管理器

```bash
# 查看系统信息和可用后端
python tools/yirage_backend_manager.py info

# 自动优化配置
python tools/yirage_backend_manager.py optimize --backend cuda --apply

# 性能基准测试
python tools/yirage_backend_manager.py benchmark --duration 30

# 设置当前后端
python tools/yirage_backend_manager.py set cuda

# 测试后端功能
python tools/yirage_backend_manager.py test cpu
```

### 多后端演示

```bash
# 运行多后端演示
python demo/demo_multi_backend.py --backend all --iterations 20

# 仅测试CPU后端
python demo/demo_multi_backend.py --backend cpu

# 跳过PersistentKernel演示
python demo/demo_multi_backend.py --skip-persistent
```

### 性能基准测试

```bash
# 运行综合性能测试
python benchmark/multi_backend_benchmark.py --iterations 50 --output results.json

# 自定义测试配置
python benchmark/multi_backend_benchmark.py \
    --batch-sizes 1 4 8 \
    --seq-lengths 128 512 1024 \
    --hidden-sizes 768 1024 \
    --dtype float16
```

## 环境配置

### 环境变量

```bash
# 设置默认后端
export YIRAGE_BACKEND=cuda
export YIRAGE_BACKEND=cpu
export YIRAGE_BACKEND=mps
export YIRAGE_BACKEND=auto

# 设置日志级别
export YIRAGE_LOG_LEVEL=DEBUG
```

### 编译选项

```bash
# 选择要编译的后端
cmake -DYIRAGE_USE_CUDA=ON \
      -DYIRAGE_USE_CPU=ON \
      -DYIRAGE_USE_MPS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..
```

## 高级功能

### 自动配置优化

```python
# 自动检测最优配置
config = yr.auto_configure_backend()
print("Optimal config:", config)

# 系统信息分析
optimizer = yr.BackendOptimizer()
optimizer.print_system_info()
```

### 性能监控

```python
# 获取内存使用信息
memory_info = yr.get_memory_info('cuda')
print("Memory usage:", memory_info)

# 快速性能基准测试
results = yr.benchmark_backends(duration_seconds=10)
print("Performance results:", results)
```

### 配置管理

```python
# 保存优化配置
optimizer = yr.BackendOptimizer()
optimizer.save_config('optimal_config.json', backend='cuda')

# 加载配置
config = optimizer.load_config('optimal_config.json')
```

## 性能调优指南

### CUDA后端优化

```python
# 高性能GPU (>= 24GB)
yr.set_backend('cuda')
mpk = yr.PersistentKernel(
    num_workers=96,
    num_local_schedulers=48,
    # ... 其他参数
)
```

### CPU后端优化

```python
import os

# 设置OpenMP线程数
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

yr.set_backend('cpu')
mpk = yr.PersistentKernel(
    num_workers=8,
    num_local_schedulers=4,
    # ... 其他参数
)
```

### MPS后端优化

```python
# Apple Silicon优化
yr.set_backend('mps')
mpk = yr.PersistentKernel(
    num_workers=64,
    num_local_schedulers=32,
    # ... 其他参数
)
```

## 故障排除

### 常见问题

1. **后端不可用**
   ```python
   # 检查可用后端
   available = yr.get_available_backends()
   print(f"Available: {[b.value for b in available]}")
   
   # 使用自动选择
   yr.set_backend('auto')
   ```

2. **性能不佳**
   ```bash
   # 运行系统分析
   python tools/yirage_backend_manager.py info
   
   # 获取优化建议
   python tools/yirage_backend_manager.py optimize --backend auto
   ```

3. **内存不足**
   ```python
   # 检查内存使用
   memory_info = yr.get_memory_info(yr.get_backend().value)
   print("Memory status:", memory_info)
   
   # 减少批次大小或工作线程数
   ```

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
import os
os.environ['YIRAGE_LOG_LEVEL'] = 'DEBUG'
```

### 测试后端功能

```bash
# 运行后端测试套件
python tests/test_multi_backend.py

# 测试特定后端
python tools/yirage_backend_manager.py test cuda
```

## 示例项目

### 简单的LLM推理

```python
import yirage as yr
import torch

# 自动配置
yr.set_backend('auto')
print(f"Using backend: {yr.get_backend().value}")

# 创建模拟的LLM组件
batch_size, seq_len, hidden_size = 1, 128, 768

# 输入数据
input_ids = torch.randint(0, 32000, (batch_size, seq_len))
embeddings = torch.randn(batch_size, seq_len, hidden_size)

# 简单的前向传播
def simple_forward(x):
    # Layer norm
    x = torch.layer_norm(x, (hidden_size,))
    
    # Linear transformation
    weight = torch.randn(hidden_size, hidden_size)
    x = torch.matmul(x, weight)
    
    # Activation
    x = torch.relu(x)
    
    return x

# 执行推理
with torch.no_grad():
    output = simple_forward(embeddings)
    print(f"Output shape: {output.shape}")
    print(f"Backend used: {yr.get_backend().value}")
```

### 多后端性能比较

```python
import yirage as yr
import time
import torch

def benchmark_operation(backend, operation_name, operation_func, *args):
    """基准测试特定操作"""
    yr.set_backend(backend)
    
    # 热身
    for _ in range(5):
        result = operation_func(*args)
    
    # 计时
    start_time = time.time()
    for _ in range(100):
        result = operation_func(*args)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # ms
    return avg_time

# 测试操作
def matmul_op(a, b):
    return torch.matmul(a, b)

# 比较所有后端
backends = [b.value for b in yr.get_available_backends()]
a = torch.randn(1024, 1024)
b = torch.randn(1024, 1024)

results = {}
for backend in backends:
    try:
        time_ms = benchmark_operation(backend, 'matmul', matmul_op, a, b)
        results[backend] = time_ms
        print(f"{backend}: {time_ms:.2f} ms")
    except Exception as e:
        print(f"{backend}: Error - {e}")

# 显示最佳后端
if results:
    best_backend = min(results, key=results.get)
    print(f"\nBest backend: {best_backend} ({results[best_backend]:.2f} ms)")
```

## 贡献指南

### 添加新后端

1. 创建后端接口实现
2. 实现内核接口
3. 添加到工厂方法
4. 更新CMake配置
5. 添加测试用例

### 性能优化

1. 分析性能瓶颈
2. 实现优化内核
3. 添加基准测试
4. 验证正确性

## 许可证

本项目使用Apache License 2.0许可证。详见LICENSE文件。
