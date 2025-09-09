# YiRage 后端解耦迁移指南

本文档提供了从CUDA耦合版本迁移到多后端支持版本的详细指南。

## 概述

YiRage现在支持多种计算后端：
- **CUDA**: 原有的GPU加速后端
- **CPU**: 新的CPU优化后端
- **MPS**: Apple Silicon优化后端

## 向后兼容性

**重要**: 现有的CUDA代码在新版本中无需修改即可运行。所有原有API保持兼容。

## 新功能

### 1. 后端选择API

```python
import yirage as yr

# 设置全局后端
yr.set_backend('cuda')    # CUDA后端
yr.set_backend('cpu')     # CPU后端  
yr.set_backend('mps')     # MPS后端
yr.set_backend('auto')    # 自动选择最佳后端

# 检查后端可用性
print(yr.get_available_backends())  # 列出可用后端
print(yr.is_backend_available('cuda'))  # 检查特定后端

# 获取当前后端信息
info = yr.get_backend_info()
print(info)
```

### 2. 图创建时指定后端

```python
# 为特定图指定后端
cuda_graph = yr.new_kernel_graph(backend='cuda')
cpu_graph = yr.new_kernel_graph(backend='cpu')
mps_graph = yr.new_kernel_graph(backend='mps')

# 使用默认后端
default_graph = yr.new_kernel_graph()
```

### 3. PersistentKernel后端支持

```python
# 创建时指定后端
mpk = yr.PersistentKernel(
    world_size=1,
    mpi_rank=0,
    num_workers=96,
    num_local_schedulers=48,
    num_remote_schedulers=0,
    max_seq_length=512,
    eos_token_id=2,
    meta_tensors=[step, tokens],
    profiler_tensor=profiler_tensor,
    spec_decode_config=None,
    backend='cpu'  # 新参数
)
```

## 迁移步骤

### 阶段1：无修改运行（立即可用）

现有代码无需任何修改即可运行：

```python
# 这些代码在新版本中完全兼容
import yirage as yr

graph = yr.new_kernel_graph()
mpk = yr.PersistentKernel(...)
# ... 其他现有代码
```

### 阶段2：添加后端选择（可选）

根据需要添加后端选择：

```python
import yirage as yr

# 在程序开始时设置首选后端
yr.set_backend('auto')  # 或 'cuda', 'cpu', 'mps'

# 现有代码保持不变
graph = yr.new_kernel_graph()
# ...
```

### 阶段3：优化特定后端（高级）

针对不同后端进行优化：

```python
import yirage as yr

# 根据可用硬件选择最佳后端
if yr.is_backend_available('cuda'):
    yr.set_backend('cuda')
    # CUDA特定优化
    num_workers = 96
elif yr.is_backend_available('mps'):
    yr.set_backend('mps')
    # MPS特定优化
    num_workers = 64
else:
    yr.set_backend('cpu')
    # CPU特定优化
    num_workers = 8

mpk = yr.PersistentKernel(
    num_workers=num_workers,
    # ... 其他参数
)
```

## 环境配置

### 环境变量

```bash
# 通过环境变量设置默认后端
export YIRAGE_BACKEND=cuda
export YIRAGE_BACKEND=cpu
export YIRAGE_BACKEND=mps
export YIRAGE_BACKEND=auto
```

### 编译选项

```bash
# 编译时选择要包含的后端
cmake -DYIRAGE_USE_CUDA=ON \
      -DYIRAGE_USE_CPU=ON \
      -DYIRAGE_USE_MPS=OFF \
      ..
```

## 性能优化建议

### CUDA后端
```python
yr.set_backend('cuda')
# 保持原有的CUDA优化参数
num_workers = 96
num_local_schedulers = 48
```

### CPU后端
```python
yr.set_backend('cpu')
# CPU后端推荐设置
num_workers = 8  # 通常为CPU核心数
num_local_schedulers = 4
```

### MPS后端
```python
yr.set_backend('mps')
# MPS后端推荐设置
num_workers = 64  # 适合Apple Silicon
num_local_schedulers = 32
```

## 故障排除

### 常见问题

1. **后端不可用错误**
   ```python
   # 检查可用后端
   available = yr.get_available_backends()
   print(f"Available backends: {available}")
   
   # 使用自动选择
   yr.set_backend('auto')
   ```

2. **性能差异**
   ```python
   # 比较不同后端性能
   backends = yr.get_available_backends()
   for backend in backends:
       yr.set_backend(backend.value)
       # 运行基准测试
       # ...
   ```

3. **内存错误**
   ```python
   # CPU后端使用系统内存，调整批次大小
   if yr.get_backend().value == 'cpu':
       batch_size = batch_size // 4  # 减少内存使用
   ```

### 调试技巧

```python
# 启用详细日志
import os
os.environ['YIRAGE_LOG_LEVEL'] = 'DEBUG'

# 获取后端详细信息
info = yr.get_backend_info()
print(f"Current backend: {info}")

# 测试后端功能
from tests.test_multi_backend import run_tests
run_tests()
```

## 示例代码

### 完整的多后端示例

```python
#!/usr/bin/env python3
import yirage as yr
import torch

def main():
    # 自动选择最佳后端
    yr.set_backend('auto')
    print(f"Using backend: {yr.get_backend().value}")
    
    # 创建测试数据
    input_tensor = torch.randn(1, 128, 768)
    
    # 创建图
    graph = yr.new_kernel_graph()
    
    # 创建PersistentKernel
    step = torch.tensor([0], dtype=torch.int32)
    tokens = torch.full((1, 1024), 0, dtype=torch.long)
    
    mpk = yr.PersistentKernel(
        world_size=1,
        mpi_rank=0,
        num_workers=8,
        num_local_schedulers=4,
        num_remote_schedulers=0,
        max_seq_length=1024,
        eos_token_id=2,
        meta_tensors=[step, tokens],
        profiler_tensor=None,
        spec_decode_config=None
    )
    
    print("✓ Successfully created multi-backend YiRage setup")

if __name__ == "__main__":
    main()
```

### 性能比较示例

```python
import yirage as yr
import time

def benchmark_backend(backend_name, iterations=100):
    """基准测试特定后端"""
    yr.set_backend(backend_name)
    
    # 创建测试负载
    graph = yr.new_kernel_graph()
    
    # 计时
    start_time = time.time()
    for _ in range(iterations):
        # 执行操作
        pass
    end_time = time.time()
    
    return (end_time - start_time) / iterations

# 比较所有可用后端
results = {}
for backend in yr.get_available_backends():
    backend_name = backend.value
    try:
        avg_time = benchmark_backend(backend_name)
        results[backend_name] = avg_time
        print(f"{backend_name}: {avg_time:.4f}s")
    except Exception as e:
        print(f"{backend_name}: Error - {e}")

# 显示最佳后端
if results:
    best_backend = min(results, key=results.get)
    print(f"Best backend: {best_backend}")
```

## 总结

多后端支持为YiRage带来了更大的灵活性和更广泛的硬件支持。通过渐进式迁移，您可以：

1. **立即受益**: 现有代码无需修改即可运行
2. **灵活选择**: 根据硬件和需求选择最佳后端
3. **性能优化**: 针对特定后端进行优化
4. **未来保证**: 为新硬件和后端做好准备

如有问题，请参考测试套件或联系开发团队。
