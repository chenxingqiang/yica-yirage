---
name: Performance Issue
about: Report performance problems or optimization suggestions
title: "[PERFORMANCE] "
labels: performance
assignees: ''

---

## ⚡ Performance Issue Description
Describe the performance problem you're experiencing.

## 📊 Benchmark Results
Please provide performance measurements:

**Current Performance**:
- Operation: [e.g., matrix multiplication, attention computation]
- Backend: [CPU/CUDA/MPS]
- Input size: [e.g., 4096x4096, batch_size=8, seq_len=512]
- Average time: [e.g., 45.2ms]
- Throughput: [e.g., 1250 ops/sec]

**Expected Performance**:
- Expected time: [e.g., <20ms]
- Reference: [e.g., cuBLAS, MKL, other frameworks]

## 🔧 Reproduction Code
```python
import yirage as yr

# Your benchmark code here
yr.set_backend('cuda')
# ... 
```

## 🖥️ System Information
Run and paste the output of:
```bash
python tools/yirage_backend_manager.py info
python tools/yirage_backend_manager.py benchmark --duration 30
```

**Additional Hardware Details**:
- GPU Memory: [e.g., 24GB]
- Memory Bandwidth: [e.g., 1TB/s]
- Compute Capability: [e.g., 8.9]

## 📈 Performance Analysis
Have you done any profiling? Please share:

**Profiling Tools Used**:
- [ ] NVIDIA Nsight
- [ ] Intel VTune
- [ ] Apple Instruments
- [ ] PyTorch Profiler
- [ ] Other: ___________

**Bottlenecks Identified**:
- [ ] Memory bandwidth
- [ ] Compute utilization
- [ ] Memory allocation
- [ ] Kernel launch overhead
- [ ] Data transfer overhead
- [ ] Other: ___________

## 🎯 Optimization Suggestions
If you have ideas for optimization:
- [ ] Better memory layout
- [ ] Kernel fusion opportunities
- [ ] Algorithm improvements
- [ ] Hardware-specific optimizations

## 📋 Additional Context
Any other information that might help understand the performance issue.

## ✅ Checklist
- [ ] I have provided complete benchmark results
- [ ] I have tested on the latest YiRage version
- [ ] I have compared with other implementations (if applicable)
- [ ] I have profiled the code to identify bottlenecks
