"""
CPU Backend Implementation for YiRage
"""

import torch
import numpy as np
from typing import Any, Dict, Optional
import multiprocessing
import warnings

from .base import BackendInterface, KernelInterface


class CPUKernelInterface(KernelInterface):
    """CPU-optimized kernel implementations."""
    
    def __init__(self, backend: 'CPUBackend'):
        self.backend = backend
        self.use_openmp = self._check_openmp_support()
        self.use_mkl = self._check_mkl_support()
    
    def _check_openmp_support(self) -> bool:
        """Check if OpenMP is available."""
        try:
            # Check if torch was compiled with OpenMP
            return torch.get_num_threads() > 1
        except:
            return False
    
    def _check_mkl_support(self) -> bool:
        """Check if Intel MKL is available."""
        try:
            return torch.backends.mkl.is_available()
        except:
            return False
    
    def matmul(self, output: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
               m: int, n: int, k: int, stream: Any = None) -> None:
        """Optimized CPU matrix multiplication."""
        # Ensure tensors are on CPU
        a_cpu = a.cpu() if a.device.type != 'cpu' else a
        b_cpu = b.cpu() if b.device.type != 'cpu' else b
        
        # Use optimized torch operations
        with torch.no_grad():
            if self.use_mkl:
                # Use Intel MKL if available
                result = torch.mm(a_cpu, b_cpu)
            else:
                # Fallback to standard implementation
                result = torch.matmul(a_cpu, b_cpu)
            
            output.copy_(result)
    
    def rms_norm(self, output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                 batch_size: int, hidden_size: int, eps: float = 1e-6, stream: Any = None) -> None:
        """CPU RMS normalization."""
        input_cpu = input.cpu() if input.device.type != 'cpu' else input
        weight_cpu = weight.cpu() if weight.device.type != 'cpu' else weight
        
        with torch.no_grad():
            # Compute RMS
            variance = input_cpu.pow(2).mean(dim=-1, keepdim=True)
            input_normalized = input_cpu * torch.rsqrt(variance + eps)
            
            # Apply weight scaling
            result = input_normalized * weight_cpu
            output.copy_(result)
    
    def element_wise_unary(self, output: torch.Tensor, input: torch.Tensor,
                          op_type: str, stream: Any = None) -> None:
        """CPU element-wise unary operations."""
        input_cpu = input.cpu() if input.device.type != 'cpu' else input
        
        with torch.no_grad():
            if op_type.lower() == 'relu':
                result = torch.relu(input_cpu)
            elif op_type.lower() == 'gelu':
                result = torch.nn.functional.gelu(input_cpu)
            elif op_type.lower() == 'silu' or op_type.lower() == 'swish':
                result = torch.nn.functional.silu(input_cpu)
            elif op_type.lower() == 'tanh':
                result = torch.tanh(input_cpu)
            elif op_type.lower() == 'sigmoid':
                result = torch.sigmoid(input_cpu)
            else:
                raise ValueError(f"Unsupported unary operation: {op_type}")
            
            output.copy_(result)
    
    def element_wise_binary(self, output: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                           op_type: str, stream: Any = None) -> None:
        """CPU element-wise binary operations."""
        a_cpu = a.cpu() if a.device.type != 'cpu' else a
        b_cpu = b.cpu() if b.device.type != 'cpu' else b
        
        with torch.no_grad():
            if op_type.lower() == 'add':
                result = torch.add(a_cpu, b_cpu)
            elif op_type.lower() == 'mul':
                result = torch.mul(a_cpu, b_cpu)
            elif op_type.lower() == 'sub':
                result = torch.sub(a_cpu, b_cpu)
            elif op_type.lower() == 'div':
                result = torch.div(a_cpu, b_cpu)
            else:
                raise ValueError(f"Unsupported binary operation: {op_type}")
            
            output.copy_(result)
    
    def argmax(self, output: torch.Tensor, input: torch.Tensor,
               batch_size: int, vocab_size: int, stream: Any = None) -> None:
        """CPU argmax operation."""
        input_cpu = input.cpu() if input.device.type != 'cpu' else input
        
        with torch.no_grad():
            # Compute argmax along the last dimension
            result = torch.argmax(input_cpu, dim=-1, keepdim=True)
            output.copy_(result)
    
    def embedding_lookup(self, output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                        batch_size: int, seq_len: int, vocab_size: int, hidden_size: int,
                        stream: Any = None) -> None:
        """CPU embedding lookup."""
        input_cpu = input.cpu() if input.device.type != 'cpu' else input
        weight_cpu = weight.cpu() if weight.device.type != 'cpu' else weight
        
        with torch.no_grad():
            result = torch.nn.functional.embedding(input_cpu, weight_cpu)
            output.copy_(result)
    
    def attention(self, output: torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 batch_size: int, seq_len: int, num_heads: int, head_dim: int,
                 stream: Any = None) -> None:
        """CPU multi-head attention."""
        q_cpu = query.cpu() if query.device.type != 'cpu' else query
        k_cpu = key.cpu() if key.device.type != 'cpu' else key
        v_cpu = value.cpu() if value.device.type != 'cpu' else value
        
        with torch.no_grad():
            # Scaled dot-product attention
            scale = 1.0 / (head_dim ** 0.5)
            
            # Compute attention scores
            scores = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale
            
            # Apply softmax
            attn_weights = torch.softmax(scores, dim=-1)
            
            # Apply to values
            result = torch.matmul(attn_weights, v_cpu)
            output.copy_(result)


class CPUBackend(BackendInterface):
    """CPU backend implementation."""
    
    def __init__(self, **kwargs):
        """Initialize CPU backend."""
        self.num_threads = kwargs.get('num_threads', multiprocessing.cpu_count())
        self.use_openmp = kwargs.get('use_openmp', True)
        self.use_mkl = kwargs.get('use_mkl', True)
        
        # Set number of threads for PyTorch
        torch.set_num_threads(self.num_threads)
        
        self._kernel_interface = CPUKernelInterface(self)
        
        print(f"Initialized CPU backend with {self.num_threads} threads")
        print(f"OpenMP support: {self._kernel_interface.use_openmp}")
        print(f"MKL support: {self._kernel_interface.use_mkl}")
    
    def get_name(self) -> str:
        """Get backend name."""
        return "CPU"
    
    def is_available(self) -> bool:
        """CPU is always available."""
        return True
    
    def get_device_info(self) -> dict:
        """Get CPU device information."""
        return {
            'device_type': 'cpu',
            'num_cores': multiprocessing.cpu_count(),
            'num_threads': self.num_threads,
            'openmp_available': self._kernel_interface.use_openmp,
            'mkl_available': self._kernel_interface.use_mkl,
            'memory_available': self._get_available_memory()
        }
    
    def _get_available_memory(self) -> int:
        """Get available system memory in bytes."""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            # Fallback estimate
            return 8 * 1024 * 1024 * 1024  # 8GB
    
    def allocate_memory(self, size: int, alignment: int = 32) -> Any:
        """Allocate CPU memory."""
        try:
            # Use numpy for aligned memory allocation
            dtype = np.uint8
            aligned_size = (size + alignment - 1) // alignment * alignment
            return np.empty(aligned_size, dtype=dtype)
        except MemoryError:
            raise RuntimeError(f"Failed to allocate {size} bytes of CPU memory")
    
    def free_memory(self, ptr: Any) -> None:
        """Free CPU memory (handled by Python GC)."""
        # Python garbage collector handles memory deallocation
        del ptr
    
    def create_stream(self) -> Any:
        """Create a CPU 'stream' (no-op for CPU)."""
        # CPU execution is synchronous, return a dummy stream
        return "cpu_stream"
    
    def synchronize_stream(self, stream: Any) -> None:
        """Synchronize CPU stream (no-op)."""
        # CPU execution is synchronous, nothing to synchronize
        pass
    
    def get_kernel_interface(self) -> KernelInterface:
        """Get CPU kernel interface."""
        return self._kernel_interface
