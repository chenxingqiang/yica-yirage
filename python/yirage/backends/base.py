"""
Base interfaces for YiRage backends
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import torch


class BackendInterface(ABC):
    """Abstract interface for YiRage compute backends."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the backend name."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> dict:
        """Get information about available devices."""
        pass
    
    @abstractmethod
    def allocate_memory(self, size: int, alignment: int = 32) -> Any:
        """Allocate memory on the backend device."""
        pass
    
    @abstractmethod
    def free_memory(self, ptr: Any) -> None:
        """Free memory allocated on the backend device."""
        pass
    
    @abstractmethod
    def create_stream(self) -> Any:
        """Create a compute stream for asynchronous execution."""
        pass
    
    @abstractmethod
    def synchronize_stream(self, stream: Any) -> None:
        """Synchronize the compute stream."""
        pass
    
    @abstractmethod
    def get_kernel_interface(self) -> 'KernelInterface':
        """Get the kernel interface for this backend."""
        pass


class KernelInterface(ABC):
    """Abstract interface for backend-specific kernel operations."""
    
    @abstractmethod
    def matmul(self, output: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
               m: int, n: int, k: int, stream: Any = None) -> None:
        """Matrix multiplication: output = a @ b"""
        pass
    
    @abstractmethod
    def rms_norm(self, output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                 batch_size: int, hidden_size: int, eps: float = 1e-6, stream: Any = None) -> None:
        """RMS normalization."""
        pass
    
    @abstractmethod
    def element_wise_unary(self, output: torch.Tensor, input: torch.Tensor,
                          op_type: str, stream: Any = None) -> None:
        """Element-wise unary operations (relu, gelu, silu, etc.)"""
        pass
    
    @abstractmethod
    def element_wise_binary(self, output: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                           op_type: str, stream: Any = None) -> None:
        """Element-wise binary operations (add, mul, sub, div)"""
        pass
    
    @abstractmethod
    def argmax(self, output: torch.Tensor, input: torch.Tensor,
               batch_size: int, vocab_size: int, stream: Any = None) -> None:
        """Argmax operation for token selection."""
        pass
    
    @abstractmethod
    def embedding_lookup(self, output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                        batch_size: int, seq_len: int, vocab_size: int, hidden_size: int,
                        stream: Any = None) -> None:
        """Embedding lookup operation."""
        pass
    
    @abstractmethod
    def attention(self, output: torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 batch_size: int, seq_len: int, num_heads: int, head_dim: int,
                 stream: Any = None) -> None:
        """Multi-head attention operation."""
        pass
