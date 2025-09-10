"""
YiRage Backend Implementations

This module provides backend-specific implementations for different hardware targets.
"""

from .base import BackendInterface, KernelInterface
from .factory import create_backend

# Import backend implementations
try:
    from .cuda_backend import CUDABackend
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    from .cpu_backend import CPUBackend
    CPU_AVAILABLE = True
except ImportError:
    CPU_AVAILABLE = False

try:
    from .mps_backend import MPSBackend
    MPS_AVAILABLE = True
except ImportError:
    MPS_AVAILABLE = False

try:
    from .llvm_backend import LLVMBackend
    LLVM_AVAILABLE = True
except ImportError:
    LLVM_AVAILABLE = False


__all__ = [
    'BackendInterface',
    'KernelInterface',
    'create_backend',
    'CUDA_AVAILABLE',
    'CPU_AVAILABLE',
    'MPS_AVAILABLE',
    'LLVM_AVAILABLE'
]
