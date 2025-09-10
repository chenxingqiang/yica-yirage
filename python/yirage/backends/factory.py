"""
Backend factory for creating appropriate backend instances
"""

from typing import Optional, Dict, Any
from ..backend_config import BackendType
from .base import BackendInterface


class BackendFactory:
    """Factory for creating backend instances."""
    
    _backends: Dict[str, type] = {}
    _instances: Dict[str, BackendInterface] = {}
    
    @classmethod
    def register_backend(cls, name: str, backend_class: type):
        """Register a backend implementation."""
        cls._backends[name] = backend_class
    
    @classmethod
    def create_backend(cls, backend_type: BackendType, **kwargs) -> BackendInterface:
        """Create a backend instance."""
        backend_name = backend_type.value
        
        # Return cached instance if available
        if backend_name in cls._instances:
            return cls._instances[backend_name]
        
        # Import and create backend based on type
        if backend_type == BackendType.CUDA:
            return cls._create_cuda_backend(**kwargs)
        elif backend_type == BackendType.CPU:
            return cls._create_cpu_backend(**kwargs)
        elif backend_type == BackendType.MPS:
            return cls._create_mps_backend(**kwargs)
        elif backend_type == BackendType.LLVM:
            return cls._create_llvm_backend(**kwargs)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
    
    @classmethod
    def _create_cuda_backend(cls, **kwargs) -> BackendInterface:
        """Create CUDA backend."""
        try:
            from .cuda_backend import CUDABackend
            backend = CUDABackend(**kwargs)
            cls._instances['cuda'] = backend
            return backend
        except ImportError as e:
            raise RuntimeError(f"CUDA backend not available: {e}")
    
    @classmethod
    def _create_cpu_backend(cls, **kwargs) -> BackendInterface:
        """Create CPU backend."""
        try:
            from .cpu_backend import CPUBackend
            backend = CPUBackend(**kwargs)
            cls._instances['cpu'] = backend
            return backend
        except ImportError as e:
            raise RuntimeError(f"CPU backend not available: {e}")
    
    @classmethod
    def _create_mps_backend(cls, **kwargs) -> BackendInterface:
        """Create MPS backend."""
        try:
            from .mps_backend import MPSBackend
            backend = MPSBackend(**kwargs)
            cls._instances['mps'] = backend
            return backend
        except ImportError as e:
            raise RuntimeError(f"MPS backend not available: {e}")
    
    @classmethod
    def _create_llvm_backend(cls, **kwargs) -> BackendInterface:
        """Create LLVM backend."""
        try:
            from .llvm_backend import LLVMBackend
            backend = LLVMBackend(**kwargs)
            cls._instances['llvm'] = backend
            return backend
        except ImportError as e:
            raise RuntimeError(f"LLVM backend not available: {e}")
    
    @classmethod
    def get_available_backends(cls) -> list:
        """Get list of available backends."""
        available = []
        
        # Check each backend
        for backend_type in BackendType:
            try:
                backend = cls.create_backend(backend_type)
                if backend.is_available():
                    available.append(backend_type.value)
            except (ImportError, RuntimeError):
                pass
        
        return available


def create_backend(backend_type: BackendType, **kwargs) -> BackendInterface:
    """Convenience function to create a backend."""
    return BackendFactory.create_backend(backend_type, **kwargs)
