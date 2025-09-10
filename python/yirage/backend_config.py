# Copyright 2025-2026 YICA TEAM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend configuration and management for YiRage."""

import os
from enum import Enum
from typing import Optional, List, Dict, Any


class BackendType(Enum):
    """Available backend types for YiRage computation."""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
    AUTO = "auto"


class BackendConfig:
    """Configuration for YiRage backends."""
    
    def __init__(self):
        self._current_backend: Optional[BackendType] = None
        self._backend_options: Dict[str, Any] = {}
        self._available_backends: List[BackendType] = []
        self._detect_available_backends()
    
    def _detect_available_backends(self):
        """Detect which backends are available on the current system."""
        # This will be implemented with actual backend detection
        # For now, assume all backends are potentially available
        self._available_backends = [BackendType.CPU]
        
        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self._available_backends.append(BackendType.CUDA)
        except ImportError:
            pass
        
        # Check for MPS (Apple Silicon)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._available_backends.append(BackendType.MPS)
        except (ImportError, AttributeError):
            pass
    
    def set_backend(self, backend: BackendType, **options):
        """Set the current backend with optional configuration."""
        if isinstance(backend, str):
            backend = BackendType(backend.lower())
        
        if backend == BackendType.AUTO:
            backend = self.get_best_backend()
        
        if backend not in self._available_backends:
            available = [b.value for b in self._available_backends]
            raise ValueError(f"Backend {backend.value} is not available. "
                           f"Available backends: {available}")
        
        self._current_backend = backend
        self._backend_options.update(options)
        
        # Set environment variable for C++ backend selection
        os.environ['YIRAGE_BACKEND'] = backend.value
    
    def get_backend(self) -> BackendType:
        """Get the current backend."""
        if self._current_backend is None:
            # Auto-detect and set the best backend
            self._current_backend = self.get_best_backend()
            os.environ['YIRAGE_BACKEND'] = self._current_backend.value
        return self._current_backend
    
    def get_best_backend(self) -> BackendType:
        """Get the best available backend based on system capabilities."""
        # Priority order: CUDA > MPS > CPU
        if BackendType.CUDA in self._available_backends:
            return BackendType.CUDA
        elif BackendType.MPS in self._available_backends:
            return BackendType.MPS
        elif BackendType.CPU in self._available_backends:
            return BackendType.CPU
        else:
            raise RuntimeError("No supported backend available")
    
    def get_available_backends(self) -> List[BackendType]:
        """Get list of available backends."""
        return self._available_backends.copy()
    
    def is_backend_available(self, backend: BackendType) -> bool:
        """Check if a specific backend is available."""
        if isinstance(backend, str):
            backend = BackendType(backend.lower())
        return backend in self._available_backends
    
    def get_backend_options(self) -> Dict[str, Any]:
        """Get current backend options."""
        return self._backend_options.copy()
    
    def reset(self):
        """Reset backend configuration to default."""
        self._current_backend = None
        self._backend_options.clear()
        if 'YIRAGE_BACKEND' in os.environ:
            del os.environ['YIRAGE_BACKEND']


# Global backend configuration instance
_backend_config = BackendConfig()


def set_backend(backend: BackendType, **options):
    """Set the global YiRage backend.
    
    Args:
        backend: The backend type to use
        **options: Backend-specific options
        
    Examples:
        >>> import yirage as yr
        >>> yr.set_backend('cuda')
        >>> yr.set_backend('cpu', num_threads=8)
        >>> yr.set_backend('mps', use_unified_memory=True)
    """
    _backend_config.set_backend(backend, **options)


def get_backend() -> BackendType:
    """Get the current global backend."""
    return _backend_config.get_backend()


def get_available_backends() -> List[BackendType]:
    """Get list of available backends."""
    return _backend_config.get_available_backends()


def get_current_backend() -> BackendType:
    """Get the currently active backend."""
    return _backend_config.get_backend()


def is_backend_available(backend: BackendType) -> bool:
    """Check if a backend is available."""
    return _backend_config.is_backend_available(backend)


def get_backend_info() -> Dict[str, Any]:
    """Get information about the current backend configuration."""
    return {
        'current_backend': _backend_config.get_backend().value,
        'available_backends': [b.value for b in _backend_config.get_available_backends()],
        'backend_options': _backend_config.get_backend_options()
    }


def reset_backend():
    """Reset backend configuration to default."""
    _backend_config.reset()
