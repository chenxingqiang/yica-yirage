"""
YiRage: A Multi-Level Superoptimizer for Tensor Algebra
"""

import os
import warnings

# Try to import optional dependencies
_NATIVE_EXTENSIONS_AVAILABLE = False

try:
    # Try to load native extensions if available
    import ctypes
    
    # Check for Z3 (optional for advanced features)
    # Skip Z3 for now due to compatibility issues
    
    # Try to load native YiRage extensions
    _this_dir = os.path.dirname(__file__)
    _yirage_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))
    
    # Check for compiled extensions
    extension_paths = [
        os.path.join(_yirage_root, "build", "abstract_subexpr", "release", "libabstract_subexpr.so"),
        os.path.join(_yirage_root, "build", "formal_verifier", "release", "libformal_verifier.so"),
    ]
    
    for path in extension_paths:
        if os.path.exists(path):
            try:
                ctypes.CDLL(path)
            except OSError:
                pass  # Continue with Python-only mode
    
    # Try to import core modules
    try:
        # Try native extensions first
        from ._cython.core import *  # Cython compiled version
        from .kernel import *
        from .threadblock import *
        _NATIVE_EXTENSIONS_AVAILABLE = True
    except ImportError:
        try:
            # Fallback to Python implementation
            from .core import *  # Python-only version
            from .kernel import *
            from .threadblock import *
            _NATIVE_EXTENSIONS_AVAILABLE = False
            warnings.warn("Using Python-only implementation. For better performance, build with native extensions.")
        except ImportError:
            warnings.warn("Native YiRage extensions not available. Running in Python-only mode.")

except ImportError:
    warnings.warn("YiRage native extensions not available. Some features may be limited.")

# Always available Python modules
from .version import __version__

# Define data types for compatibility
import torch
float16 = torch.float16
float32 = torch.float32
bfloat16 = torch.bfloat16
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64

# Import PersistentKernel based on backend availability
try:
    from .persistent_kernel import PersistentKernel
    _PERSISTENT_KERNEL_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"PersistentKernel not available: {e}")
    _PERSISTENT_KERNEL_AVAILABLE = False
    # Create a warning version that works in Python-only mode
    from .persistent_kernel import PersistentKernel
from .backend_config import (
    BackendType,
    set_backend,
    get_backend,
    get_current_backend,
    get_available_backends,
    is_backend_available,
    get_backend_info,
    reset_backend
)

# Utility functions for backend management
try:
    from .utils.backend_utils import (
        auto_configure_backend,
        benchmark_backends,
        get_memory_info,
        BackendOptimizer
    )
except ImportError:
    # Gracefully handle missing dependencies
    auto_configure_backend = None
    benchmark_backends = None
    get_memory_info = None
    BackendOptimizer = None


class InputNotFoundError(Exception):
    """Raised when cannot find input tensors"""

    pass


def set_gpu_device_id(device_id: int):
    global_config.gpu_device_id = device_id
    core.set_gpu_device_id(device_id)


def bypass_compile_errors(value: bool = True):
    global_config.bypass_compile_errors = value


def new_kernel_graph(backend=None):
    """Create a new kernel graph with optional backend specification.
    
    Args:
        backend: Backend type to use ('cuda', 'cpu', 'mps', 'auto', or None for current)
    
    Returns:
        KNGraph: A new kernel graph instance
    """
    if backend is not None:
        from .backend_config import BackendType
        if isinstance(backend, str):
            backend = BackendType(backend.lower())
        # Temporarily set backend for this graph creation
        current_backend = get_backend()
        set_backend(backend)
        try:
            kgraph = core.CyKNGraph()
            result = KNGraph(kgraph)
            result._backend = backend
            return result
        finally:
            set_backend(current_backend)
    else:
        kgraph = core.CyKNGraph()
        result = KNGraph(kgraph)
        result._backend = get_backend()
        return result


def new_threadblock_graph(
    grid_dim: tuple, block_dim: tuple, forloop_range: int, reduction_dimx: int
):
    bgraph = core.CyTBGraph(grid_dim, block_dim, forloop_range, reduction_dimx)
    return TBGraph(bgraph)


# Other Configurations
from .global_config import global_config

# Graph Datasets
from .graph_dataset import graph_dataset
from .version import __version__