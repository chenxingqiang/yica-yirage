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

"""Utility functions for YiRage backend management."""

import os
import json
import platform
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

from ..backend_config import BackendType, get_available_backends, is_backend_available

logger = logging.getLogger(__name__)

@dataclass
class SystemInfo:
    """System information for backend optimization."""
    cpu_count: int
    total_memory_gb: float
    available_memory_gb: float
    platform: str
    architecture: str
    has_cuda: bool
    has_mps: bool
    cuda_device_count: int
    cuda_devices: List[Dict[str, Any]]
    recommended_backend: str
    recommended_config: Dict[str, Any]

class BackendOptimizer:
    """Optimize backend configuration based on system capabilities."""
    
    def __init__(self):
        self.system_info = self._gather_system_info()
    
    def _gather_system_info(self) -> SystemInfo:
        """Gather comprehensive system information."""
        # Basic system info
        cpu_count = os.cpu_count() or 4
        platform_name = platform.system()
        architecture = platform.machine()
        
        # Memory information
        total_memory_gb = 8.0  # Default fallback
        available_memory_gb = 4.0  # Default fallback
        
        if psutil:
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
        
        # GPU information
        has_cuda = False
        has_mps = False
        cuda_device_count = 0
        cuda_devices = []
        
        if torch:
            # Check CUDA
            if torch.cuda.is_available():
                has_cuda = True
                cuda_device_count = torch.cuda.device_count()
                
                for i in range(cuda_device_count):
                    props = torch.cuda.get_device_properties(i)
                    cuda_devices.append({
                        'id': i,
                        'name': props.name,
                        'total_memory_mb': props.total_memory // (1024 * 1024),
                        'multiprocessor_count': props.multi_processor_count,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
            
            # Check MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                has_mps = True
        
        # Determine recommended backend and configuration
        recommended_backend, recommended_config = self._determine_optimal_backend(
            cpu_count, total_memory_gb, has_cuda, has_mps, cuda_devices
        )
        
        return SystemInfo(
            cpu_count=cpu_count,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            platform=platform_name,
            architecture=architecture,
            has_cuda=has_cuda,
            has_mps=has_mps,
            cuda_device_count=cuda_device_count,
            cuda_devices=cuda_devices,
            recommended_backend=recommended_backend,
            recommended_config=recommended_config
        )
    
    def _determine_optimal_backend(self, cpu_count: int, total_memory_gb: float,
                                 has_cuda: bool, has_mps: bool, 
                                 cuda_devices: List[Dict]) -> Tuple[str, Dict[str, Any]]:
        """Determine the optimal backend and configuration."""
        
        # Priority: CUDA > MPS > CPU
        if has_cuda and cuda_devices:
            # Choose CUDA with optimized configuration
            best_device = max(cuda_devices, key=lambda d: d['total_memory_mb'])
            
            # Configure based on GPU capabilities
            if best_device['total_memory_mb'] >= 24000:  # >= 24GB
                config = {
                    'num_workers': 96,
                    'num_local_schedulers': 48,
                    'batch_size_multiplier': 4
                }
            elif best_device['total_memory_mb'] >= 16000:  # >= 16GB
                config = {
                    'num_workers': 64,
                    'num_local_schedulers': 32,
                    'batch_size_multiplier': 3
                }
            elif best_device['total_memory_mb'] >= 8000:  # >= 8GB
                config = {
                    'num_workers': 32,
                    'num_local_schedulers': 16,
                    'batch_size_multiplier': 2
                }
            else:  # < 8GB
                config = {
                    'num_workers': 16,
                    'num_local_schedulers': 8,
                    'batch_size_multiplier': 1
                }
            
            return 'cuda', config
        
        elif has_mps:
            # Apple Silicon optimization
            config = {
                'num_workers': min(64, cpu_count * 4),
                'num_local_schedulers': min(32, cpu_count * 2),
                'batch_size_multiplier': 2 if total_memory_gb >= 16 else 1
            }
            return 'mps', config
        
        else:
            # CPU optimization
            config = {
                'num_workers': min(cpu_count, 16),
                'num_local_schedulers': min(cpu_count // 2, 8),
                'batch_size_multiplier': 1,
                'use_openmp': True,
                'openmp_threads': cpu_count
            }
            return 'cpu', config
    
    def get_optimal_config(self, backend: Optional[str] = None) -> Dict[str, Any]:
        """Get optimal configuration for specified backend or auto-detected best backend."""
        if backend is None:
            backend = self.system_info.recommended_backend
        
        if backend == 'cuda' and self.system_info.has_cuda:
            return self._get_cuda_config()
        elif backend == 'mps' and self.system_info.has_mps:
            return self._get_mps_config()
        elif backend == 'cpu':
            return self._get_cpu_config()
        else:
            logger.warning(f"Backend {backend} not available, falling back to CPU")
            return self._get_cpu_config()
    
    def _get_cuda_config(self) -> Dict[str, Any]:
        """Get optimized CUDA configuration."""
        if not self.system_info.cuda_devices:
            raise ValueError("No CUDA devices available")
        
        best_device = max(self.system_info.cuda_devices, 
                         key=lambda d: d['total_memory_mb'])
        
        return {
            'backend': 'cuda',
            'device_id': best_device['id'],
            'num_workers': self.system_info.recommended_config.get('num_workers', 32),
            'num_local_schedulers': self.system_info.recommended_config.get('num_local_schedulers', 16),
            'memory_pool_size_mb': best_device['total_memory_mb'] * 0.8,  # 80% of GPU memory
            'use_mixed_precision': True,
            'optimize_for_inference': True
        }
    
    def _get_mps_config(self) -> Dict[str, Any]:
        """Get optimized MPS configuration."""
        return {
            'backend': 'mps',
            'num_workers': self.system_info.recommended_config.get('num_workers', 32),
            'num_local_schedulers': self.system_info.recommended_config.get('num_local_schedulers', 16),
            'use_unified_memory': True,
            'memory_pool_size_mb': min(8192, self.system_info.total_memory_gb * 1024 * 0.3),
            'optimize_for_apple_silicon': True
        }
    
    def _get_cpu_config(self) -> Dict[str, Any]:
        """Get optimized CPU configuration."""
        return {
            'backend': 'cpu',
            'num_workers': self.system_info.recommended_config.get('num_workers', 8),
            'num_local_schedulers': self.system_info.recommended_config.get('num_local_schedulers', 4),
            'use_openmp': True,
            'openmp_threads': self.system_info.cpu_count,
            'memory_pool_size_mb': min(4096, self.system_info.available_memory_gb * 1024 * 0.5),
            'use_blas': True,
            'optimize_for_cpu': True
        }
    
    def print_system_info(self) -> None:
        """Print detailed system information."""
        info = self.system_info
        
        print("YiRage System Information")
        print("=" * 50)
        print(f"Platform: {info.platform} {info.architecture}")
        print(f"CPU Cores: {info.cpu_count}")
        print(f"Total Memory: {info.total_memory_gb:.1f} GB")
        print(f"Available Memory: {info.available_memory_gb:.1f} GB")
        
        print(f"\nGPU Information:")
        if info.has_cuda:
            print(f"CUDA Available: Yes ({info.cuda_device_count} devices)")
            for device in info.cuda_devices:
                print(f"  Device {device['id']}: {device['name']} "
                      f"({device['total_memory_mb']} MB, "
                      f"CC {device['compute_capability']})")
        else:
            print("CUDA Available: No")
        
        if info.has_mps:
            print("MPS Available: Yes")
        else:
            print("MPS Available: No")
        
        print(f"\nRecommended Configuration:")
        print(f"Backend: {info.recommended_backend}")
        for key, value in info.recommended_config.items():
            print(f"  {key}: {value}")
    
    def save_config(self, filename: str, backend: Optional[str] = None) -> None:
        """Save optimal configuration to file."""
        config = self.get_optimal_config(backend)
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {filename}")
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Configuration loaded from {filename}")
        return config

def auto_configure_backend() -> Dict[str, Any]:
    """Automatically configure the best backend for current system."""
    optimizer = BackendOptimizer()
    return optimizer.get_optimal_config()

def benchmark_backends(duration_seconds: int = 30) -> Dict[str, float]:
    """Quick benchmark of available backends."""
    available = get_available_backends()
    results = {}
    
    for backend in available:
        backend_name = backend.value
        if not is_backend_available(backend):
            continue
        
        try:
            import yirage as yr
            yr.set_backend(backend_name)
            
            # Simple benchmark: matrix multiplication
            device = 'cuda' if backend_name == 'cuda' else ('mps' if backend_name == 'mps' else 'cpu')
            
            if torch:
                a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
                b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
                
                # Warmup
                for _ in range(5):
                    _ = torch.matmul(a, b)
                    if device != 'cpu':
                        torch.cuda.synchronize() if device == 'cuda' else None
                
                # Benchmark
                import time
                start_time = time.time()
                iterations = 0
                
                while time.time() - start_time < duration_seconds:
                    _ = torch.matmul(a, b)
                    if device != 'cpu':
                        torch.cuda.synchronize() if device == 'cuda' else None
                    iterations += 1
                
                elapsed = time.time() - start_time
                ops_per_sec = iterations / elapsed
                results[backend_name] = ops_per_sec
            
        except Exception as e:
            logger.warning(f"Failed to benchmark {backend_name}: {e}")
            results[backend_name] = 0.0
    
    return results

def get_memory_info(backend: str) -> Dict[str, float]:
    """Get memory information for specified backend."""
    if not torch:
        return {}
    
    info = {}
    
    if backend == 'cuda' and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            total = props.total_memory / (1024**3)  # GB
            
            info[f'cuda_{i}'] = {
                'total_gb': total,
                'allocated_gb': allocated,
                'cached_gb': cached,
                'free_gb': total - cached
            }
    
    elif backend == 'mps' and hasattr(torch, 'mps') and torch.backends.mps.is_available():
        if hasattr(torch.mps, 'current_allocated_memory'):
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            info['mps'] = {
                'allocated_gb': allocated
            }
    
    elif backend == 'cpu':
        if psutil:
            memory = psutil.virtual_memory()
            info['cpu'] = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            }
    
    return info
