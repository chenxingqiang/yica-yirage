#!/usr/bin/env python3
"""
YiRage Backend Configuration Templates

This module provides pre-configured templates for different use cases and hardware setups.
"""

from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class BackendTemplate:
    """Base class for backend configuration templates."""
    name: str
    description: str
    backend: str
    num_workers: int
    num_local_schedulers: int
    num_remote_schedulers: int = 0
    recommended_batch_size: int = 1
    memory_pool_size_mb: int = 4096
    additional_config: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        result = asdict(self)
        if self.additional_config:
            result.update(self.additional_config)
        return result

# High-performance CUDA templates
CUDA_HIGH_END = BackendTemplate(
    name="cuda_high_end",
    description="High-end NVIDIA GPU (RTX 4090, A100, H100) - Maximum performance",
    backend="cuda",
    num_workers=96,
    num_local_schedulers=48,
    recommended_batch_size=8,
    memory_pool_size_mb=20480,  # 20GB
    additional_config={
        "use_mixed_precision": True,
        "enable_tensor_cores": True,
        "cuda_graphs": True,
        "memory_fraction": 0.9,
        "compute_capability": "8.0+"
    }
)

CUDA_MID_RANGE = BackendTemplate(
    name="cuda_mid_range",
    description="Mid-range NVIDIA GPU (RTX 3080, RTX 4070) - Balanced performance",
    backend="cuda",
    num_workers=64,
    num_local_schedulers=32,
    recommended_batch_size=4,
    memory_pool_size_mb=10240,  # 10GB
    additional_config={
        "use_mixed_precision": True,
        "enable_tensor_cores": True,
        "cuda_graphs": False,
        "memory_fraction": 0.8,
        "compute_capability": "7.5+"
    }
)

CUDA_LOW_END = BackendTemplate(
    name="cuda_low_end",
    description="Entry-level NVIDIA GPU (GTX 1660, RTX 3060) - Memory optimized",
    backend="cuda",
    num_workers=32,
    num_local_schedulers=16,
    recommended_batch_size=1,
    memory_pool_size_mb=4096,  # 4GB
    additional_config={
        "use_mixed_precision": True,
        "enable_tensor_cores": False,
        "cuda_graphs": False,
        "memory_fraction": 0.7,
        "compute_capability": "7.0+"
    }
)

# Apple Silicon MPS templates
MPS_M3_MAX = BackendTemplate(
    name="mps_m3_max",
    description="Apple M3 Max - Optimized for 36-core GPU",
    backend="mps",
    num_workers=72,
    num_local_schedulers=36,
    recommended_batch_size=4,
    memory_pool_size_mb=16384,  # 16GB unified memory
    additional_config={
        "use_unified_memory": True,
        "metal_performance_shaders": True,
        "apple_silicon_optimizations": True,
        "memory_fraction": 0.6  # Conservative for unified memory
    }
)

MPS_M2_PRO = BackendTemplate(
    name="mps_m2_pro",
    description="Apple M2 Pro - Optimized for 19-core GPU",
    backend="mps",
    num_workers=38,
    num_local_schedulers=19,
    recommended_batch_size=2,
    memory_pool_size_mb=8192,  # 8GB unified memory
    additional_config={
        "use_unified_memory": True,
        "metal_performance_shaders": True,
        "apple_silicon_optimizations": True,
        "memory_fraction": 0.5
    }
)

MPS_M1 = BackendTemplate(
    name="mps_m1",
    description="Apple M1 - Optimized for 8-core GPU",
    backend="mps",
    num_workers=16,
    num_local_schedulers=8,
    recommended_batch_size=1,
    memory_pool_size_mb=4096,  # 4GB unified memory
    additional_config={
        "use_unified_memory": True,
        "metal_performance_shaders": True,
        "apple_silicon_optimizations": True,
        "memory_fraction": 0.4
    }
)

# CPU templates
CPU_HIGH_CORE_COUNT = BackendTemplate(
    name="cpu_high_core",
    description="High core count CPU (32+ cores) - Server/Workstation",
    backend="cpu",
    num_workers=16,
    num_local_schedulers=8,
    recommended_batch_size=1,
    memory_pool_size_mb=8192,
    additional_config={
        "use_openmp": True,
        "openmp_threads": 32,
        "use_blas": True,
        "blas_threads": 16,
        "simd_optimizations": True,
        "numa_aware": True
    }
)

CPU_STANDARD = BackendTemplate(
    name="cpu_standard",
    description="Standard CPU (8-16 cores) - Desktop/Laptop",
    backend="cpu",
    num_workers=8,
    num_local_schedulers=4,
    recommended_batch_size=1,
    memory_pool_size_mb=4096,
    additional_config={
        "use_openmp": True,
        "openmp_threads": 8,
        "use_blas": True,
        "blas_threads": 4,
        "simd_optimizations": True,
        "numa_aware": False
    }
)

CPU_LOW_POWER = BackendTemplate(
    name="cpu_low_power",
    description="Low power CPU (4-8 cores) - Mobile/Embedded",
    backend="cpu",
    num_workers=4,
    num_local_schedulers=2,
    recommended_batch_size=1,
    memory_pool_size_mb=2048,
    additional_config={
        "use_openmp": True,
        "openmp_threads": 4,
        "use_blas": False,  # Might not be available
        "simd_optimizations": False,
        "numa_aware": False,
        "power_efficient": True
    }
)

# Use case specific templates
INFERENCE_OPTIMIZED = BackendTemplate(
    name="inference_optimized",
    description="Optimized for fast inference with low latency",
    backend="auto",  # Will be auto-selected
    num_workers=64,
    num_local_schedulers=32,
    recommended_batch_size=1,
    memory_pool_size_mb=8192,
    additional_config={
        "optimize_for_inference": True,
        "enable_fusion": True,
        "prefetch_data": True,
        "minimize_memory_transfers": True,
        "low_latency_mode": True
    }
)

TRAINING_OPTIMIZED = BackendTemplate(
    name="training_optimized",
    description="Optimized for training with high throughput",
    backend="auto",
    num_workers=96,
    num_local_schedulers=48,
    recommended_batch_size=8,
    memory_pool_size_mb=16384,
    additional_config={
        "optimize_for_training": True,
        "enable_gradient_accumulation": True,
        "high_throughput_mode": True,
        "memory_efficient": False,  # Prefer speed over memory
        "use_mixed_precision": True
    }
)

MEMORY_CONSTRAINED = BackendTemplate(
    name="memory_constrained",
    description="Optimized for systems with limited memory",
    backend="auto",
    num_workers=16,
    num_local_schedulers=8,
    recommended_batch_size=1,
    memory_pool_size_mb=2048,
    additional_config={
        "memory_efficient": True,
        "enable_memory_reuse": True,
        "gradient_checkpointing": True,
        "offload_to_cpu": True,
        "conservative_memory": True
    }
)

DEVELOPMENT = BackendTemplate(
    name="development",
    description="Development and debugging configuration",
    backend="cpu",  # CPU for easier debugging
    num_workers=4,
    num_local_schedulers=2,
    recommended_batch_size=1,
    memory_pool_size_mb=1024,
    additional_config={
        "debug_mode": True,
        "enable_profiling": True,
        "validate_operations": True,
        "verbose_logging": True,
        "deterministic": True
    }
)

# Template registry
TEMPLATE_REGISTRY = {
    # CUDA templates
    "cuda_high_end": CUDA_HIGH_END,
    "cuda_mid_range": CUDA_MID_RANGE,
    "cuda_low_end": CUDA_LOW_END,

    # MPS templates
    "mps_m3_max": MPS_M3_MAX,
    "mps_m2_pro": MPS_M2_PRO,
    "mps_m1": MPS_M1,

    # CPU templates
    "cpu_high_core": CPU_HIGH_CORE_COUNT,
    "cpu_standard": CPU_STANDARD,
    "cpu_low_power": CPU_LOW_POWER,

    # Use case templates
    "inference": INFERENCE_OPTIMIZED,
    "training": TRAINING_OPTIMIZED,
    "memory_constrained": MEMORY_CONSTRAINED,
    "development": DEVELOPMENT,
}

def get_template(name: str) -> BackendTemplate:
    """Get a configuration template by name."""
    if name not in TEMPLATE_REGISTRY:
        available = list(TEMPLATE_REGISTRY.keys())
        raise ValueError(f"Template '{name}' not found. Available templates: {available}")

    return TEMPLATE_REGISTRY[name]

def list_templates() -> Dict[str, str]:
    """List all available templates with descriptions."""
    return {name: template.description for name, template in TEMPLATE_REGISTRY.items()}

def get_recommended_template(system_info: Dict[str, Any]) -> BackendTemplate:
    """Get recommended template based on system information."""
    # Extract system info
    has_cuda = system_info.get('has_cuda', False)
    has_mps = system_info.get('has_mps', False)
    cpu_count = system_info.get('cpu_count', 4)
    total_memory_gb = system_info.get('total_memory_gb', 8)
    cuda_devices = system_info.get('cuda_devices', [])
    platform = system_info.get('platform', 'Unknown')
    architecture = system_info.get('architecture', 'Unknown')

    # CUDA recommendations
    if has_cuda and cuda_devices:
        best_gpu = max(cuda_devices, key=lambda d: d.get('total_memory_mb', 0))
        gpu_memory_gb = best_gpu.get('total_memory_mb', 0) / 1024

        if gpu_memory_gb >= 20:  # High-end GPU
            return CUDA_HIGH_END
        elif gpu_memory_gb >= 10:  # Mid-range GPU
            return CUDA_MID_RANGE
        else:  # Entry-level GPU
            return CUDA_LOW_END

    # MPS recommendations
    elif has_mps and platform == 'Darwin' and architecture == 'arm64':
        if 'M3' in system_info.get('cpu_model', ''):
            return MPS_M3_MAX
        elif 'M2' in system_info.get('cpu_model', ''):
            return MPS_M2_PRO
        else:  # M1 or other
            return MPS_M1

    # CPU recommendations
    else:
        if cpu_count >= 32:
            return CPU_HIGH_CORE_COUNT
        elif cpu_count >= 8:
            return CPU_STANDARD
        else:
            return CPU_LOW_POWER

def create_custom_template(base_template: str, overrides: Dict[str, Any]) -> BackendTemplate:
    """Create a custom template based on an existing one with overrides."""
    base = get_template(base_template)

    # Create a copy and apply overrides
    custom_config = base.to_dict()
    custom_config.update(overrides)

    return BackendTemplate(**custom_config)

def save_template_config(template_name: str, output_file: str) -> None:
    """Save a template configuration to a JSON file."""
    import json

    template = get_template(template_name)
    config = template.to_dict()

    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)

def load_template_from_file(config_file: str) -> BackendTemplate:
    """Load a template configuration from a JSON file."""
    import json

    with open(config_file, 'r') as f:
        config = json.load(f)

    return BackendTemplate(**config)

if __name__ == "__main__":
    # CLI for template management
    import argparse
    import json

    parser = argparse.ArgumentParser(description="YiRage Backend Template Manager")
    subparsers = parser.add_subparsers(dest='command')

    # List templates
    list_parser = subparsers.add_parser('list', help='List available templates')

    # Show template
    show_parser = subparsers.add_parser('show', help='Show template configuration')
    show_parser.add_argument('template', help='Template name')

    # Save template
    save_parser = subparsers.add_parser('save', help='Save template to file')
    save_parser.add_argument('template', help='Template name')
    save_parser.add_argument('output', help='Output file path')

    args = parser.parse_args()

    if args.command == 'list':
        templates = list_templates()
        print("Available Templates:")
        print("=" * 50)
        for name, description in templates.items():
            print(f"{name:20} - {description}")

    elif args.command == 'show':
        try:
            template = get_template(args.template)
            config = template.to_dict()
            print(f"Template: {args.template}")
            print("=" * 50)
            print(json.dumps(config, indent=2))
        except ValueError as e:
            print(f"Error: {e}")

    elif args.command == 'save':
        try:
            save_template_config(args.template, args.output)
            print(f"Template '{args.template}' saved to '{args.output}'")
        except ValueError as e:
            print(f"Error: {e}")

    else:
        parser.print_help()
