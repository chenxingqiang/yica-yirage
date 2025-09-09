#!/usr/bin/env python3
"""
YiRage Multi-Backend LLM Inference Example

This example demonstrates how to use YiRage's multi-backend system for
efficient LLM inference across different hardware configurations.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# Add YiRage to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

try:
    import yirage as yr
    from yirage.utils import BackendOptimizer, auto_configure_backend
except ImportError as e:
    print(f"Error: Could not import YiRage: {e}")
    sys.exit(1)

# Add configs to path for templates
sys.path.insert(0, str(Path(__file__).parent.parent / 'configs'))

try:
    from backend_templates import get_template, get_recommended_template, list_templates
except ImportError:
    print("Warning: Could not import backend templates")
    get_template = None

class MultiBackendLLMInference:
    """Example LLM inference system with multi-backend support."""

    def __init__(self, model_config: Dict, backend: Optional[str] = None):
        self.model_config = model_config
        self.backend = backend
        self.current_backend = None
        self.mpk = None
        self.performance_stats = {}

        # Initialize backend
        self._initialize_backend()

        # Model parameters
        self.batch_size = model_config.get('batch_size', 1)
        self.seq_length = model_config.get('seq_length', 512)
        self.vocab_size = model_config.get('vocab_size', 32000)
        self.hidden_size = model_config.get('hidden_size', 4096)
        self.num_layers = model_config.get('num_layers', 32)
        self.num_heads = model_config.get('num_heads', 32)
        self.head_dim = model_config.get('head_dim', 128)

        print(f"Model Configuration:")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Sequence Length: {self.seq_length}")
        print(f"  Vocab Size: {self.vocab_size}")
        print(f"  Hidden Size: {self.hidden_size}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Backend: {self.current_backend}")

    def _initialize_backend(self):
        """Initialize and configure the backend."""
        if self.backend:
            if not yr.is_backend_available(yr.BackendType(self.backend)):
                available = [b.value for b in yr.get_available_backends()]
                raise ValueError(f"Backend '{self.backend}' not available. "
                               f"Available: {available}")
            yr.set_backend(self.backend)
        else:
            # Auto-configure optimal backend
            yr.set_backend('auto')

        self.current_backend = yr.get_backend().value
        print(f"Using backend: {self.current_backend}")

        # Get backend-specific configuration
        if get_template:
            try:
                # Try to get system info for template recommendation
                optimizer = BackendOptimizer()
                system_info = optimizer.system_info
                template = get_recommended_template(system_info.__dict__)

                print(f"Recommended template: {template.name}")
                print(f"Template description: {template.description}")

                # Apply template configuration
                self.backend_config = template.to_dict()

            except Exception as e:
                print(f"Warning: Could not apply template: {e}")
                self.backend_config = auto_configure_backend()
        else:
            self.backend_config = auto_configure_backend()

    def create_mock_model_weights(self) -> Dict[str, torch.Tensor]:
        """Create mock model weights for demonstration."""
        device = self._get_device()
        dtype = torch.float16

        weights = {}

        # Embedding weights
        weights['embed_tokens'] = torch.randn(
            self.vocab_size, self.hidden_size,
            device=device, dtype=dtype
        )

        # Layer weights
        for layer_idx in range(self.num_layers):
            # Attention weights
            weights[f'layer_{layer_idx}_q_proj'] = torch.randn(
                self.hidden_size, self.hidden_size,
                device=device, dtype=dtype
            )
            weights[f'layer_{layer_idx}_k_proj'] = torch.randn(
                self.hidden_size, self.hidden_size,
                device=device, dtype=dtype
            )
            weights[f'layer_{layer_idx}_v_proj'] = torch.randn(
                self.hidden_size, self.hidden_size,
                device=device, dtype=dtype
            )
            weights[f'layer_{layer_idx}_o_proj'] = torch.randn(
                self.hidden_size, self.hidden_size,
                device=device, dtype=dtype
            )

            # Feed-forward weights
            weights[f'layer_{layer_idx}_gate_proj'] = torch.randn(
                self.hidden_size, self.hidden_size * 4,
                device=device, dtype=dtype
            )
            weights[f'layer_{layer_idx}_up_proj'] = torch.randn(
                self.hidden_size, self.hidden_size * 4,
                device=device, dtype=dtype
            )
            weights[f'layer_{layer_idx}_down_proj'] = torch.randn(
                self.hidden_size * 4, self.hidden_size,
                device=device, dtype=dtype
            )

            # Normalization weights
            weights[f'layer_{layer_idx}_input_layernorm'] = torch.randn(
                self.hidden_size, device=device, dtype=dtype
            )
            weights[f'layer_{layer_idx}_post_attention_layernorm'] = torch.randn(
                self.hidden_size, device=device, dtype=dtype
            )

        # Output weights
        weights['norm'] = torch.randn(self.hidden_size, device=device, dtype=dtype)
        weights['lm_head'] = torch.randn(
            self.vocab_size, self.hidden_size,
            device=device, dtype=dtype
        )

        print(f"Created mock model weights on {device}")
        return weights

    def _get_device(self) -> str:
        """Get appropriate device for current backend."""
        if self.current_backend == 'cuda':
            return 'cuda'
        elif self.current_backend == 'mps':
            return 'mps'
        else:
            return 'cpu'

    def setup_persistent_kernel(self):
        """Setup YiRage PersistentKernel."""
        device = self._get_device()

        # Create meta tensors
        step = torch.tensor([0], dtype=torch.int32, device=device)
        tokens = torch.full(
            (self.batch_size, self.seq_length), 0,
            dtype=torch.long, device=device
        )
        num_new_tokens = torch.tensor([1], dtype=torch.int32, device=device)

        # Profiler tensor (CUDA only)
        profiler_tensor = None
        if self.current_backend == 'cuda':
            profiler_tensor = torch.empty(
                3000 * 128, dtype=torch.uint64, device=device
            ).contiguous()

        # Get configuration from backend template
        num_workers = self.backend_config.get('num_workers', 32)
        num_local_schedulers = self.backend_config.get('num_local_schedulers', 16)

        print(f"Creating PersistentKernel with:")
        print(f"  Workers: {num_workers}")
        print(f"  Local Schedulers: {num_local_schedulers}")

        try:
            self.mpk = yr.PersistentKernel(
                world_size=1,
                mpi_rank=0,
                num_workers=num_workers,
                num_local_schedulers=num_local_schedulers,
                num_remote_schedulers=0,
                max_seq_length=self.seq_length,
                eos_token_id=2,
                meta_tensors=[step, tokens, num_new_tokens],
                profiler_tensor=profiler_tensor,
                spec_decode_config=None,
                backend=self.current_backend
            )

            print("✓ PersistentKernel created successfully")

        except Exception as e:
            print(f"Warning: Could not create PersistentKernel: {e}")
            print("Continuing with PyTorch operations only")

    def run_simple_inference(self, input_text: str = "Hello world") -> Dict[str, float]:
        """Run simple inference benchmark."""
        device = self._get_device()

        # Create input tokens (mock tokenization)
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, 64),
            device=device, dtype=torch.long
        )

        # Create model weights
        weights = self.create_mock_model_weights()

        print(f"\nRunning inference on {self.current_backend.upper()} backend...")

        # Warm-up runs
        warmup_iterations = 3
        for i in range(warmup_iterations):
            with torch.no_grad():
                output = self._forward_pass(input_ids, weights)
            if device != 'cpu':
                if device == 'cuda':
                    torch.cuda.synchronize()
                # Note: MPS doesn't have explicit synchronize

        # Benchmark runs
        iterations = 10
        times = []

        for i in range(iterations):
            start_time = time.perf_counter()

            with torch.no_grad():
                output = self._forward_pass(input_ids, weights)

            if device != 'cpu':
                if device == 'cuda':
                    torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        # Memory usage
        memory_mb = 0
        if device == 'cuda':
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        elif device == 'mps' and hasattr(torch, 'mps'):
            if hasattr(torch.mps, 'current_allocated_memory'):
                memory_mb = torch.mps.current_allocated_memory() / 1024 / 1024

        results = {
            'backend': self.current_backend,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'throughput_tokens_per_sec': (input_ids.shape[1] * 1000) / avg_time,
            'memory_usage_mb': memory_mb
        }

        print(f"Results:")
        print(f"  Average time: {avg_time:.2f}±{std_time:.2f} ms")
        print(f"  Throughput: {results['throughput_tokens_per_sec']:.1f} tokens/sec")
        print(f"  Memory usage: {memory_mb:.1f} MB")

        return results

    def _forward_pass(self, input_ids: torch.Tensor, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Simple forward pass simulation."""
        x = torch.embedding(input_ids, weights['embed_tokens'])

        # Simplified transformer layers
        for layer_idx in range(min(4, self.num_layers)):  # Only do a few layers for demo
            # Layer norm
            x = torch.layer_norm(x, (self.hidden_size,), weights[f'layer_{layer_idx}_input_layernorm'])

            # Self-attention (simplified)
            q = torch.matmul(x, weights[f'layer_{layer_idx}_q_proj'].T)
            k = torch.matmul(x, weights[f'layer_{layer_idx}_k_proj'].T)
            v = torch.matmul(x, weights[f'layer_{layer_idx}_v_proj'].T)

            # Attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

            # Output projection
            attn_output = torch.matmul(attn_output, weights[f'layer_{layer_idx}_o_proj'].T)
            x = x + attn_output  # Residual connection

            # Feed-forward network
            x_norm = torch.layer_norm(x, (self.hidden_size,), weights[f'layer_{layer_idx}_post_attention_layernorm'])
            gate = torch.matmul(x_norm, weights[f'layer_{layer_idx}_gate_proj'].T)
            up = torch.matmul(x_norm, weights[f'layer_{layer_idx}_up_proj'].T)
            ff_output = torch.silu(gate) * up
            ff_output = torch.matmul(ff_output, weights[f'layer_{layer_idx}_down_proj'].T)
            x = x + ff_output  # Residual connection

        # Final layer norm and projection
        x = torch.layer_norm(x, (self.hidden_size,), weights['norm'])
        logits = torch.matmul(x, weights['lm_head'].T)

        return logits

    def compare_backends(self) -> Dict[str, Dict[str, float]]:
        """Compare performance across all available backends."""
        available_backends = [b.value for b in yr.get_available_backends()]
        results = {}

        original_backend = self.current_backend

        print(f"\nComparing performance across {len(available_backends)} backends...")
        print("=" * 60)

        for backend in available_backends:
            print(f"\nTesting {backend.upper()} backend:")
            print("-" * 30)

            try:
                # Switch backend
                yr.set_backend(backend)
                self.current_backend = backend
                self._initialize_backend()

                # Run inference
                result = self.run_simple_inference()
                results[backend] = result

            except Exception as e:
                print(f"Error with {backend} backend: {e}")
                results[backend] = {'error': str(e)}

        # Restore original backend
        yr.set_backend(original_backend)
        self.current_backend = original_backend

        # Print comparison
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print("="*60)

        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            # Sort by performance
            sorted_results = sorted(
                valid_results.items(),
                key=lambda x: x[1]['avg_time_ms']
            )

            fastest_time = sorted_results[0][1]['avg_time_ms']

            for i, (backend, result) in enumerate(sorted_results, 1):
                speedup = fastest_time / result['avg_time_ms']
                print(f"{i}. {backend.upper():8} - "
                      f"{result['avg_time_ms']:6.2f}ms "
                      f"({result['throughput_tokens_per_sec']:6.1f} tok/s, "
                      f"{speedup:.2f}x speedup)")

        return results

    def save_results(self, results: Dict, filename: str):
        """Save benchmark results to file."""
        output_data = {
            'model_config': self.model_config,
            'backend_config': self.backend_config,
            'results': results,
            'timestamp': time.time()
        }

        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='YiRage Multi-Backend LLM Inference Example')
    parser.add_argument('--backend', choices=['cuda', 'cpu', 'mps', 'auto'], default='auto',
                       help='Backend to use (default: auto)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--seq-length', type=int, default=512,
                       help='Sequence length (default: 512)')
    parser.add_argument('--hidden-size', type=int, default=4096,
                       help='Hidden size (default: 4096)')
    parser.add_argument('--num-layers', type=int, default=32,
                       help='Number of layers (default: 32)')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all available backends')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    parser.add_argument('--template', type=str, default=None,
                       help='Use specific configuration template')

    args = parser.parse_args()

    # Model configuration
    model_config = {
        'batch_size': args.batch_size,
        'seq_length': args.seq_length,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'vocab_size': 32000,
        'num_heads': args.hidden_size // 128,
        'head_dim': 128
    }

    print("YiRage Multi-Backend LLM Inference Example")
    print("=" * 50)

    # Show available templates
    if get_template and args.template is None:
        print("\nAvailable configuration templates:")
        templates = list_templates()
        for name, desc in templates.items():
            print(f"  {name:20} - {desc}")
        print()

    # Create inference system
    backend = args.backend if args.backend != 'auto' else None
    inference_system = MultiBackendLLMInference(model_config, backend)

    # Apply template if specified
    if args.template and get_template:
        try:
            template = get_template(args.template)
            print(f"Applying template: {template.name}")
            print(f"Description: {template.description}")
            inference_system.backend_config.update(template.to_dict())
        except Exception as e:
            print(f"Warning: Could not apply template '{args.template}': {e}")

    # Setup PersistentKernel
    inference_system.setup_persistent_kernel()

    # Run benchmarks
    if args.compare_all:
        results = inference_system.compare_backends()
    else:
        print(f"\nRunning inference benchmark...")
        results = inference_system.run_simple_inference()

    # Save results
    if args.output:
        inference_system.save_results(results, args.output)

    print("\n✓ Example completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
