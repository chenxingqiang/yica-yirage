#!/usr/bin/env python3
"""
YiRage Multi-Backend Performance Benchmark

This script benchmarks different backends (CUDA, CPU, MPS) across various operations
to help users choose the best backend for their specific use case.
"""

import time
import statistics
import argparse
import json
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
import numpy as np

# Add YiRage to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    import yirage as yr
except ImportError as e:
    print(f"Error: Could not import YiRage: {e}")
    print("Please ensure YiRage is installed or built properly")
    sys.exit(1)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    batch_sizes: List[int]
    sequence_lengths: List[int]
    hidden_sizes: List[int]
    vocab_sizes: List[int]
    iterations: int
    warmup_iterations: int
    dtype: torch.dtype

@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    backend: str
    operation: str
    config: Dict[str, Any]
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float

class MultiBackendBenchmark:
    """Multi-backend benchmark suite for YiRage."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.available_backends = []
        
        # Get available backends
        try:
            self.available_backends = [b.value for b in yr.get_available_backends()]
            print(f"Available backends: {self.available_backends}")
        except Exception as e:
            print(f"Warning: Could not get available backends: {e}")
            self.available_backends = ['cpu']  # Fallback
    
    def benchmark_matrix_multiplication(self, backend: str, m: int, n: int, k: int) -> BenchmarkResult:
        """Benchmark matrix multiplication operation."""
        print(f"  Benchmarking matmul {m}x{n}x{k} on {backend}")
        
        yr.set_backend(backend)
        
        # Determine device
        device = self._get_device_for_backend(backend)
        
        # Create test data
        a = torch.randn(m, k, dtype=self.config.dtype, device=device)
        b = torch.randn(k, n, dtype=self.config.dtype, device=device)
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            result = torch.matmul(a, b)
            if backend != 'cpu':
                torch.cuda.synchronize() if backend == 'cuda' else None
        
        # Benchmark
        times = []
        memory_before = self._get_memory_usage(device)
        
        for _ in range(self.config.iterations):
            start_time = time.perf_counter()
            result = torch.matmul(a, b)
            if backend != 'cpu':
                torch.cuda.synchronize() if backend == 'cuda' else None
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        memory_after = self._get_memory_usage(device)
        memory_usage = max(0, memory_after - memory_before)
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        
        # Calculate throughput (operations per second)
        ops_per_sec = 1000.0 / avg_time if avg_time > 0 else 0
        
        return BenchmarkResult(
            backend=backend,
            operation="matmul",
            config={"m": m, "n": n, "k": k, "dtype": str(self.config.dtype)},
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            throughput_ops_per_sec=ops_per_sec,
            memory_usage_mb=memory_usage
        )
    
    def benchmark_element_wise_operations(self, backend: str, num_elements: int) -> List[BenchmarkResult]:
        """Benchmark element-wise operations (ReLU, GELU, SiLU)."""
        print(f"  Benchmarking element-wise ops ({num_elements} elements) on {backend}")
        
        yr.set_backend(backend)
        device = self._get_device_for_backend(backend)
        
        # Create test data
        input_tensor = torch.randn(num_elements, dtype=self.config.dtype, device=device)
        
        operations = [
            ("relu", torch.relu),
            ("gelu", torch.nn.functional.gelu),
            ("silu", torch.nn.functional.silu)
        ]
        
        results = []
        
        for op_name, op_func in operations:
            # Warmup
            for _ in range(self.config.warmup_iterations):
                result = op_func(input_tensor)
                if backend != 'cpu':
                    torch.cuda.synchronize() if backend == 'cuda' else None
            
            # Benchmark
            times = []
            memory_before = self._get_memory_usage(device)
            
            for _ in range(self.config.iterations):
                start_time = time.perf_counter()
                result = op_func(input_tensor)
                if backend != 'cpu':
                    torch.cuda.synchronize() if backend == 'cuda' else None
                end_time = time.perf_counter()
                
                times.append((end_time - start_time) * 1000)
            
            memory_after = self._get_memory_usage(device)
            memory_usage = max(0, memory_after - memory_before)
            
            # Calculate statistics
            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)
            ops_per_sec = 1000.0 / avg_time if avg_time > 0 else 0
            
            results.append(BenchmarkResult(
                backend=backend,
                operation=op_name,
                config={"num_elements": num_elements, "dtype": str(self.config.dtype)},
                avg_time_ms=avg_time,
                std_time_ms=std_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                throughput_ops_per_sec=ops_per_sec,
                memory_usage_mb=memory_usage
            ))
        
        return results
    
    def benchmark_attention_operations(self, backend: str, batch_size: int, seq_len: int, hidden_size: int) -> BenchmarkResult:
        """Benchmark attention operations."""
        print(f"  Benchmarking attention {batch_size}x{seq_len}x{hidden_size} on {backend}")
        
        yr.set_backend(backend)
        device = self._get_device_for_backend(backend)
        
        # Create test data for attention
        query = torch.randn(batch_size, seq_len, hidden_size, dtype=self.config.dtype, device=device)
        key = torch.randn(batch_size, seq_len, hidden_size, dtype=self.config.dtype, device=device)
        value = torch.randn(batch_size, seq_len, hidden_size, dtype=self.config.dtype, device=device)
        
        # Simple scaled dot-product attention
        def attention_forward(q, k, v):
            scores = torch.matmul(q, k.transpose(-2, -1)) / (hidden_size ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, v)
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            result = attention_forward(query, key, value)
            if backend != 'cpu':
                torch.cuda.synchronize() if backend == 'cuda' else None
        
        # Benchmark
        times = []
        memory_before = self._get_memory_usage(device)
        
        for _ in range(self.config.iterations):
            start_time = time.perf_counter()
            result = attention_forward(query, key, value)
            if backend != 'cpu':
                torch.cuda.synchronize() if backend == 'cuda' else None
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)
        
        memory_after = self._get_memory_usage(device)
        memory_usage = max(0, memory_after - memory_before)
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        ops_per_sec = 1000.0 / avg_time if avg_time > 0 else 0
        
        return BenchmarkResult(
            backend=backend,
            operation="attention",
            config={"batch_size": batch_size, "seq_len": seq_len, "hidden_size": hidden_size, "dtype": str(self.config.dtype)},
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            throughput_ops_per_sec=ops_per_sec,
            memory_usage_mb=memory_usage
        )
    
    def run_comprehensive_benchmark(self) -> None:
        """Run comprehensive benchmark across all backends and operations."""
        print("Starting YiRage Multi-Backend Benchmark")
        print("=" * 60)
        
        for backend in self.available_backends:
            print(f"\nBenchmarking {backend.upper()} backend:")
            print("-" * 40)
            
            try:
                # Matrix multiplication benchmarks
                print("Matrix Multiplication:")
                for hidden_size in self.config.hidden_sizes:
                    result = self.benchmark_matrix_multiplication(
                        backend, 
                        self.config.batch_sizes[0], 
                        hidden_size, 
                        hidden_size
                    )
                    self.results.append(result)
                
                # Element-wise operation benchmarks
                print("Element-wise Operations:")
                for seq_len in self.config.sequence_lengths:
                    for hidden_size in self.config.hidden_sizes:
                        num_elements = seq_len * hidden_size
                        element_results = self.benchmark_element_wise_operations(backend, num_elements)
                        self.results.extend(element_results)
                
                # Attention benchmarks
                print("Attention Operations:")
                for batch_size in self.config.batch_sizes:
                    for seq_len in self.config.sequence_lengths[:2]:  # Limit for memory
                        for hidden_size in self.config.hidden_sizes[:2]:  # Limit for memory
                            result = self.benchmark_attention_operations(
                                backend, batch_size, seq_len, hidden_size
                            )
                            self.results.append(result)
                
            except Exception as e:
                print(f"  Error benchmarking {backend}: {e}")
                continue
    
    def generate_report(self, output_file: str = None) -> None:
        """Generate benchmark report."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        # Group results by operation
        operation_groups = {}
        for result in self.results:
            op_key = f"{result.operation}_{result.config}"
            if op_key not in operation_groups:
                operation_groups[op_key] = []
            operation_groups[op_key].append(result)
        
        # Print summary for each operation
        for op_key, results in operation_groups.items():
            print(f"\n{op_key}:")
            print("-" * 40)
            
            # Sort by average time
            results.sort(key=lambda x: x.avg_time_ms)
            
            for i, result in enumerate(results, 1):
                speedup = results[0].avg_time_ms / result.avg_time_ms if result.avg_time_ms > 0 else 0
                print(f"{i}. {result.backend.upper()}: "
                      f"{result.avg_time_ms:.2f}±{result.std_time_ms:.2f}ms "
                      f"({result.throughput_ops_per_sec:.1f} ops/s, "
                      f"{result.memory_usage_mb:.1f}MB, "
                      f"{speedup:.2f}x)")
        
        # Overall backend ranking
        print(f"\n{'='*60}")
        print("OVERALL BACKEND RANKING")
        print("="*60)
        
        backend_scores = {}
        for result in self.results:
            if result.backend not in backend_scores:
                backend_scores[result.backend] = []
            # Use inverse of time as score (higher is better)
            score = 1.0 / result.avg_time_ms if result.avg_time_ms > 0 else 0
            backend_scores[result.backend].append(score)
        
        # Calculate average scores
        backend_avg_scores = {
            backend: statistics.mean(scores) 
            for backend, scores in backend_scores.items()
        }
        
        # Sort by average score
        ranked_backends = sorted(backend_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (backend, score) in enumerate(ranked_backends, 1):
            print(f"{i}. {backend.upper()}: {score:.4f} (average inverse time score)")
        
        # Save detailed results to JSON if requested
        if output_file:
            self._save_results_to_json(output_file)
            print(f"\nDetailed results saved to: {output_file}")
    
    def _get_device_for_backend(self, backend: str) -> str:
        """Get appropriate device string for backend."""
        if backend == 'cuda':
            return 'cuda'
        elif backend == 'mps':
            return 'mps'
        else:
            return 'cpu'
    
    def _get_memory_usage(self, device: str) -> float:
        """Get memory usage in MB."""
        if device == 'cuda':
            return torch.cuda.memory_allocated() / 1024 / 1024
        elif device == 'mps':
            return torch.mps.current_allocated_memory() / 1024 / 1024 if hasattr(torch, 'mps') else 0
        else:
            return 0  # CPU memory tracking is more complex
    
    def _save_results_to_json(self, filename: str) -> None:
        """Save results to JSON file."""
        json_results = []
        for result in self.results:
            json_results.append({
                'backend': result.backend,
                'operation': result.operation,
                'config': result.config,
                'avg_time_ms': result.avg_time_ms,
                'std_time_ms': result.std_time_ms,
                'min_time_ms': result.min_time_ms,
                'max_time_ms': result.max_time_ms,
                'throughput_ops_per_sec': result.throughput_ops_per_sec,
                'memory_usage_mb': result.memory_usage_mb
            })
        
        with open(filename, 'w') as f:
            json.dump({
                'config': {
                    'batch_sizes': self.config.batch_sizes,
                    'sequence_lengths': self.config.sequence_lengths,
                    'hidden_sizes': self.config.hidden_sizes,
                    'vocab_sizes': self.config.vocab_sizes,
                    'iterations': self.config.iterations,
                    'warmup_iterations': self.config.warmup_iterations,
                    'dtype': str(self.config.dtype)
                },
                'results': json_results
            }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='YiRage Multi-Backend Benchmark')
    parser.add_argument('--iterations', type=int, default=10, 
                       help='Number of benchmark iterations')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for detailed results')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4],
                       help='Batch sizes to test')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[128, 512],
                       help='Sequence lengths to test')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[768, 1024],
                       help='Hidden sizes to test')
    parser.add_argument('--vocab-sizes', type=int, nargs='+', default=[32000, 50000],
                       help='Vocabulary sizes to test')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16',
                       help='Data type for tensors')
    
    args = parser.parse_args()
    
    # Create benchmark configuration
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.seq_lengths,
        hidden_sizes=args.hidden_sizes,
        vocab_sizes=args.vocab_sizes,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        dtype=dtype
    )
    
    # Run benchmark
    benchmark = MultiBackendBenchmark(config)
    benchmark.run_comprehensive_benchmark()
    benchmark.generate_report(args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
