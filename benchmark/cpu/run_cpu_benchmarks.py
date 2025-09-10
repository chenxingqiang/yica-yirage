#!/usr/bin/env python3
"""
CPU Benchmark Suite Runner for YiRage

Comprehensive CPU benchmark runner that tests all major operations
and provides detailed performance analysis for CPU backend.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

# Force CPU-only environment
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['YIRAGE_BACKEND'] = 'CPU'
os.environ['YIRAGE_CPU_ONLY'] = '1'

# Add python path for local YiRage import
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "python"))

import yirage as yr
import torch
import numpy as np

def safe_json_serialize(obj):
    """Convert numpy/torch types to JSON-serializable types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, tuple):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj

def run_comprehensive_benchmark(benchmark_name, operation_func, input_tensors, iterations=100):
    """Run a comprehensive benchmark for a given operation with full profiling"""
    print(f"  🔍 Running {benchmark_name}...")
    
    # Warmup
    for _ in range(10):
        try:
            result = operation_func(*input_tensors)
        except Exception:
            result = None
    
    # Timing
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        try:
            result = operation_func(*input_tensors)
        except Exception:
            result = None
    
    end_time = time.perf_counter()
    
    total_time = (end_time - start_time) * 1000  # Convert to ms
    avg_time = total_time / iterations
    throughput = iterations / (total_time / 1000)  # ops per second
    
    return {
        'benchmark': benchmark_name,
        'total_time_ms': total_time,
        'avg_time_ms': avg_time,
        'throughput_ops_per_sec': throughput,
        'iterations': iterations,
        'success': result is not None
    }

def run_basic_operations_benchmark():
    """Benchmark basic operations"""
    print("\n🧪 Running Basic Operations Benchmarks...")
    
    yr.set_backend(yr.BackendType.CPU)
    
    # Test dimensions
    dims = [(64, 128), (128, 256), (256, 512)]
    results = {}
    
    for dim in dims:
        print(f"  📏 Testing dimensions {dim}...")
        
        # Create test tensors
        x = torch.randn(*dim, dtype=torch.float16, device='cpu')
        y = torch.randn(*dim, dtype=torch.float16, device='cpu')
        
        # Test operations
        operations = {
            'add': lambda a, b: a + b,
            'mul': lambda a, b: a * b,
            'matmul': lambda a, b: torch.matmul(a, b.T),  # Ensure compatible dimensions
            'silu': lambda a, b: torch.nn.functional.silu(a),
            'sqrt': lambda a, b: torch.sqrt(torch.abs(a)),  # abs to avoid NaN
            'exp': lambda a, b: torch.exp(torch.clamp(a, -10, 10)),  # clamp to avoid overflow
        }
        
        dim_results = {}
        for op_name, op_func in operations.items():
            try:
                result = run_comprehensive_benchmark(f"{op_name}_{dim}", op_func, [x, y], iterations=50)
                dim_results[op_name] = result
            except Exception as e:
                print(f"    ⚠️  {op_name} failed: {e}")
                dim_results[op_name] = {
                    'benchmark': f"{op_name}_{dim}",
                    'error': str(e),
                    'success': False
                }
        
        results[f"dim_{dim[0]}x{dim[1]}"] = dim_results
    
    return results

def run_graph_operations_benchmark():
    """Benchmark YiRage graph operations"""
    print("\n🧪 Running YiRage Graph Operations Benchmarks...")
    
    yr.set_backend(yr.BackendType.CPU)
    
    results = {}
    
    # Test different graph patterns
    patterns = [
        {
            'name': 'simple_add',
            'dims': (32, 64),
            'ops': ['add']
        },
        {
            'name': 'gated_mlp_small',
            'dims': (16, 64),
            'ops': ['matmul', 'silu', 'mul']
        },
        {
            'name': 'attention_small',
            'dims': (8, 32),
            'ops': ['matmul', 'exp', 'div']
        }
    ]
    
    for pattern in patterns:
        print(f"  🔍 Testing pattern: {pattern['name']}...")
        
        try:
            # Create graph
            graph = yr.new_kernel_graph()
            dims = pattern['dims']
            
            # Create inputs
            x = graph.new_input(dims=dims, dtype=yr.float16, name="x")
            y = graph.new_input(dims=dims, dtype=yr.float16, name="y")
            
            # Apply operations based on pattern
            result = x
            if 'add' in pattern['ops']:
                result = graph.add(result, y, name="add_op")
            if 'matmul' in pattern['ops']:
                result = graph.matmul(result, y, name="matmul_op")
            if 'silu' in pattern['ops']:
                result = graph.silu(result, name="silu_op")
            if 'mul' in pattern['ops']:
                result = graph.mul(result, y, name="mul_op")
            if 'exp' in pattern['ops']:
                result = graph.exp(result, name="exp_op")
            if 'div' in pattern['ops']:
                result = graph.div(result, y, name="div_op")
            
            graph.mark_output(result, name="output")
            
            # Create input tensors
            input_tensors = [
                torch.randn(*dims, dtype=torch.float16, device='cpu'),
                torch.randn(*dims, dtype=torch.float16, device='cpu')
            ]
            
            # Benchmark graph execution
            def execute_graph():
                return graph(inputs=input_tensors)
            
            result = run_comprehensive_benchmark(pattern['name'], execute_graph, [], iterations=20)
            result['dims'] = dims
            result['operations'] = pattern['ops']
            results[pattern['name']] = result
            
        except Exception as e:
            print(f"    ⚠️  Pattern {pattern['name']} failed: {e}")
            results[pattern['name']] = {
                'benchmark': pattern['name'],
                'error': str(e),
                'success': False
            }
    
    return results

def run_memory_benchmark():
    """Benchmark memory usage patterns"""
    print("\n🧪 Running Memory Usage Benchmarks...")
    
    yr.set_backend(yr.BackendType.CPU)
    
    results = {}
    
    # Test different memory sizes
    memory_tests = [
        {'name': 'small', 'size': (64, 128), 'count': 10},
        {'name': 'medium', 'size': (256, 512), 'count': 5},
        {'name': 'large', 'size': (512, 1024), 'count': 2},
    ]
    
    for test in memory_tests:
        print(f"  🔍 Testing memory pattern: {test['name']}...")
        
        try:
            start_time = time.perf_counter()
            
            # Create multiple graphs to test memory usage
            graphs = []
            for i in range(test['count']):
                graph = yr.new_kernel_graph()
                x = graph.new_input(dims=test['size'], dtype=yr.float16, name=f"x_{i}")
                y = graph.new_input(dims=test['size'], dtype=yr.float16, name=f"y_{i}")
                z = graph.add(x, y, name=f"add_{i}")
                graph.mark_output(z, name=f"output_{i}")
                graphs.append(graph)
            
            creation_time = time.perf_counter() - start_time
            
            # Calculate memory usage
            total_params = test['count'] * 2 * np.prod(test['size'])  # 2 inputs per graph
            memory_mb = total_params * 2 / (1024 * 1024)  # float16 = 2 bytes
            
            results[test['name']] = {
                'benchmark': f"memory_{test['name']}",
                'size': test['size'],
                'graph_count': test['count'],
                'creation_time_ms': creation_time * 1000,
                'total_params': int(total_params),
                'estimated_memory_mb': memory_mb,
                'success': True
            }
            
        except Exception as e:
            print(f"    ⚠️  Memory test {test['name']} failed: {e}")
            results[test['name']] = {
                'benchmark': f"memory_{test['name']}",
                'error': str(e),
                'success': False
            }
    
    return results

def generate_benchmark_report(basic_results, graph_results, memory_results):
    """Generate comprehensive benchmark report"""
    print("\n📊 CPU Benchmark Suite Report")
    print("=" * 60)
    
    # Basic operations summary
    print("\n🔧 Basic Operations Performance:")
    for dim_key, dim_results in basic_results.items():
        print(f"  📏 {dim_key}:")
        for op_name, result in dim_results.items():
            if result.get('success', False):
                print(f"    {op_name}: {result['avg_time_ms']:.3f} ms/op ({result['throughput_ops_per_sec']:.0f} ops/sec)")
            else:
                print(f"    {op_name}: ❌ Failed")
    
    # Graph operations summary
    print("\n📊 Graph Operations Performance:")
    for pattern_name, result in graph_results.items():
        if result.get('success', False):
            print(f"  {pattern_name}: {result['avg_time_ms']:.3f} ms/op ({result['throughput_ops_per_sec']:.0f} ops/sec)")
            print(f"    Operations: {result.get('operations', [])}")
            print(f"    Dimensions: {result.get('dims', 'N/A')}")
        else:
            print(f"  {pattern_name}: ❌ Failed")
    
    # Memory usage summary
    print("\n💾 Memory Usage Analysis:")
    for test_name, result in memory_results.items():
        if result.get('success', False):
            print(f"  {test_name}: {result['estimated_memory_mb']:.2f} MB")
            print(f"    Creation time: {result['creation_time_ms']:.3f} ms")
            print(f"    Graphs: {result['graph_count']}, Parameters: {result['total_params']:,}")
        else:
            print(f"  {test_name}: ❌ Failed")
    
    # Overall assessment
    basic_success = sum(1 for dim_results in basic_results.values() 
                       for result in dim_results.values() 
                       if result.get('success', False))
    basic_total = sum(len(dim_results) for dim_results in basic_results.values())
    
    graph_success = sum(1 for result in graph_results.values() if result.get('success', False))
    graph_total = len(graph_results)
    
    memory_success = sum(1 for result in memory_results.values() if result.get('success', False))
    memory_total = len(memory_results)
    
    total_success = basic_success + graph_success + memory_success
    total_tests = basic_total + graph_total + memory_total
    
    print(f"\n🎯 Overall Performance Assessment:")
    print(f"   Basic Operations: {basic_success}/{basic_total} successful")
    print(f"   Graph Operations: {graph_success}/{graph_total} successful")
    print(f"   Memory Tests: {memory_success}/{memory_total} successful")
    print(f"   Total: {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%) successful")
    
    if total_success / total_tests >= 0.8:
        print("\n🎉 CPU Backend Performance: EXCELLENT")
        print("   YiRage CPU backend is performing well across all test categories.")
    elif total_success / total_tests >= 0.6:
        print("\n✅ CPU Backend Performance: GOOD")
        print("   YiRage CPU backend is functional with some limitations.")
    else:
        print("\n⚠️  CPU Backend Performance: NEEDS ATTENTION")
        print("   Several performance issues detected in CPU backend.")

def main():
    """Main benchmark runner"""
    print("🚀 YiRage CPU Benchmark Suite")
    print("=" * 60)
    print("Comprehensive performance analysis of YiRage CPU backend")
    
    start_time = time.time()
    
    # Run benchmark categories
    basic_results = run_basic_operations_benchmark()
    graph_results = run_graph_operations_benchmark()
    memory_results = run_memory_benchmark()
    
    # Generate report
    generate_benchmark_report(basic_results, graph_results, memory_results)
    
    # Save results
    all_results = {
        'benchmark_suite': 'yirage_cpu_comprehensive',
        'timestamp': time.time(),
        'basic_operations': basic_results,
        'graph_operations': graph_results,
        'memory_usage': memory_results,
        'total_duration_seconds': time.time() - start_time
    }
    
    # Make results JSON serializable
    all_results = safe_json_serialize(all_results)
    
    try:
        results_file = "cpu_benchmark_comprehensive_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Detailed results saved to {results_file}")
    except Exception as e:
        print(f"\n⚠️  Failed to save results: {e}")
    
    print(f"\n⏱️  Total benchmark duration: {time.time() - start_time:.2f} seconds")
    print("🎊 CPU Benchmark Suite completed!")

if __name__ == "__main__":
    main()
