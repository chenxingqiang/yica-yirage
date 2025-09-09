#!/usr/bin/env python3
"""
YiRage Multi-Backend Demo

This demo shows how to use YiRage with different backends (CUDA, CPU, MPS).
It demonstrates the new backend abstraction layer and how to switch between backends.
"""

import torch
import yirage as yr
import argparse
import time
import sys

def create_test_tensors(batch_size=1, seq_len=128, hidden_size=768, device='cpu'):
    """Create test tensors for demonstration."""
    torch.manual_seed(42)  # For reproducible results
    
    # Create some sample tensors
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
    weight_tensor = torch.randn(hidden_size, hidden_size * 3, device=device, dtype=torch.float16)
    norm_weight = torch.randn(hidden_size, device=device, dtype=torch.float16)
    
    return input_tensor, weight_tensor, norm_weight

def benchmark_backend(backend_name, num_iterations=10):
    """Benchmark a specific backend."""
    print(f"\n=== Benchmarking {backend_name.upper()} Backend ===")
    
    try:
        # Set the backend
        yr.set_backend(backend_name)
        print(f"✓ Successfully set backend to {backend_name}")
        
        # Get backend info
        info = yr.get_backend_info()
        print(f"Backend info: {info}")
        
        # Create a simple kernel graph
        graph = yr.new_kernel_graph(backend=backend_name)
        print(f"✓ Created kernel graph with {backend_name} backend")
        
        # For this demo, we'll create some simple operations
        device = 'cuda' if backend_name == 'cuda' else 'cpu'
        if backend_name == 'mps':
            device = 'mps'
            
        input_tensor, weight_tensor, norm_weight = create_test_tensors(device=device)
        
        # Time the operations
        start_time = time.time()
        
        for i in range(num_iterations):
            # Simple matrix multiplication as a test
            result = torch.matmul(input_tensor, weight_tensor)
            if backend_name != 'cpu':  # Synchronize for GPU backends
                torch.cuda.synchronize() if backend_name == 'cuda' else None
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
        
        print(f"✓ Average operation time: {avg_time:.2f} ms")
        print(f"✓ Output shape: {result.shape}")
        
        return True, avg_time
        
    except Exception as e:
        print(f"✗ Error with {backend_name} backend: {e}")
        return False, float('inf')

def demonstrate_persistent_kernel(backend_name):
    """Demonstrate PersistentKernel with different backends."""
    print(f"\n=== PersistentKernel Demo with {backend_name.upper()} ===")
    
    try:
        # Create meta tensors
        step = torch.tensor([0], dtype=torch.int32)
        tokens = torch.full((1, 1024), 0, dtype=torch.long)
        
        # Create profiler tensor (optional)
        profiler_tensor = torch.empty(1000, dtype=torch.uint64) if backend_name == 'cuda' else None
        
        # Create PersistentKernel with specified backend
        mpk = yr.PersistentKernel(
            world_size=1,
            mpi_rank=0,
            num_workers=4,  # Reduced for CPU
            num_local_schedulers=2,
            num_remote_schedulers=0,
            max_seq_length=1024,
            eos_token_id=2,
            meta_tensors=[step, tokens],
            profiler_tensor=profiler_tensor,
            spec_decode_config=None,
            backend=backend_name  # Specify backend
        )
        
        print(f"✓ Created PersistentKernel with {backend_name} backend")
        print(f"✓ Backend: {mpk.backend}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating PersistentKernel with {backend_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='YiRage Multi-Backend Demo')
    parser.add_argument('--backend', choices=['cuda', 'cpu', 'mps', 'auto', 'all'], 
                       default='all', help='Backend to test')
    parser.add_argument('--iterations', type=int, default=10, 
                       help='Number of iterations for benchmarking')
    parser.add_argument('--skip-persistent', action='store_true',
                       help='Skip PersistentKernel demonstration')
    
    args = parser.parse_args()
    
    print("YiRage Multi-Backend Demo")
    print("=" * 50)
    
    # Get available backends
    available_backends = yr.get_available_backends()
    print(f"Available backends: {[b.value for b in available_backends]}")
    
    # Determine which backends to test
    if args.backend == 'all':
        backends_to_test = [b.value for b in available_backends]
    else:
        backends_to_test = [args.backend]
    
    results = {}
    
    # Test each backend
    for backend in backends_to_test:
        if backend == 'auto':
            backend = yr.BackendFactory.detect_best_backend().value
            
        if yr.is_backend_available(yr.BackendType(backend)):
            success, avg_time = benchmark_backend(backend, args.iterations)
            results[backend] = {'success': success, 'time': avg_time}
            
            if success and not args.skip_persistent:
                demonstrate_persistent_kernel(backend)
        else:
            print(f"\n{backend.upper()} backend is not available on this system")
            results[backend] = {'success': False, 'time': float('inf')}
    
    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    
    successful_backends = [(k, v) for k, v in results.items() if v['success']]
    if successful_backends:
        # Sort by performance
        successful_backends.sort(key=lambda x: x[1]['time'])
        
        print("Backend Performance Ranking:")
        for i, (backend, result) in enumerate(successful_backends, 1):
            print(f"{i}. {backend.upper()}: {result['time']:.2f} ms")
    else:
        print("No backends were successfully tested!")
        return 1
    
    print("\n✓ Demo completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
