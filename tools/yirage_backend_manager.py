#!/usr/bin/env python3
"""
YiRage Backend Manager

A command-line tool for managing YiRage backends, including:
- System information and capability detection
- Backend configuration optimization
- Performance benchmarking
- Configuration management
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add YiRage to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))

try:
    import yirage as yr
    from yirage.utils.backend_utils import BackendOptimizer, auto_configure_backend, benchmark_backends, get_memory_info
except ImportError as e:
    print(f"Error: Could not import YiRage: {e}")
    print("Please ensure YiRage is installed or built properly")
    sys.exit(1)

def cmd_info(args):
    """Display system information and backend capabilities."""
    print("YiRage Backend Manager - System Information")
    print("=" * 60)

    # Get available backends
    available = yr.get_available_backends()
    print(f"Available backends: {[b.value for b in available]}")

    # Current backend
    try:
        current = yr.get_backend()
        print(f"Current backend: {current.value}")
    except:
        print("Current backend: Not set")

    # Detailed system information
    optimizer = BackendOptimizer()
    optimizer.print_system_info()

    # Backend-specific information
    for backend in available:
        backend_name = backend.value
        print(f"\n{backend_name.upper()} Backend Details:")
        print("-" * 30)

        try:
            yr.set_backend(backend_name)
            backend_info = yr.get_backend_info()
            print(f"Status: Available")

            # Memory information
            memory_info = get_memory_info(backend_name)
            if memory_info:
                print("Memory Information:")
                for device, info in memory_info.items():
                    print(f"  {device}:")
                    for key, value in info.items():
                        if isinstance(value, float):
                            print(f"    {key}: {value:.2f}")
                        else:
                            print(f"    {key}: {value}")

        except Exception as e:
            print(f"Status: Error - {e}")

def cmd_optimize(args):
    """Generate optimized configuration for specified backend."""
    print("YiRage Backend Optimization")
    print("=" * 40)

    optimizer = BackendOptimizer()

    if args.backend:
        if not yr.is_backend_available(yr.BackendType(args.backend)):
            print(f"Error: Backend '{args.backend}' is not available")
            return 1

        config = optimizer.get_optimal_config(args.backend)
        print(f"Optimized configuration for {args.backend.upper()}:")
    else:
        config = optimizer.get_optimal_config()
        print(f"Optimized configuration for recommended backend:")

    print(json.dumps(config, indent=2))

    # Save to file if requested
    if args.output:
        optimizer.save_config(args.output, args.backend)
        print(f"\nConfiguration saved to: {args.output}")

    # Apply configuration if requested
    if args.apply:
        backend = config['backend']
        yr.set_backend(backend)
        print(f"\nApplied configuration: backend set to {backend}")

def cmd_benchmark(args):
    """Run performance benchmark across backends."""
    print("YiRage Backend Performance Benchmark")
    print("=" * 50)

    duration = args.duration
    print(f"Running benchmark for {duration} seconds per backend...")

    results = benchmark_backends(duration)

    if not results:
        print("No benchmark results available")
        return 1

    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"\nBenchmark Results (operations per second):")
    print("-" * 40)

    for i, (backend, ops_per_sec) in enumerate(sorted_results, 1):
        if ops_per_sec > 0:
            speedup = ops_per_sec / sorted_results[-1][1] if sorted_results[-1][1] > 0 else 0
            print(f"{i}. {backend.upper()}: {ops_per_sec:.1f} ops/sec ({speedup:.2f}x)")
        else:
            print(f"{i}. {backend.upper()}: Failed")

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

def cmd_set(args):
    """Set the current backend."""
    backend = args.backend

    if not yr.is_backend_available(yr.BackendType(backend)):
        print(f"Error: Backend '{backend}' is not available")
        available = [b.value for b in yr.get_available_backends()]
        print(f"Available backends: {available}")
        return 1

    try:
        yr.set_backend(backend)
        print(f"Backend set to: {backend}")

        # Show current configuration
        info = yr.get_backend_info()
        print(f"Backend info: {json.dumps(info, indent=2)}")

    except Exception as e:
        print(f"Error setting backend: {e}")
        return 1

def cmd_reset(args):
    """Reset backend configuration to default."""
    try:
        yr.reset_backend()
        print("Backend configuration reset to default")

        current = yr.get_backend()
        print(f"Current backend: {current.value}")

    except Exception as e:
        print(f"Error resetting backend: {e}")
        return 1

def cmd_config(args):
    """Manage configuration files."""
    if args.action == 'save':
        if not args.file:
            print("Error: --file is required for save action")
            return 1

        config = auto_configure_backend()

        with open(args.file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved to: {args.file}")

    elif args.action == 'load':
        if not args.file:
            print("Error: --file is required for load action")
            return 1

        if not os.path.exists(args.file):
            print(f"Error: Configuration file '{args.file}' not found")
            return 1

        with open(args.file, 'r') as f:
            config = json.load(f)

        # Apply configuration
        backend = config.get('backend', 'auto')
        yr.set_backend(backend)

        print(f"Configuration loaded from: {args.file}")
        print(f"Backend set to: {backend}")

    elif args.action == 'show':
        if args.file:
            if not os.path.exists(args.file):
                print(f"Error: Configuration file '{args.file}' not found")
                return 1

            with open(args.file, 'r') as f:
                config = json.load(f)

            print(f"Configuration from {args.file}:")
        else:
            config = auto_configure_backend()
            print("Current optimal configuration:")

        print(json.dumps(config, indent=2))

def cmd_test(args):
    """Test backend functionality."""
    backend = args.backend

    if not yr.is_backend_available(yr.BackendType(backend)):
        print(f"Error: Backend '{backend}' is not available")
        return 1

    print(f"Testing {backend.upper()} backend...")

    try:
        # Set backend
        yr.set_backend(backend)
        print("✓ Backend set successfully")

        # Test basic operations
        import torch
        device = 'cuda' if backend == 'cuda' else ('mps' if backend == 'mps' else 'cpu')

        # Test tensor creation
        a = torch.randn(100, 100, device=device)
        print("✓ Tensor creation successful")

        # Test basic operations
        b = torch.matmul(a, a)
        print("✓ Matrix multiplication successful")

        c = torch.relu(a)
        print("✓ Activation function successful")

        # Test memory management
        del a, b, c
        if device == 'cuda':
            torch.cuda.empty_cache()
        print("✓ Memory management successful")

        print(f"\n{backend.upper()} backend test: PASSED")

    except Exception as e:
        print(f"\n{backend.upper()} backend test: FAILED")
        print(f"Error: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='YiRage Backend Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s info                    # Show system information
  %(prog)s optimize --backend cuda # Optimize for CUDA backend
  %(prog)s benchmark --duration 10 # Run 10-second benchmark
  %(prog)s set cuda               # Set CUDA as current backend
  %(prog)s test cpu               # Test CPU backend functionality
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show system and backend information')
    info_parser.set_defaults(func=cmd_info)

    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Generate optimized configuration')
    optimize_parser.add_argument('--backend', choices=['cuda', 'cpu', 'mps'],
                               help='Target backend (default: auto-detect)')
    optimize_parser.add_argument('--output', '-o', help='Save configuration to file')
    optimize_parser.add_argument('--apply', action='store_true',
                               help='Apply the configuration immediately')
    optimize_parser.set_defaults(func=cmd_optimize)

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    benchmark_parser.add_argument('--duration', type=int, default=10,
                                help='Benchmark duration per backend (seconds)')
    benchmark_parser.add_argument('--output', '-o', help='Save results to file')
    benchmark_parser.set_defaults(func=cmd_benchmark)

    # Set command
    set_parser = subparsers.add_parser('set', help='Set current backend')
    set_parser.add_argument('backend', choices=['cuda', 'cpu', 'mps', 'auto'],
                          help='Backend to set')
    set_parser.set_defaults(func=cmd_set)

    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset backend configuration')
    reset_parser.set_defaults(func=cmd_reset)

    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration files')
    config_parser.add_argument('action', choices=['save', 'load', 'show'],
                             help='Configuration action')
    config_parser.add_argument('--file', '-f', help='Configuration file path')
    config_parser.set_defaults(func=cmd_config)

    # Test command
    test_parser = subparsers.add_parser('test', help='Test backend functionality')
    test_parser.add_argument('backend', choices=['cuda', 'cpu', 'mps'],
                           help='Backend to test')
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
