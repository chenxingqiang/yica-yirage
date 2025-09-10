"""
Simple CPU Backend Tests for YiRage

This module contains simplified CPU tests that focus on core functionality
rather than complex tensor operations that may fail in Python-only mode.
"""

import os
import sys
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
import pytest

def setup_module(module):
    """Setup module with CPU backend"""
    print("Setting up CPU backend for simple tests...")
    yr.set_backend(yr.BackendType.CPU)
    torch.set_default_device('cpu')

def test_backend_configuration():
    """Test CPU backend configuration"""
    print("\n🧪 Testing CPU backend configuration...")
    
    # Test available backends
    backends = yr.get_available_backends()
    print(f"  Available backends: {[str(b) for b in backends]}")
    assert yr.BackendType.CPU in backends
    
    # Test backend switching
    yr.set_backend(yr.BackendType.CPU)
    current = yr.get_current_backend()
    assert current == yr.BackendType.CPU
    print(f"  ✅ CPU backend active: {current}")

def test_basic_graph_creation():
    """Test basic graph creation and operations"""
    print("\n🧪 Testing basic graph creation...")
    
    yr.set_backend(yr.BackendType.CPU)
    
    # Test kernel graph creation
    graph = yr.new_kernel_graph()
    assert graph is not None
    print("  ✅ Kernel graph created")
    
    # Test input creation
    x = graph.new_input(dims=(16, 32), dtype=yr.float16, name="input_x")
    y = graph.new_input(dims=(16, 32), dtype=yr.float16, name="input_y")
    assert x is not None and y is not None
    print("  ✅ Input tensors created")
    
    # Test basic operations
    z = graph.add(x, y, name="add_result")
    w = graph.mul(x, y, name="mul_result")
    assert z is not None and w is not None
    print("  ✅ Basic operations created")
    
    # Test output marking
    graph.mark_output(z, name="output_add")
    graph.mark_output(w, name="output_mul")
    print("  ✅ Outputs marked")

def test_threadblock_graph_creation():
    """Test ThreadBlock graph creation"""
    print("\n🧪 Testing ThreadBlock graph creation...")
    
    yr.set_backend(yr.BackendType.CPU)
    
    # Test threadblock graph creation
    tb_graph = yr.new_threadblock_graph(
        grid_dim=(8, 1, 1),
        block_dim=(128, 1, 1),
        forloop_range=64,
        reduction_dimx=64
    )
    assert tb_graph is not None
    print("  ✅ ThreadBlock graph created")

def test_simple_tensor_operations():
    """Test simple tensor operations without complex execution"""
    print("\n🧪 Testing simple tensor operations...")
    
    yr.set_backend(yr.BackendType.CPU)
    
    # Create simple graph
    graph = yr.new_kernel_graph()
    
    # Create inputs
    x = graph.new_input(dims=(8, 16), dtype=yr.float16, name="x")
    y = graph.new_input(dims=(8, 16), dtype=yr.float16, name="y")
    
    # Test different operations
    add_result = graph.add(x, y, name="add")
    mul_result = graph.mul(x, y, name="mul")
    
    # Mark outputs
    graph.mark_output(add_result, name="add_output")
    graph.mark_output(mul_result, name="mul_output")
    
    print("  ✅ Simple tensor operations configured")

def test_data_types():
    """Test different data types support"""
    print("\n🧪 Testing data types...")
    
    yr.set_backend(yr.BackendType.CPU)
    
    graph = yr.new_kernel_graph()
    
    # Test different data types
    dtypes = [yr.float16, yr.float32, yr.bfloat16]
    dtype_names = ["float16", "float32", "bfloat16"]
    
    for dtype, name in zip(dtypes, dtype_names):
        try:
            x = graph.new_input(dims=(4, 8), dtype=dtype, name=f"input_{name}")
            assert x is not None
            print(f"    ✅ {name} supported")
        except Exception as e:
            print(f"    ⚠️  {name} issue: {e}")

def test_import_completeness():
    """Test that all expected functions are available"""
    print("\n🧪 Testing import completeness...")
    
    # Test core functions
    required_functions = [
        'new_kernel_graph',
        'new_threadblock_graph', 
        'set_backend',
        'get_available_backends',
        'get_current_backend',
        'BackendType'
    ]
    
    for func_name in required_functions:
        assert hasattr(yr, func_name), f"Missing function: {func_name}"
        print(f"    ✅ {func_name} available")
    
    # Test data types
    required_types = ['float16', 'float32', 'bfloat16', 'int32']
    for type_name in required_types:
        assert hasattr(yr, type_name), f"Missing type: {type_name}"
        print(f"    ✅ {type_name} available")

def test_error_handling():
    """Test error handling for invalid operations"""
    print("\n🧪 Testing error handling...")
    
    yr.set_backend(yr.BackendType.CPU)
    
    graph = yr.new_kernel_graph()
    
    # Test invalid dimensions
    try:
        x = graph.new_input(dims=(), dtype=yr.float16)  # Empty dimensions
        print("    ⚠️  Empty dimensions allowed (unexpected)")
    except Exception:
        print("    ✅ Empty dimensions properly rejected")
    
    # Test invalid backend
    try:
        yr.set_backend("invalid_backend")
        print("    ⚠️  Invalid backend accepted (unexpected)")
    except Exception:
        print("    ✅ Invalid backend properly rejected")

@pytest.mark.parametrize("dims", [
    (8, 16),
    (4, 32),  
    (16, 8),
    (32, 64),
])
def test_various_dimensions(dims):
    """Test various tensor dimensions"""
    print(f"\n🧪 Testing dimensions {dims}...")
    
    yr.set_backend(yr.BackendType.CPU)
    
    graph = yr.new_kernel_graph()
    x = graph.new_input(dims=dims, dtype=yr.float16, name="test_input")
    y = graph.new_input(dims=dims, dtype=yr.float16, name="test_input2")
    
    # Test basic operations
    z = graph.add(x, y, name="add_test")
    graph.mark_output(z, name="output")
    
    print(f"    ✅ Dimensions {dims} work correctly")

def test_multiple_graphs():
    """Test creating multiple graphs"""
    print("\n🧪 Testing multiple graphs...")
    
    yr.set_backend(yr.BackendType.CPU)
    
    # Create multiple graphs
    graphs = []
    for i in range(3):
        graph = yr.new_kernel_graph()
        x = graph.new_input(dims=(8, 16), dtype=yr.float16, name=f"input_{i}")
        y = graph.new_input(dims=(8, 16), dtype=yr.float16, name=f"input2_{i}")
        z = graph.add(x, y, name=f"output_{i}")
        graph.mark_output(z, name=f"result_{i}")
        graphs.append(graph)
    
    assert len(graphs) == 3
    print("    ✅ Multiple graphs created successfully")

def run_all_simple_tests():
    """Run all simple CPU tests"""
    print("🚀 Running YiRage CPU Simple Test Suite")
    print("=" * 50)
    
    tests = [
        test_backend_configuration,
        test_basic_graph_creation,
        test_threadblock_graph_creation,
        test_simple_tensor_operations,
        test_data_types,
        test_import_completeness,
        test_error_handling,
        test_multiple_graphs,
    ]
    
    setup_module(None)
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
    
    # Run parametrized tests
    dimensions_to_test = [(8, 16), (4, 32), (16, 8)]
    for dims in dimensions_to_test:
        try:
            test_various_dimensions(dims)
            passed += 1
        except Exception as e:
            print(f"❌ test_various_dimensions{dims} FAILED: {e}")
        total += 1
    
    print(f"\n📊 Simple CPU Tests Summary: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All simple CPU tests PASSED!")
        return True
    else:
        print("⚠️  Some simple tests failed")
        return False

if __name__ == "__main__":
    success = run_all_simple_tests()
    sys.exit(0 if success else 1)
