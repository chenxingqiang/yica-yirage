"""
Pytest configuration for CPU backend tests
"""

import os
import sys
import pytest
from pathlib import Path

# Force CPU-only environment for all CPU tests
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['YIRAGE_BACKEND'] = 'CPU'
os.environ['YIRAGE_CPU_ONLY'] = '1'

# Add python path for local YiRage import
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "python"))

@pytest.fixture(scope="session", autouse=True)
def setup_cpu_environment():
    """Setup CPU environment for all tests in this directory"""
    print("\n🔧 Setting up CPU test environment...")
    
    # Import and configure YiRage
    import yirage as yr
    
    # Ensure CPU backend is set
    yr.set_backend(yr.BackendType.CPU)
    current = yr.get_current_backend()
    
    print(f"✅ CPU backend configured: {current}")
    print(f"✅ Available backends: {[str(b) for b in yr.get_available_backends()]}")
    
    # Ensure torch uses CPU
    try:
        import torch
        torch.set_default_device('cpu')
        print("✅ PyTorch configured for CPU")
    except ImportError:
        print("⚠️  PyTorch not available")
    
    yield
    
    print("\n🧹 Cleaning up CPU test environment...")

@pytest.fixture
def cpu_graph():
    """Fixture to provide a clean CPU kernel graph for each test"""
    import yirage as yr
    yr.set_backend(yr.BackendType.CPU)
    return yr.new_kernel_graph()

@pytest.fixture
def cpu_tb_graph():
    """Fixture to provide a clean CPU threadblock graph for each test"""
    import yirage as yr
    yr.set_backend(yr.BackendType.CPU)
    return yr.new_threadblock_graph(
        grid_dim=(8, 1, 1),
        block_dim=(32, 1, 1),
        forloop_range=16,
        reduction_dimx=16
    )

@pytest.fixture(params=[(8, 16), (16, 32), (32, 64)])
def tensor_dims(request):
    """Parametrized fixture for different tensor dimensions"""
    return request.param

@pytest.fixture(params=["float16", "float32"])
def dtype_name(request):
    """Parametrized fixture for different data types"""
    return request.param

def pytest_configure(config):
    """Configure pytest for CPU tests"""
    config.addinivalue_line(
        "markers", "cpu: mark test as CPU backend specific"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection for CPU-specific behavior"""
    for item in items:
        # Add cpu marker to all tests in this directory
        item.add_marker(pytest.mark.cpu)
        
        # Mark tests with complex operations as potentially slow
        if any(keyword in item.name.lower() for keyword in ['attention', 'mlp', 'complex']):
            item.add_marker(pytest.mark.slow)
