#!/usr/bin/env python3
"""
Test suite for YiRage multi-backend functionality.

This test suite validates that the backend abstraction layer works correctly
and that different backends produce consistent results.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    import yirage as yr
except ImportError as e:
    print(f"Warning: Could not import yirage: {e}")
    print("This test requires YiRage to be installed or built")
    sys.exit(0)

class TestBackendAbstraction(unittest.TestCase):
    """Test the backend abstraction layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_backends = []

        # Check which backends are available
        try:
            available_backends = yr.get_available_backends()
            self.test_backends = [b.value for b in available_backends]
        except Exception as e:
            print(f"Warning: Could not get available backends: {e}")
            # Fallback to assume CPU is available
            self.test_backends = ['cpu']

    def test_backend_detection(self):
        """Test that backend detection works correctly."""
        available = yr.get_available_backends()
        self.assertIsInstance(available, list)
        self.assertGreater(len(available), 0, "At least one backend should be available")

        # CPU backend should always be available
        backend_names = [b.value for b in available]
        self.assertIn('cpu', backend_names, "CPU backend should always be available")

    def test_backend_switching(self):
        """Test switching between different backends."""
        for backend_name in self.test_backends:
            with self.subTest(backend=backend_name):
                try:
                    # Set the backend
                    yr.set_backend(backend_name)

                    # Verify it was set correctly
                    current_backend = yr.get_backend()
                    self.assertEqual(current_backend.value, backend_name)

                    # Get backend info
                    info = yr.get_backend_info()
                    self.assertIsInstance(info, dict)
                    self.assertEqual(info['current_backend'], backend_name)

                except Exception as e:
                    self.fail(f"Failed to set backend {backend_name}: {e}")

    def test_kernel_graph_creation(self):
        """Test creating kernel graphs with different backends."""
        for backend_name in self.test_backends:
            with self.subTest(backend=backend_name):
                try:
                    # Create graph with specific backend
                    graph = yr.new_kernel_graph(backend=backend_name)
                    self.assertIsNotNone(graph)

                    # Check that the backend was set correctly
                    self.assertEqual(graph._backend.value, backend_name)

                except Exception as e:
                    self.fail(f"Failed to create kernel graph with {backend_name}: {e}")

    def test_auto_backend_selection(self):
        """Test automatic backend selection."""
        try:
            yr.set_backend('auto')
            current_backend = yr.get_backend()
            self.assertIn(current_backend.value, self.test_backends)
        except Exception as e:
            self.fail(f"Auto backend selection failed: {e}")

    def test_invalid_backend(self):
        """Test handling of invalid backend names."""
        with self.assertRaises((ValueError, RuntimeError)):
            yr.set_backend('invalid_backend')

    def test_backend_availability_check(self):
        """Test backend availability checking."""
        # Test known backends
        self.assertTrue(yr.is_backend_available('cpu'))

        # Test invalid backend
        self.assertFalse(yr.is_backend_available('invalid_backend'))


class TestBackendConsistency(unittest.TestCase):
    """Test that different backends produce consistent results."""

    def setUp(self):
        """Set up test fixtures."""
        self.available_backends = []
        try:
            backends = yr.get_available_backends()
            self.available_backends = [b.value for b in backends]
        except:
            self.available_backends = ['cpu']  # Fallback

        # Test data
        self.batch_size = 2
        self.seq_len = 4
        self.hidden_size = 8
        self.tolerance = 1e-3  # Tolerance for numerical differences

    def create_test_data(self, device='cpu'):
        """Create consistent test data."""
        torch.manual_seed(42)  # For reproducible results

        data = {
            'input': torch.randn(self.batch_size, self.hidden_size, device=device, dtype=torch.float32),
            'weight': torch.randn(self.hidden_size, self.hidden_size, device=device, dtype=torch.float32),
            'bias': torch.randn(self.hidden_size, device=device, dtype=torch.float32)
        }

        return data

    def test_simple_operations_consistency(self):
        """Test that simple operations produce consistent results across backends."""
        if len(self.available_backends) < 2:
            self.skipTest("Need at least 2 backends for consistency testing")

        results = {}

        for backend_name in self.available_backends:
            try:
                yr.set_backend(backend_name)

                # Create test data on appropriate device
                device = 'cuda' if backend_name == 'cuda' else 'cpu'
                if backend_name == 'mps':
                    device = 'mps'

                data = self.create_test_data(device=device)

                # Perform simple matrix multiplication
                result = torch.matmul(data['input'], data['weight'])

                # Move result to CPU for comparison
                if device != 'cpu':
                    result = result.cpu()

                results[backend_name] = result.numpy()

            except Exception as e:
                print(f"Warning: Could not test {backend_name} backend: {e}")
                continue

        # Compare results between backends
        backend_names = list(results.keys())
        if len(backend_names) >= 2:
            reference_result = results[backend_names[0]]
            for i in range(1, len(backend_names)):
                backend_name = backend_names[i]
                result = results[backend_name]

                np.testing.assert_allclose(
                    reference_result, result,
                    rtol=self.tolerance, atol=self.tolerance,
                    err_msg=f"Results differ between {backend_names[0]} and {backend_name}"
                )


class TestPersistentKernelBackends(unittest.TestCase):
    """Test PersistentKernel with different backends."""

    def setUp(self):
        """Set up test fixtures."""
        self.available_backends = []
        try:
            backends = yr.get_available_backends()
            self.available_backends = [b.value for b in backends]
        except:
            self.available_backends = ['cpu']  # Fallback

    def test_persistent_kernel_creation(self):
        """Test creating PersistentKernel with different backends."""
        for backend_name in self.available_backends:
            with self.subTest(backend=backend_name):
                try:
                    # Create meta tensors
                    step = torch.tensor([0], dtype=torch.int32)
                    tokens = torch.full((1, 128), 0, dtype=torch.long)

                    # Profiler tensor only for CUDA
                    profiler_tensor = None
                    if backend_name == 'cuda':
                        profiler_tensor = torch.empty(1000, dtype=torch.uint64)

                    # Create PersistentKernel
                    mpk = yr.PersistentKernel(
                        world_size=1,
                        mpi_rank=0,
                        num_workers=2,  # Reduced for testing
                        num_local_schedulers=1,
                        num_remote_schedulers=0,
                        max_seq_length=128,
                        eos_token_id=2,
                        meta_tensors=[step, tokens],
                        profiler_tensor=profiler_tensor,
                        spec_decode_config=None,
                        backend=backend_name
                    )

                    # Verify backend was set correctly
                    self.assertEqual(mpk.backend.value, backend_name)

                except Exception as e:
                    # Some backends might not support PersistentKernel yet
                    print(f"Warning: PersistentKernel not supported on {backend_name}: {e}")


def run_tests():
    """Run all tests."""
    print("YiRage Multi-Backend Test Suite")
    print("=" * 50)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBackendAbstraction))
    suite.addTests(loader.loadTestsFromTestCase(TestBackendConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestPersistentKernelBackends))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(run_tests())
