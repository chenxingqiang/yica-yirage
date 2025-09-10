#!/usr/bin/env python3
"""
YiRage CPU-only Build Script for Mac M3

This script builds YiRage with CPU backend only, without CUDA dependencies.
Specifically optimized for Mac M3 environment.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

def detect_environment():
    """Detect the build environment."""
    print("🔍 Detecting build environment...")
    
    system = platform.system()
    machine = platform.machine()
    
    print(f"  System: {system}")
    print(f"  Architecture: {machine}")
    print(f"  Python: {sys.version}")
    
    # Check for Mac M3
    is_apple_silicon = system == "Darwin" and machine == "arm64"
    if is_apple_silicon:
        print("  ✅ Apple Silicon (M3) detected")
    
    return {
        'system': system,
        'machine': machine,
        'is_apple_silicon': is_apple_silicon,
        'is_mac': system == "Darwin",
        'is_linux': system == "Linux"
    }

def check_dependencies(env_info):
    """Check for required build dependencies."""
    print("\n🔧 Checking build dependencies...")
    
    deps = {
        'cmake': 'cmake --version',
        'make': 'make --version' if not env_info['is_mac'] else 'which make',
        'clang++': 'clang++ --version',
    }
    
    if env_info['is_mac']:
        deps['xcode'] = 'xcode-select --print-path'
    
    missing_deps = []
    
    for name, cmd in deps.items():
        try:
            result = subprocess.run(cmd.split(), 
                                  capture_output=True, 
                                  text=True, 
                                  check=False)
            if result.returncode == 0:
                print(f"  ✅ {name}: Available")
            else:
                print(f"  ❌ {name}: Not available")
                missing_deps.append(name)
        except FileNotFoundError:
            print(f"  ❌ {name}: Not found")
            missing_deps.append(name)
    
    # Check for OpenMP
    try:
        result = subprocess.run(['clang++', '-fopenmp', '-x', 'c++', '-', '-o', '/dev/null'],
                              input='#include <omp.h>\nint main(){return 0;}',
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ OpenMP: Available")
        else:
            print("  ⚠️  OpenMP: Not available (single-threaded CPU backend)")
    except:
        print("  ⚠️  OpenMP: Cannot detect")
    
    return missing_deps

def clean_build_dir():
    """Clean previous build artifacts."""
    print("\n🧹 Cleaning build directory...")
    
    build_dir = Path("build_cpu")
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("  Removed existing build_cpu directory")
    
    build_dir.mkdir(exist_ok=True)
    print("  Created clean build_cpu directory")

def configure_cmake(env_info):
    """Configure CMake for CPU-only build."""
    print("\n⚙️  Configuring CMake for CPU-only build...")
    
    # Use simplified CPU CMakeLists.txt
    # First copy the CPU CMakeLists to build directory
    import shutil
    shutil.copy('CMakeLists_CPU_Simple.txt', 'build_cpu/CMakeLists.txt')
    
    cmake_args = [
        'cmake',
        '-S', 'build_cpu',
        '-B', 'build_cpu/build',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DYIRAGE_USE_CUDA=OFF',
        '-DYIRAGE_USE_CPU=ON',
        '-DYIRAGE_CPU_ONLY=ON',
    ]
    
    if env_info['is_apple_silicon']:
        cmake_args.extend([
            '-DCMAKE_OSX_ARCHITECTURES=arm64',
            '-DYIRAGE_USE_MPS=ON',
            '-DCMAKE_CXX_FLAGS=-march=native -mtune=native -O3',
        ])
    
    # CMakeLists.txt is already copied above
    
    print(f"  Running: {' '.join(cmake_args)}")
    
    try:
        result = subprocess.run(cmake_args, check=True, capture_output=True, text=True)
        print("  ✅ CMake configuration successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ CMake configuration failed:")
        print(f"    stdout: {e.stdout}")
        print(f"    stderr: {e.stderr}")
        return False

def build_project():
    """Build the project."""
    print("\n🔨 Building YiRage CPU version...")
    
    build_cmd = ['cmake', '--build', 'build_cpu/build', '--config', 'Release', '-j']
    
    # Add parallel build jobs
    import multiprocessing
    num_jobs = multiprocessing.cpu_count()
    build_cmd.append(str(num_jobs))
    
    print(f"  Building with {num_jobs} parallel jobs...")
    print(f"  Running: {' '.join(build_cmd)}")
    
    try:
        result = subprocess.run(build_cmd, check=True, capture_output=True, text=True)
        print("  ✅ Build successful")
        print("  Build output:")
        for line in result.stdout.split('\n')[-10:]:  # Show last 10 lines
            if line.strip():
                print(f"    {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Build failed:")
        print(f"    Return code: {e.returncode}")
        print(f"    stdout: {e.stdout}")
        print(f"    stderr: {e.stderr}")
        return False

def test_cpu_build():
    """Test the CPU build."""
    print("\n🧪 Testing CPU build...")
    
    # Check if library was built
    lib_path = Path("build_cpu/build/libyirage_cpu_core.a")
    if lib_path.exists():
        print("  ✅ CPU library built successfully")
        size_mb = lib_path.stat().st_size / (1024 * 1024)
        print(f"  📦 Library size: {size_mb:.1f} MB")
        
        # Try to run the test executable
        test_exe = Path("build_cpu/build/test_cpu_build")
        if test_exe.exists():
            print("  🧪 Running CPU test...")
            try:
                result = subprocess.run([str(test_exe)], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=30)
                if result.returncode == 0:
                    print("  ✅ CPU test passed")
                    print("  Test output:")
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            print(f"    {line}")
                else:
                    print(f"  ⚠️  CPU test failed with code {result.returncode}")
                    print(f"    stderr: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("  ⚠️  CPU test timed out")
            except Exception as e:
                print(f"  ⚠️  CPU test error: {e}")
        
        return True
    else:
        print("  ❌ CPU library not found")
        print("  Expected location:", lib_path)
        # List what was actually built
        build_dir = Path("build_cpu/build")
        if build_dir.exists():
            print("  Files in build directory:")
            for file in build_dir.rglob("*"):
                if file.is_file():
                    print(f"    {file}")
        return False

def create_cpu_setup_py():
    """Create a setup.py specifically for CPU build."""
    print("\n📦 Creating CPU-specific setup.py...")
    
    setup_content = '''"""
YiRage CPU-only Setup Script
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Read version
version_file = Path(__file__).parent / "python" / "yirage" / "version.py"
version_globals = {}
with open(version_file, "r") as f:
    exec(f.read(), version_globals)
__version__ = version_globals["__version__"]

# CPU-only extensions
ext_modules = [
    Pybind11Extension(
        "yirage.core",
        [
            "python/yirage/_cython/core_cpu.cpp",  # We'll create this
        ],
        include_dirs=[
            "include",
            "include/yirage",
            "include/yirage/cpu",
        ],
        define_macros=[
            ("YIRAGE_CPU_ONLY", "1"),
            ("YIRAGE_USE_CPU", "1"),
            ("YIRAGE_USE_CUDA", "0"),
        ],
        cxx_std=17,
        libraries=["yirage_cpu"],
        library_dirs=["build_cpu"],
    ),
]

setup(
    name="yica-yirage",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.4",
        "numpy>=1.21.0",
        "tqdm",
    ],
    zip_safe=False,
)
'''
    
    with open("setup_cpu.py", "w") as f:
        f.write(setup_content)
    
    print("  ✅ Created setup_cpu.py")

def main():
    """Main build function."""
    print("🚀 YiRage CPU-Only Build for Mac M3")
    print("=" * 50)
    
    # Detect environment
    env_info = detect_environment()
    
    # Check dependencies
    missing_deps = check_dependencies(env_info)
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("\nPlease install missing dependencies:")
        if 'cmake' in missing_deps:
            print("  brew install cmake")
        if 'xcode' in missing_deps:
            print("  xcode-select --install")
        return 1
    
    # Clean build directory
    clean_build_dir()
    
    # Configure CMake
    if not configure_cmake(env_info):
        return 1
    
    # Build project
    if not build_project():
        return 1
    
    # Test build
    if not test_cpu_build():
        return 1
    
    # Create CPU setup.py
    create_cpu_setup_py()
    
    print("\n🎉 CPU-only build completed successfully!")
    print("\nNext steps:")
    print("1. Test the build: python3 -c 'import yirage; print(yirage.__version__)'")
    print("2. Build Python package: python3 setup_cpu.py bdist_wheel")
    print("3. Install: pip install dist/*.whl")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
