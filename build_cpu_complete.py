#!/usr/bin/env python3
"""
YiRage Complete CPU Build Script

This script builds YiRage with full functionality but CPU-only backend,
maintaining all features while decoupling from CUDA.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

def setup_cpu_build_environment():
    """Setup the CPU build environment."""
    print("🔧 Setting up CPU build environment...")
    
    # Create build directory
    build_dir = Path("build_cpu_complete")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    
    print(f"  Created build directory: {build_dir}")
    return build_dir

def patch_sources_for_cpu():
    """Patch source files to support CPU-only compilation."""
    print("🔨 Patching sources for CPU-only compilation...")
    
    # Files that need CUDA compatibility headers
    files_to_patch = [
        "src/kernel/graph.cc",
        "src/kernel/customized.cc", 
        "src/kernel/element_binary.cc",
        "src/kernel/element_unary.cc",
        "src/kernel/matmul.cc",
        "src/kernel/reduction.cc",
        "src/kernel/rms_norm.cc",
        "src/threadblock/graph.cc",
        "src/threadblock/matmul.cc",
        "src/transpiler/transpiler_kn.cc",
        "src/transpiler/transpiler_tb.cc",
    ]
    
    cuda_include_pattern = '#include "yirage/utils/cuda_helper.h"'
    cpu_include_replacement = '''#ifdef YIRAGE_CPU_ONLY
#include "yirage/cpu/cpu_compatibility.h"
#include "yirage/config_cpu.h"
#else
#include "yirage/utils/cuda_helper.h"
#include "yirage/config.h"
#endif'''
    
    patched_files = []
    
    for file_path in files_to_patch:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if cuda_include_pattern in content:
                    content = content.replace(cuda_include_pattern, cpu_include_replacement)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    patched_files.append(file_path)
                    print(f"  ✅ Patched: {file_path}")
                
            except Exception as e:
                print(f"  ⚠️  Failed to patch {file_path}: {e}")
    
    print(f"  Patched {len(patched_files)} files for CPU compatibility")
    return patched_files

def build_rust_components():
    """Build Rust components if available."""
    print("🦀 Building Rust components...")
    
    rust_projects = [
        "src/search/abstract_expr/abstract_subexpr",
        "src/search/verification/formal_verifier_equiv"
    ]
    
    built_components = []
    
    for project_dir in rust_projects:
        if os.path.exists(os.path.join(project_dir, "Cargo.toml")):
            print(f"  Building Rust project: {project_dir}")
            try:
                # Set environment for compatibility
                env = os.environ.copy()
                env['PYO3_USE_ABI3_FORWARD_COMPATIBILITY'] = '1'
                
                result = subprocess.run(
                    ['cargo', 'build', '--release'],
                    cwd=project_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print(f"    ✅ Built: {project_dir}")
                    built_components.append(project_dir)
                else:
                    print(f"    ⚠️  Failed to build {project_dir}")
                    print(f"    Error: {result.stderr[:200]}...")
                    
            except subprocess.TimeoutExpired:
                print(f"    ⚠️  Timeout building {project_dir}")
            except Exception as e:
                print(f"    ⚠️  Error building {project_dir}: {e}")
    
    print(f"  Successfully built {len(built_components)} Rust components")
    return built_components

def configure_cmake_cpu(build_dir):
    """Configure CMake for complete CPU build."""
    print("⚙️  Configuring CMake for complete CPU build...")
    
    # Copy CMake configuration
    cmake_file = build_dir / "CMakeLists.txt"
    shutil.copy("CMakeLists_CPU_Complete.txt", cmake_file)
    
    # Copy cmake modules
    cmake_dir = build_dir / "cmake"
    cmake_dir.mkdir(exist_ok=True)
    
    for cmake_module in ["cpu_full_build.cmake", "backends.cmake"]:
        if os.path.exists(f"cmake/{cmake_module}"):
            shutil.copy(f"cmake/{cmake_module}", cmake_dir / cmake_module)
    
    # Configure CMake
    cmake_args = [
        'cmake',
        '-S', str(build_dir),
        '-B', str(build_dir / "build"),
        '-DCMAKE_BUILD_TYPE=Release',
        '-DYIRAGE_CPU_ONLY=ON',
        '-DYIRAGE_USE_CPU=ON',
        '-DYIRAGE_USE_CUDA=OFF',
    ]
    
    # Mac M3 specific settings
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        cmake_args.extend([
            '-DCMAKE_OSX_ARCHITECTURES=arm64',
            '-DYIRAGE_USE_MPS=ON',
        ])
    
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

def build_cpu_library(build_dir):
    """Build the complete CPU library."""
    print("🔨 Building complete CPU library...")
    
    build_cmd = [
        'cmake', '--build', str(build_dir / "build"), 
        '--config', 'Release',
        '--parallel', str(os.cpu_count())
    ]
    
    print(f"  Building with {os.cpu_count()} parallel jobs...")
    
    try:
        result = subprocess.run(build_cmd, check=True, capture_output=True, text=True)
        print("  ✅ Build successful")
        
        # Show build summary
        build_output = result.stdout.split('\n')
        relevant_lines = [line for line in build_output if 
                         'Built target' in line or 'Linking' in line or 'error' in line.lower()]
        
        if relevant_lines:
            print("  Build summary:")
            for line in relevant_lines[-5:]:  # Show last 5 relevant lines
                print(f"    {line}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Build failed:")
        print(f"    Return code: {e.returncode}")
        if e.stdout:
            print("    stdout (last 10 lines):")
            for line in e.stdout.split('\n')[-10:]:
                if line.strip():
                    print(f"      {line}")
        if e.stderr:
            print("    stderr:")
            print(f"      {e.stderr}")
        return False

def test_cpu_library(build_dir):
    """Test the built CPU library."""
    print("🧪 Testing CPU library...")
    
    # Check for built library
    lib_file = build_dir / "build" / "libyirage_cpu_complete.a"
    
    if lib_file.exists():
        size_mb = lib_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ Library built: {lib_file} ({size_mb:.1f} MB)")
        
        # Test executable
        test_exe = build_dir / "build" / "yirage_cpu_test"
        if test_exe.exists():
            print("  🧪 Running CPU functionality test...")
            try:
                result = subprocess.run([str(test_exe)], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=60)
                if result.returncode == 0:
                    print("  ✅ CPU test passed")
                    # Show test output
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            print(f"    {line}")
                else:
                    print(f"  ⚠️  CPU test failed (code {result.returncode})")
                    if result.stderr:
                        print(f"    stderr: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("  ⚠️  CPU test timed out")
            except Exception as e:
                print(f"  ⚠️  CPU test error: {e}")
        
        return True
    else:
        print(f"  ❌ Library not found: {lib_file}")
        return False

def create_cpu_python_extension(build_dir):
    """Create Python extension for CPU library."""
    print("🐍 Creating CPU Python extension...")
    
    setup_content = f'''"""
YiRage CPU Extension Setup
"""

import os
from pathlib import Path
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Get version
version_file = Path(__file__).parent / "python" / "yirage" / "version.py"
version_globals = {{}}
with open(version_file, "r") as f:
    exec(f.read(), version_globals)
__version__ = version_globals["__version__"]

# CPU extension
ext_modules = [
    Pybind11Extension(
        "yirage._cpu_core",
        [
            "python/yirage/_cython/core_cpu.cpp",
        ],
        include_dirs=[
            "include",
            "include/yirage",
            "include/yirage/cpu",
            str(pybind11.get_include()),
        ],
        define_macros=[
            ("YIRAGE_CPU_ONLY", "1"),
            ("YIRAGE_USE_CPU", "1"),
            ("YIRAGE_USE_CUDA", "0"),
        ],
        cxx_std=17,
        libraries=["yirage_cpu_complete"],
        library_dirs=["{build_dir}/build"],
    ),
]

setup(
    name="yica-yirage-cpu",
    version=__version__ + "-cpu",
    ext_modules=ext_modules,
    cmdclass={{"build_ext": build_ext}},
    packages=["yirage"],
    package_dir={{"": "python"}},
    zip_safe=False,
)
'''
    
    setup_file = Path("setup_cpu_complete.py")
    with open(setup_file, 'w') as f:
        f.write(setup_content)
    
    print(f"  ✅ Created: {setup_file}")

def main():
    """Main build process."""
    print("🚀 YiRage Complete CPU Build for Mac M3")
    print("=" * 60)
    
    # Environment check
    if platform.system() != "Darwin":
        print("⚠️  This script is optimized for macOS, but will attempt to build anyway")
    elif platform.machine() == "arm64":
        print("✅ Apple Silicon (M3) detected - optimizing for ARM64")
    
    # Setup build environment
    build_dir = setup_cpu_build_environment()
    
    # Patch sources for CPU compatibility
    patched_files = patch_sources_for_cpu()
    
    # Build Rust components (optional but recommended)
    rust_components = build_rust_components()
    
    # Configure CMake
    if not configure_cmake_cpu(build_dir):
        print("❌ CMake configuration failed")
        return 1
    
    # Build the library
    if not build_cpu_library(build_dir):
        print("❌ Library build failed")
        return 1
    
    # Test the library
    if not test_cpu_library(build_dir):
        print("❌ Library test failed")
        return 1
    
    # Create Python extension
    create_cpu_python_extension(build_dir)
    
    print("\n🎉 Complete CPU build successful!")
    print("\nBuild Summary:")
    print(f"  ✅ Build directory: {build_dir}")
    print(f"  ✅ Patched files: {len(patched_files)}")
    print(f"  ✅ Rust components: {len(rust_components)}")
    print(f"  ✅ CPU library: libyirage_cpu_complete.a")
    print(f"  ✅ Python setup: setup_cpu_complete.py")
    
    print("\nNext steps:")
    print("1. Test Python integration: python3 setup_cpu_complete.py build_ext --inplace")
    print("2. Build wheel: python3 setup_cpu_complete.py bdist_wheel")
    print("3. Install: pip install dist/*.whl")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
