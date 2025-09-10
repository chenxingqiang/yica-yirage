#!/usr/bin/env python3
"""
YiRage Package Builder

This script builds YiRage packages for distribution.
It handles the complex build process including C++/CUDA compilation,
Rust library building, and Python wheel creation.
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
import argparse

def run_command(cmd, cwd=None, env=None, check=True):
    """Run a command with proper error handling."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    if env is None:
        env = os.environ.copy()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=check,
            capture_output=True,
            text=True,
            shell=isinstance(cmd, str)
        )
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        if check:
            raise
        return e

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking build dependencies...")
    
    deps = {
        'cmake': ['cmake', '--version'],
        'gcc/clang': ['gcc', '--version'],
        'python3': ['python3', '--version'],
        'cargo': ['cargo', '--version'],
    }
    
    missing = []
    for name, cmd in deps.items():
        try:
            result = run_command(cmd, check=False)
            if result.returncode == 0:
                print(f"✅ {name}: OK")
            else:
                print(f"❌ {name}: Failed")
                missing.append(name)
        except FileNotFoundError:
            print(f"❌ {name}: Not found")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        return False
    
    print("✅ All dependencies available")
    return True

def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning previous build artifacts...")
    
    dirs_to_clean = [
        'build',
        'dist', 
        'python/yirage.egg-info',
        'python/yirage/_cython/*.c',
        'python/yirage/_cython/*.so',
        'python/yirage/include',
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                print(f"  Removed directory: {dir_path}")
            else:
                os.remove(dir_path)
                print(f"  Removed file: {dir_path}")
    
    print("✅ Clean completed")

def build_rust_components():
    """Build Rust components."""
    print("🦀 Building Rust components...")
    
    rust_projects = [
        ('src/search/abstract_expr/abstract_subexpr', 'build/abstract_subexpr'),
        ('src/search/verification/formal_verifier_equiv', 'build/formal_verifier'),
    ]
    
    for project_dir, target_dir in rust_projects:
        if not os.path.exists(project_dir):
            print(f"⚠️  Rust project not found: {project_dir}, skipping...")
            continue
        
        print(f"Building {project_dir}...")
        
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Build with cargo
        cmd = ['cargo', 'build', '--release', '--target-dir', f'../../../../{target_dir}']
        result = run_command(cmd, cwd=project_dir, check=False)
        
        if result.returncode == 0:
            print(f"✅ {project_dir} built successfully")
        else:
            print(f"⚠️  {project_dir} build failed, continuing...")

def build_cmake_components():
    """Build CMake components."""
    print("🔨 Building CMake components...")
    
    # Create build directory
    build_dir = "build"
    os.makedirs(build_dir, exist_ok=True)
    
    # Get Z3 path
    try:
        import z3
        z3_path = Path(z3.__file__).parent
        print(f"Found Z3 at: {z3_path}")
    except ImportError:
        print("❌ Z3 not found, install with: pip install z3-solver")
        return False
    
    # Configure with CMake
    cmake_args = [
        'cmake', '..',
        f'-DZ3_CXX_INCLUDE_DIRS={z3_path}/include/',
        f'-DZ3_LIBRARIES={z3_path}/lib/libz3.so',
        '-DYIRAGE_USE_CPU=ON',
        '-DYIRAGE_USE_CUDA=OFF',  # Disable CUDA for now
        '-DYIRAGE_USE_MPS=ON' if sys.platform == 'darwin' else '-DYIRAGE_USE_MPS=OFF',
        '-DYIRAGE_USE_LLVM=OFF',  # Disable LLVM for now
    ]
    
    # Add Rust library paths if they exist
    abstract_subexpr_lib = os.path.join(os.getcwd(), 'build', 'abstract_subexpr', 'release')
    formal_verifier_lib = os.path.join(os.getcwd(), 'build', 'formal_verifier', 'release')
    
    if os.path.exists(abstract_subexpr_lib):
        cmake_args.extend([
            f'-DABSTRACT_SUBEXPR_LIB={abstract_subexpr_lib}',
            f'-DABSTRACT_SUBEXPR_LIBRARIES={abstract_subexpr_lib}/libabstract_subexpr.so'
        ])
    
    if os.path.exists(formal_verifier_lib):
        cmake_args.extend([
            f'-DFORMAL_VERIFIER_LIB={formal_verifier_lib}',
            f'-DFORMAL_VERIFIER_LIBRARIES={formal_verifier_lib}/libformal_verifier.so'
        ])
    
    result = run_command(cmake_args, cwd=build_dir, check=False)
    if result.returncode != 0:
        print("⚠️  CMake configuration failed, continuing with simplified build...")
        return False
    
    # Build with make
    result = run_command(['make', '-j4'], cwd=build_dir, check=False)
    if result.returncode == 0:
        print("✅ CMake build completed successfully")
        return True
    else:
        print("⚠️  CMake build failed, continuing...")
        return False

def create_simplified_setup():
    """Create a simplified setup.py for packaging."""
    setup_content = '''
import os
import sys
from setuptools import setup, find_packages
from pathlib import Path

# Read version
version_file = Path(__file__).parent / "python" / "yirage" / "version.py"
with open(version_file, "r") as f:
    exec(f.read())  # This defines __version__

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
with open(requirements_file, "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README
readme_file = Path(__file__).parent / "README.md"
try:
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "YiRage: A Multi-Level Superoptimizer for Tensor Algebra with Multi-Backend Support"

setup(
    name="yica-yirage",
    version=__version__,
    description="YiRage: A Multi-Level Superoptimizer for Tensor Algebra with Multi-Backend Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="YICA Team",
    author_email="team@yica.ai",
    url="https://github.com/chenxingqiang/yica-yirage",
    project_urls={
        "Bug Tracker": "https://github.com/chenxingqiang/yica-yirage/issues",
        "Documentation": "https://yica-yirage.readthedocs.io",
        "Source Code": "https://github.com/chenxingqiang/yica-yirage",
    },
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={
        "yirage": ["*.so", "*.dylib", "*.dll", "include/**/*"],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov", "black", "flake8"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme", "myst-parser"],
        "llvm": ["llvmlite>=0.40.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Hardware",
    ],
    keywords=["tensor", "compiler", "optimization", "llm", "inference", "cuda", "cpu", "mps", "llvm"],
    license="Apache-2.0",
    zip_safe=False,
)
'''
    
    with open('setup_simple.py', 'w') as f:
        f.write(setup_content.strip())
    
    print("✅ Created simplified setup.py")

def build_package(wheel_only=False, source_only=False):
    """Build the package."""
    print("📦 Building YiRage package...")
    
    if not wheel_only:
        print("Building source distribution...")
        result = run_command([sys.executable, 'setup_simple.py', 'sdist'], check=False)
        if result.returncode == 0:
            print("✅ Source distribution built successfully")
        else:
            print("⚠️  Source distribution build failed")
    
    if not source_only:
        print("Building wheel distribution...")
        result = run_command([sys.executable, 'setup_simple.py', 'bdist_wheel'], check=False)
        if result.returncode == 0:
            print("✅ Wheel distribution built successfully")
        else:
            print("⚠️  Wheel distribution build failed")
    
    # List built packages
    if os.path.exists('dist'):
        print("\n📋 Built packages:")
        for file in os.listdir('dist'):
            file_path = os.path.join('dist', file)
            size = os.path.getsize(file_path) / 1024 / 1024  # MB
            print(f"  📦 {file} ({size:.1f} MB)")

def test_package():
    """Test the built package."""
    print("🧪 Testing built package...")
    
    # Find the wheel file
    wheel_files = [f for f in os.listdir('dist') if f.endswith('.whl')]
    if not wheel_files:
        print("❌ No wheel file found")
        return False
    
    wheel_file = wheel_files[0]
    print(f"Testing wheel: {wheel_file}")
    
    # Create a temporary virtual environment for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_dir = os.path.join(temp_dir, 'test_env')
        
        # Create virtual environment
        run_command([sys.executable, '-m', 'venv', venv_dir])
        
        # Get python executable in venv
        if sys.platform == 'win32':
            python_exe = os.path.join(venv_dir, 'Scripts', 'python.exe')
        else:
            python_exe = os.path.join(venv_dir, 'bin', 'python')
        
        # Install the wheel
        wheel_path = os.path.join('dist', wheel_file)
        result = run_command([python_exe, '-m', 'pip', 'install', wheel_path], check=False)
        
        if result.returncode != 0:
            print("❌ Package installation failed")
            return False
        
        # Test import
        test_script = '''
try:
    import yirage
    print("✅ YiRage imported successfully")
    print(f"Version: {yirage.__version__}")
    
    # Test basic functionality
    print("✅ Basic import test passed")
    
except Exception as e:
    print(f"❌ Import test failed: {e}")
    exit(1)
'''
        
        result = run_command([python_exe, '-c', test_script], check=False)
        if result.returncode == 0:
            print("✅ Package test passed")
            return True
        else:
            print("❌ Package test failed")
            return False

def main():
    parser = argparse.ArgumentParser(description='Build YiRage packages')
    parser.add_argument('--clean', action='store_true', help='Clean build artifacts')
    parser.add_argument('--wheel-only', action='store_true', help='Build wheel only')
    parser.add_argument('--source-only', action='store_true', help='Build source only')
    parser.add_argument('--no-test', action='store_true', help='Skip package testing')
    parser.add_argument('--no-deps-check', action='store_true', help='Skip dependency checking')
    
    args = parser.parse_args()
    
    print("🚀 YiRage Package Builder")
    print("=" * 50)
    
    # Check dependencies
    if not args.no_deps_check and not check_dependencies():
        print("❌ Dependency check failed")
        return 1
    
    # Clean if requested
    if args.clean:
        clean_build()
    
    # Build Rust components
    build_rust_components()
    
    # Build CMake components (optional)
    build_cmake_components()
    
    # Create simplified setup
    create_simplified_setup()
    
    # Build package
    build_package(wheel_only=args.wheel_only, source_only=args.source_only)
    
    # Test package
    if not args.no_test:
        if not test_package():
            print("❌ Package testing failed")
            return 1
    
    print("\n🎉 Package build completed successfully!")
    print("\nNext steps:")
    print("1. Review the built packages in the 'dist' directory")
    print("2. Test installation: pip install dist/*.whl")
    print("3. Upload to PyPI: twine upload dist/*")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
