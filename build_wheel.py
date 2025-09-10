#!/usr/bin/env python3
"""
Build YiRage wheel package for distribution
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning previous build artifacts...")
    dirs_to_clean = ['build', 'dist', '*.egg-info', 'python/yica_yirage.egg-info']
    for pattern in dirs_to_clean:
        if '*' in pattern:
            import glob
            for path in glob.glob(pattern):
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"  Removed: {path}")
        else:
            if os.path.exists(pattern):
                shutil.rmtree(pattern)
                print(f"  Removed: {pattern}")

def build_wheel():
    """Build wheel package."""
    print("📦 Building YiRage wheel package...")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYO3_USE_ABI3_FORWARD_COMPATIBILITY'] = '1'
    
    # Try to build with minimal dependencies
    result = subprocess.run(
        [sys.executable, 'setup.py', 'bdist_wheel', '--universal'],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("⚠️  Standard build failed, trying source distribution...")
        print(f"Error: {result.stderr}")
        
        # Fall back to source distribution
        result = subprocess.run(
            [sys.executable, 'setup.py', 'sdist'],
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Source distribution also failed: {result.stderr}")
            return False
    
    print("✅ Build completed successfully")
    return True

def list_packages():
    """List built packages."""
    if os.path.exists('dist'):
        print("\n📋 Built packages:")
        for file in os.listdir('dist'):
            file_path = os.path.join('dist', file)
            size = os.path.getsize(file_path) / 1024 / 1024  # MB
            print(f"  📦 {file} ({size:.1f} MB)")

def main():
    print("🚀 YiRage Wheel Builder")
    print("=" * 50)
    
    # Clean build
    clean_build()
    
    # Build package
    if not build_wheel():
        print("❌ Build failed")
        return 1
    
    # List packages
    list_packages()
    
    print("\n🎉 Build completed!")
    print("\nNext steps:")
    print("1. Test installation: pip install dist/*.whl")
    print("2. Upload to PyPI: twine upload dist/*")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
