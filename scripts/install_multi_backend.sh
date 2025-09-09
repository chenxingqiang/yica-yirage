#!/bin/bash
# YiRage Multi-Backend Installation Script
# Copyright 2025-2026 YICA TEAM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
INSTALL_CUDA=auto
INSTALL_CPU=yes
INSTALL_MPS=auto
BUILD_TYPE=Release
PYTHON_VERSION=""
INSTALL_DEPS=yes

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
YiRage Multi-Backend Installation Script

Usage: $0 [OPTIONS]

OPTIONS:
    --cuda-backend [yes|no|auto]    Install CUDA backend (default: auto)
    --cpu-backend [yes|no]          Install CPU backend (default: yes)
    --mps-backend [yes|no|auto]     Install MPS backend (default: auto)
    --build-type [Release|Debug]    Build type (default: Release)
    --python-version VERSION        Specific Python version to use
    --skip-deps                     Skip dependency installation
    --help                          Show this help message

EXAMPLES:
    $0                              # Auto-detect and install available backends
    $0 --cuda-backend yes           # Force CUDA backend installation
    $0 --cpu-backend yes --skip-deps # Install only CPU backend, skip deps
    $0 --build-type Debug           # Install with debug build

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-backend)
            INSTALL_CUDA="$2"
            shift 2
            ;;
        --cpu-backend)
            INSTALL_CPU="$2"
            shift 2
            ;;
        --mps-backend)
            INSTALL_MPS="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --skip-deps)
            INSTALL_DEPS=no
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Detect system information
print_info "Detecting system information..."

OS=$(uname -s)
ARCH=$(uname -m)
print_info "System: $OS $ARCH"

# Detect Python
if [[ -n "$PYTHON_VERSION" ]]; then
    PYTHON_CMD="python$PYTHON_VERSION"
else
    # Try to find Python 3
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.7+ or specify --python-version"
        exit 1
    fi
fi

PYTHON_VERSION_FULL=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
print_info "Python: $PYTHON_VERSION_FULL"

# Auto-detect backend availability
print_info "Auto-detecting backend availability..."

# CUDA detection
if [[ "$INSTALL_CUDA" == "auto" ]]; then
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        print_info "CUDA detected: $CUDA_VERSION"
        INSTALL_CUDA=yes
    else
        print_warning "CUDA not detected, skipping CUDA backend"
        INSTALL_CUDA=no
    fi
fi

# MPS detection (Apple Silicon)
if [[ "$INSTALL_MPS" == "auto" ]]; then
    if [[ "$OS" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
        print_info "Apple Silicon detected, enabling MPS backend"
        INSTALL_MPS=yes
    else
        print_warning "Not on Apple Silicon, skipping MPS backend"
        INSTALL_MPS=no
    fi
fi

# Print installation plan
print_info "Installation plan:"
print_info "  CPU Backend: $INSTALL_CPU"
print_info "  CUDA Backend: $INSTALL_CUDA"
print_info "  MPS Backend: $INSTALL_MPS"
print_info "  Build Type: $BUILD_TYPE"
print_info "  Install Dependencies: $INSTALL_DEPS"

# Check for required tools
print_info "Checking required tools..."

if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Please install CMake 3.24 or higher"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
print_info "CMake: $CMAKE_VERSION"

if ! command -v make &> /dev/null; then
    print_error "Make not found. Please install build tools"
    exit 1
fi

# Install Python dependencies
if [[ "$INSTALL_DEPS" == "yes" ]]; then
    print_info "Installing Python dependencies..."
    
    # Upgrade pip
    $PYTHON_CMD -m pip install --upgrade pip
    
    # Install required packages
    $PYTHON_CMD -m pip install torch numpy cython
    
    # Install optional packages
    $PYTHON_CMD -m pip install psutil || print_warning "Could not install psutil (optional)"
    
    if [[ "$INSTALL_CUDA" == "yes" ]]; then
        print_info "Installing CUDA-specific dependencies..."
        # Install CUDA-specific packages if needed
    fi
    
    print_success "Python dependencies installed"
fi

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_info "Project root: $PROJECT_ROOT"

# Create build directory
BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"

# Configure CMake
print_info "Configuring build with CMake..."

CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DYIRAGE_USE_CPU=$([[ "$INSTALL_CPU" == "yes" ]] && echo "ON" || echo "OFF")"
    "-DYIRAGE_USE_CUDA=$([[ "$INSTALL_CUDA" == "yes" ]] && echo "ON" || echo "OFF")"
    "-DYIRAGE_USE_MPS=$([[ "$INSTALL_MPS" == "yes" ]] && echo "ON" || echo "OFF")"
)

# Add CUDA-specific options
if [[ "$INSTALL_CUDA" == "yes" ]]; then
    CMAKE_ARGS+=("-DCUDA_ARCHITECTURES=70;75;80;86;89;90")
fi

cd "$BUILD_DIR"

print_info "Running: cmake ${CMAKE_ARGS[*]} .."
cmake "${CMAKE_ARGS[@]}" ..

# Build
print_info "Building YiRage..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

print_success "Build completed successfully"

# Install Python package
print_info "Installing Python package..."
cd "$PROJECT_ROOT"

# Install in development mode
$PYTHON_CMD -m pip install -e . -v

print_success "Python package installed"

# Set environment variables
print_info "Setting up environment..."

# Create environment setup script
ENV_SCRIPT="$PROJECT_ROOT/setup_env.sh"
cat > "$ENV_SCRIPT" << EOF
#!/bin/bash
# YiRage Environment Setup
# Source this file to set up YiRage environment

export YIRAGE_HOME="$PROJECT_ROOT"
export PYTHONPATH="\$YIRAGE_HOME/python:\$PYTHONPATH"

# Auto-detect best backend
if [[ "$INSTALL_CUDA" == "yes" ]]; then
    export YIRAGE_BACKEND=cuda
elif [[ "$INSTALL_MPS" == "yes" ]]; then
    export YIRAGE_BACKEND=mps
else
    export YIRAGE_BACKEND=cpu
fi

echo "YiRage environment configured"
echo "Default backend: \$YIRAGE_BACKEND"
echo "Project root: \$YIRAGE_HOME"
EOF

chmod +x "$ENV_SCRIPT"

# Test installation
print_info "Testing installation..."

if $PYTHON_CMD -c "import yirage; print('YiRage version:', yirage.__version__ if hasattr(yirage, '__version__') else 'dev')" 2>/dev/null; then
    print_success "YiRage import test passed"
else
    print_error "YiRage import test failed"
    exit 1
fi

# Test backend functionality
print_info "Testing backend functionality..."

$PYTHON_CMD << EOF
import yirage as yr
try:
    backends = yr.get_available_backends()
    print(f"Available backends: {[b.value for b in backends]}")
    
    if backends:
        # Test first available backend
        backend = backends[0].value
        yr.set_backend(backend)
        print(f"Successfully set backend to: {backend}")
        
        # Test basic functionality
        graph = yr.new_kernel_graph()
        print("Successfully created kernel graph")
    else:
        print("Warning: No backends available")
        
except Exception as e:
    print(f"Backend test failed: {e}")
    exit(1)
EOF

print_success "Backend functionality test passed"

# Print final instructions
print_info "Installation completed successfully!"
echo
print_info "To use YiRage:"
echo "  1. Source the environment: source $PROJECT_ROOT/setup_env.sh"
echo "  2. Run backend manager: python tools/yirage_backend_manager.py info"
echo "  3. Run demo: python demo/demo_multi_backend.py"
echo "  4. Run tests: python tests/test_multi_backend.py"
echo
print_info "For more information, see:"
echo "  - Multi-backend README: docs/MULTI_BACKEND_README.md"
echo "  - Migration guide: docs/MIGRATION_GUIDE.md"
echo "  - Installation guide: INSTALL.md"

print_success "YiRage multi-backend installation complete!"

exit 0
