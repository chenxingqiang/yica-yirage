#!/bin/bash
# YiRage Multi-Backend Docker Build Script
# Copyright 2025-2026 YICA TEAM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="yirage/multi-backend"
TAG="latest"
BUILD_ARGS=""
DOCKERFILE="Dockerfile.multi-backend"

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
YiRage Multi-Backend Docker Build Script

Usage: $0 [OPTIONS]

OPTIONS:
    --tag TAG               Docker image tag (default: latest)
    --name NAME             Docker image name (default: yirage/multi-backend)
    --dockerfile FILE       Dockerfile to use (default: Dockerfile.multi-backend)
    --no-cache              Build without using cache
    --build-arg KEY=VALUE   Pass build argument to Docker
    --push                  Push image to registry after build
    --help                  Show this help message

EXAMPLES:
    $0                                      # Build with default settings
    $0 --tag v1.0                          # Build with specific tag
    $0 --no-cache                          # Build without cache
    $0 --build-arg CUDA_VERSION=12.1       # Pass build argument
    $0 --push                              # Build and push to registry

EOF
}

# Parse command line arguments
PUSH_IMAGE=false
USE_CACHE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        --push)
            PUSH_IMAGE=true
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

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_info "YiRage Multi-Backend Docker Build"
print_info "=================================="
print_info "Project root: $PROJECT_ROOT"
print_info "Image name: $IMAGE_NAME:$TAG"
print_info "Dockerfile: $DOCKERFILE"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Dockerfile exists
if [[ ! -f "$SCRIPT_DIR/$DOCKERFILE" ]]; then
    print_error "Dockerfile not found: $SCRIPT_DIR/$DOCKERFILE"
    exit 1
fi

# Check Docker daemon
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    exit 1
fi

print_success "Docker is available and running"

# Build Docker image
print_info "Building Docker image..."

cd "$PROJECT_ROOT"

# Prepare build command
BUILD_CMD="docker build -f docker/$DOCKERFILE -t $IMAGE_NAME:$TAG"

if [[ "$USE_CACHE" == false ]]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

if [[ -n "$BUILD_ARGS" ]]; then
    BUILD_CMD="$BUILD_CMD $BUILD_ARGS"
fi

BUILD_CMD="$BUILD_CMD ."

print_info "Running: $BUILD_CMD"

# Execute build
if eval $BUILD_CMD; then
    print_success "Docker image built successfully: $IMAGE_NAME:$TAG"
else
    print_error "Docker build failed"
    exit 1
fi

# Get image size
IMAGE_SIZE=$(docker images --format "table {{.Size}}" $IMAGE_NAME:$TAG | tail -n 1)
print_info "Image size: $IMAGE_SIZE"

# Show image info
print_info "Image information:"
docker images $IMAGE_NAME:$TAG

# Test the image
print_info "Testing the built image..."
if docker run --rm $IMAGE_NAME:$TAG python3 -c "import yirage; print('✓ YiRage import successful')"; then
    print_success "Image test passed"
else
    print_warning "Image test failed, but build completed"
fi

# Push image if requested
if [[ "$PUSH_IMAGE" == true ]]; then
    print_info "Pushing image to registry..."
    
    if docker push $IMAGE_NAME:$TAG; then
        print_success "Image pushed successfully"
    else
        print_error "Failed to push image"
        exit 1
    fi
fi

# Create usage instructions
cat << EOF

${GREEN}Build completed successfully!${NC}

To run the container:

  # Interactive mode with GPU support
  docker run --gpus all -it --rm \\
    -p 8888:8888 \\
    -v \$(pwd)/data:/workspace/data \\
    -v \$(pwd)/results:/workspace/results \\
    $IMAGE_NAME:$TAG

  # Run specific command
  docker run --gpus all --rm \\
    $IMAGE_NAME:$TAG \\
    yirage_backend_manager.py info

  # Start Jupyter notebook
  docker run --gpus all -p 8888:8888 \\
    $IMAGE_NAME:$TAG \\
    jupyter notebook --allow-root

  # Run benchmark
  docker run --gpus all --rm \\
    -v \$(pwd)/results:/workspace/results \\
    $IMAGE_NAME:$TAG \\
    multi_backend_benchmark.py --output /workspace/results/benchmark.json

Available tools in the container:
  - yirage_backend_manager.py  - Backend management
  - multi_backend_benchmark.py - Performance benchmarking
  - demo_multi_backend.py      - Multi-backend demo
  - jupyter notebook           - Interactive development

EOF

print_success "YiRage multi-backend Docker image is ready!"

exit 0
