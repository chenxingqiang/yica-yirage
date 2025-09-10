# CPU-only build configuration for YiRage
# Specifically designed for Mac M3 and other CPU-only environments

message(STATUS "Configuring YiRage for CPU-only build")

# Disable CUDA completely
set(YIRAGE_USE_CUDA OFF CACHE BOOL "Build with CUDA support" FORCE)
set(YIRAGE_USE_MPS ON CACHE BOOL "Build with MPS support" FORCE)  # Enable for Mac
set(YIRAGE_USE_CPU ON CACHE BOOL "Build with CPU support" FORCE)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific settings for Mac M3
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    message(STATUS "Building for Apple Silicon (arm64)")
    
    # Enable Apple-specific optimizations
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    
    # Check for MPS availability
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    
    if(METAL_FRAMEWORK AND MPS_FRAMEWORK)
        message(STATUS "Found Metal Performance Shaders - enabling MPS backend")
        set(YIRAGE_MPS_AVAILABLE ON)
    else()
        message(WARNING "Metal Performance Shaders not found - disabling MPS backend")
        set(YIRAGE_USE_MPS OFF)
    endif()
endif()

# OpenMP support for CPU parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP - enabling CPU parallelization")
    set(YIRAGE_OPENMP_AVAILABLE ON)
else()
    message(WARNING "OpenMP not found - CPU backend will use single thread")
    set(YIRAGE_OPENMP_AVAILABLE OFF)
endif()

# Define CPU-only source files
set(YIRAGE_CPU_SOURCES
    # Base sources
    src/base/data_type.cc
    src/base/layout.cc
    src/layout.cc
    
    # Backend sources (CPU only)
    src/backend/backend_factory.cc
    src/backend/kernel_factory.cc
    src/backend/cpu/cpu_backend.cc
    src/backend/cpu/cpu_kernels.cc
    
    # Kernel sources (exclude CUDA-specific)
    src/kernel/all_reduce.cc
    src/kernel/chunk.cc
    src/kernel/customized.cc
    src/kernel/device_tensor.cc
    src/kernel/element_binary.cc
    src/kernel/element_unary.cc
    src/kernel/graph.cc
    src/kernel/input.cc
    src/kernel/matmul.cc
    src/kernel/operator.cc
    src/kernel/output.cc
    src/kernel/reduction.cc
    src/kernel/rms_norm.cc
    src/kernel/runtime.cc
    src/kernel/task_register.cc
    src/kernel/triton_code_gen.cc
    
    # Threadblock sources (exclude CUDA-specific)
    src/threadblock/concat.cc
    src/threadblock/element_binary.cc
    src/threadblock/element_unary.cc
    src/threadblock/forloop_accum.cc
    src/threadblock/graph.cc
    src/threadblock/input_loader.cc
    src/threadblock/matmul.cc
    src/threadblock/operator.cc
    src/threadblock/output.cc
    src/threadblock/reduction.cc
    src/threadblock/rms_norm.cc
    src/threadblock/smem_tensor.cc
    
    # Transpiler sources
    src/transpiler/plan_dtensor_memory.cc
    src/transpiler/plan_stensor_memory.cc
    src/transpiler/plan_tb_swizzle.cc
    src/transpiler/resolve_dtensor_meta.cc
    src/transpiler/resolve_tb_fusion.cc
    src/transpiler/resolve_tensor_layout.cc
    src/transpiler/sched_tb_graph.cc
    src/transpiler/transpile.cc
    src/transpiler/transpiler_kn.cc
    src/transpiler/transpiler_tb.cc
    
    # Utilities (exclude CUDA-specific)
    src/utils/containers.cc
    src/utils/json_utils.cc
)

# Conditionally add MPS sources on Apple platforms
if(APPLE AND YIRAGE_USE_MPS)
    list(APPEND YIRAGE_CPU_SOURCES
        src/backend/mps/mps_backend.mm
    )
    # Set Objective-C++ for MPS files
    set_source_files_properties(src/backend/mps/mps_backend.mm PROPERTIES
        COMPILE_FLAGS "-x objective-c++ -fobjc-arc")
endif()

# Exclude CUDA-specific sources
set(YIRAGE_CUDA_SOURCES
    # CUDA kernel sources
    src/kernel/cuda/all_reduce_kernel.cu
    src/kernel/cuda/customized_kernel.cu
    src/kernel/cuda/device_tensor_kernel.cu
    src/kernel/cuda/element_binary_kernel.cu
    src/kernel/cuda/element_unary_kernel.cu
    src/kernel/cuda/input_kernel.cu
    src/kernel/cuda/matmul_kernel.cu
    src/kernel/cuda/output_kernel.cu
    src/kernel/cuda/reduction_kernel.cu
    src/kernel/cuda/rms_norm_kernel.cu
    
    # CUDA threadblock sources
    src/threadblock/cuda/element_unary.cu
    src/threadblock/cuda/input_executor.cu
    src/threadblock/cuda/matmul.cu
    
    # CUDA utilities
    src/utils/cuda_helper.cu
    src/kernel/device_memory_manager.cu
    
    # CUDA backend
    src/backend/cuda/cuda_backend.cu
)

# Define preprocessor macros for CPU-only build
add_definitions(
    -DYIRAGE_USE_CPU=1
    -DYIRAGE_CPU_ONLY=1
)

if(YIRAGE_USE_MPS AND APPLE)
    add_definitions(-DYIRAGE_USE_MPS=1)
endif()

if(YIRAGE_OPENMP_AVAILABLE)
    add_definitions(-DYIRAGE_OPENMP_AVAILABLE=1)
endif()

# Disable CUDA-specific definitions
add_definitions(
    -DYIRAGE_USE_CUDA=0
    -DCUDA_AVAILABLE=0
)

message(STATUS "CPU-only build configuration complete")
message(STATUS "  - CUDA support: DISABLED")
message(STATUS "  - CPU support: ENABLED")
message(STATUS "  - MPS support: ${YIRAGE_USE_MPS}")
message(STATUS "  - OpenMP support: ${YIRAGE_OPENMP_AVAILABLE}")
