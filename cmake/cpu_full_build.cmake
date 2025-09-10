# Complete CPU build configuration for YiRage
# Maintains all functionality while decoupling from CUDA

message(STATUS "=== YiRage Complete CPU Build Configuration ===")

# Backend configuration
set(YIRAGE_USE_CUDA OFF CACHE BOOL "Build with CUDA support" FORCE)
set(YIRAGE_USE_CPU ON CACHE BOOL "Build with CPU support" FORCE)
set(YIRAGE_CPU_ONLY ON CACHE BOOL "CPU-only build" FORCE)

# Platform-specific backend enablement
if(APPLE)
    set(YIRAGE_USE_MPS ON CACHE BOOL "Build with MPS support" FORCE)
    message(STATUS "Apple platform detected - enabling MPS backend")
else()
    set(YIRAGE_USE_MPS OFF CACHE BOOL "Build with MPS support" FORCE)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler optimizations for Mac M3
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    message(STATUS "Apple Silicon optimizations enabled")
endif()

# Find dependencies
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP - enabling parallel CPU execution")
    set(YIRAGE_OPENMP_AVAILABLE ON)
else()
    message(STATUS "OpenMP not found - CPU backend will be single-threaded")
    set(YIRAGE_OPENMP_AVAILABLE OFF)
endif()

# Try to find Z3 (optional but recommended)
find_package(Z3 QUIET)
if(Z3_FOUND)
    message(STATUS "Found Z3 solver - enabling advanced optimization")
    set(YIRAGE_Z3_AVAILABLE ON)
else()
    message(STATUS "Z3 solver not found - disabling some optimization features")
    set(YIRAGE_Z3_AVAILABLE OFF)
endif()

# Apple Metal frameworks (for MPS backend)
if(APPLE AND YIRAGE_USE_MPS)
    find_library(METAL_FRAMEWORK Metal)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(METALKIT_FRAMEWORK MetalKit)
    
    if(METAL_FRAMEWORK AND MPS_FRAMEWORK)
        message(STATUS "Found Metal frameworks - MPS backend enabled")
        set(YIRAGE_MPS_FRAMEWORKS_AVAILABLE ON)
    else()
        message(WARNING "Metal frameworks not found - disabling MPS backend")
        set(YIRAGE_USE_MPS OFF)
        set(YIRAGE_MPS_FRAMEWORKS_AVAILABLE OFF)
    endif()
endif()

# Define all CPU-compatible source files (complete list)
set(YIRAGE_CPU_ALL_SOURCES
    # Base functionality
    ${YIRAGE_SOURCE_DIR}/src/base/data_type.cc
    ${YIRAGE_SOURCE_DIR}/src/base/layout.cc
    ${YIRAGE_SOURCE_DIR}/src/layout.cc
    
    # CPU compatibility layer
    src/cpu/cpu_compatibility.cc
    
    # Backend implementations
    src/backend/backend_factory.cc
    src/backend/kernel_factory.cc
    src/backend/cpu/cpu_backend.cc
    src/backend/cpu/cpu_kernels.cc
    
    # Kernel functionality (CPU-compatible)
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
    src/kernel/device_memory_manager_cpu.cc
    
    # Threadblock functionality
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
    
    # Transpiler functionality
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
    
    # Triton transpiler (CPU-compatible)
    src/triton_transpiler/transpile.cc
    src/triton_transpiler/transpile_tb.cc
    
    # Utilities
    src/utils/containers.cc
    src/utils/json_utils.cc
)

# Add NKI transpiler if Z3 is available
if(YIRAGE_Z3_AVAILABLE)
    list(APPEND YIRAGE_CPU_ALL_SOURCES
        src/nki_transpiler/transpile.cc
        src/nki_transpiler/transpile_tb.cc
    )
endif()

# Add MPS backend sources for Apple platforms
if(APPLE AND YIRAGE_USE_MPS AND YIRAGE_MPS_FRAMEWORKS_AVAILABLE)
    list(APPEND YIRAGE_CPU_ALL_SOURCES
        src/backend/mps/mps_backend.mm
    )
endif()

# Search functionality (CPU-compatible)
set(YIRAGE_SEARCH_SOURCES
    src/search/config.cc
    src/search/dim_strategy.cc
    src/search/op_utils.cc
    src/search/order.cc
    src/search/search.cc
    src/search/search_c.cc
    src/search/search_context.cc
    
    # Abstract expression handling
    src/search/abstract_expr/abstract_expr.cc
    src/search/abstract_expr/abstract_expr_eval.cc
    src/search/abstract_expr/abstract_expr_for_ops.cc
    
    # Range propagation
    src/search/range_propagation/irange.cc
    src/search/range_propagation/range.cc
    src/search/range_propagation/tbrange.cc
    
    # Symbolic graph
    src/search/symbolic_graph/dim_var_assignments.cc
    src/search/symbolic_graph/op_args.cc
    src/search/symbolic_graph/symbolic_graph.cc
    src/search/symbolic_graph/symbolic_map.cc
    src/search/symbolic_graph/symbolic_op.cc
    src/search/symbolic_graph/symbolic_tensor.cc
    src/search/symbolic_graph/symbolic_tensor_dim.cc
    src/search/symbolic_graph/tensor_dim_constraint.cc
    src/search/symbolic_graph/tensor_dim_constraints.cc
    src/search/symbolic_graph/tensor_dim_expr.cc
    
    # Verification
    src/search/verification/formal_verifier.cc
    src/search/verification/output_match.cc
    src/search/verification/probabilistic_verifier.cc
)

# Add search sources to main sources
list(APPEND YIRAGE_CPU_ALL_SOURCES ${YIRAGE_SEARCH_SOURCES})

# CUDA-specific sources to exclude
set(YIRAGE_CUDA_SOURCES
    src/kernel/device_memory_manager.cu
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
    src/threadblock/cuda/element_unary.cu
    src/threadblock/cuda/input_executor.cu
    src/threadblock/cuda/matmul.cu
    src/utils/cuda_helper.cu
    src/backend/cuda/cuda_backend.cu
)

# Preprocessor definitions for CPU build
set(YIRAGE_CPU_DEFINITIONS
    YIRAGE_CPU_ONLY=1
    YIRAGE_USE_CPU=1
    YIRAGE_USE_CUDA=0
)

if(YIRAGE_USE_MPS)
    list(APPEND YIRAGE_CPU_DEFINITIONS YIRAGE_USE_MPS=1)
endif()

if(YIRAGE_OPENMP_AVAILABLE)
    list(APPEND YIRAGE_CPU_DEFINITIONS YIRAGE_OPENMP_AVAILABLE=1)
endif()

if(YIRAGE_Z3_AVAILABLE)
    list(APPEND YIRAGE_CPU_DEFINITIONS YIRAGE_Z3_AVAILABLE=1)
endif()

message(STATUS "CPU build configuration:")
message(STATUS "  - Total CPU sources: ${YIRAGE_CPU_ALL_SOURCES}")
message(STATUS "  - Excluded CUDA sources: ${YIRAGE_CUDA_SOURCES}")
message(STATUS "  - OpenMP: ${YIRAGE_OPENMP_AVAILABLE}")
message(STATUS "  - Z3: ${YIRAGE_Z3_AVAILABLE}")
message(STATUS "  - MPS: ${YIRAGE_USE_MPS}")
message(STATUS "=====================================================")
