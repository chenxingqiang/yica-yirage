# YiRage Backend Configuration
# This file configures which backends to build and their dependencies

# Backend options
option(YIRAGE_USE_CUDA "Build with CUDA backend support" ON)
option(YIRAGE_USE_CPU "Build with CPU backend support" ON)
option(YIRAGE_USE_MPS "Build with MPS backend support" OFF)

# Auto-detect available backends if not explicitly set
if(NOT DEFINED YIRAGE_USE_CUDA AND CMAKE_CUDA_COMPILER)
    set(YIRAGE_USE_CUDA ON)
endif()

if(NOT DEFINED YIRAGE_USE_MPS AND APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    if(METAL_FRAMEWORK AND MPS_FRAMEWORK)
        set(YIRAGE_USE_MPS ON)
    endif()
endif()

# Configure backend-specific settings
if(YIRAGE_USE_CUDA)
    message(STATUS "Configuring CUDA backend")
    find_package(CUDA REQUIRED)
    add_definitions(-DYIRAGE_USE_CUDA)
    
    # Add CUDA-specific sources
    file(GLOB_RECURSE YIRAGE_CUDA_BACKEND_SRCS
        src/backend/cuda/*.cu
        src/backend/cuda/*.cc
    )
    list(APPEND YIRAGE_BACKEND_SRCS ${YIRAGE_CUDA_BACKEND_SRCS})
    
    # Add CUDA libraries
    list(APPEND YIRAGE_BACKEND_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

if(YIRAGE_USE_CPU)
    message(STATUS "Configuring CPU backend")
    add_definitions(-DYIRAGE_USE_CPU)
    
    # Find OpenMP for CPU parallelization
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_definitions(-DYIRAGE_USE_OPENMP)
        list(APPEND YIRAGE_BACKEND_LIBS OpenMP::OpenMP_CXX)
    endif()
    
    # Find BLAS for optimized linear algebra
    find_package(BLAS)
    if(BLAS_FOUND)
        add_definitions(-DYIRAGE_USE_BLAS)
        list(APPEND YIRAGE_BACKEND_LIBS ${BLAS_LIBRARIES})
    endif()
    
    # Add CPU-specific sources
    file(GLOB_RECURSE YIRAGE_CPU_BACKEND_SRCS
        src/backend/cpu/*.cc
        src/backend/cpu/*.cpp
    )
    list(APPEND YIRAGE_BACKEND_SRCS ${YIRAGE_CPU_BACKEND_SRCS})
endif()

if(YIRAGE_USE_MPS)
    message(STATUS "Configuring MPS backend")
    add_definitions(-DYIRAGE_USE_MPS)
    
    # Find Metal and MPS frameworks
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    
    list(APPEND YIRAGE_BACKEND_LIBS 
        ${METAL_FRAMEWORK} 
        ${MPS_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    
    # Add MPS-specific sources
    file(GLOB_RECURSE YIRAGE_MPS_BACKEND_SRCS
        src/backend/mps/*.mm
        src/backend/mps/*.cc
    )
    list(APPEND YIRAGE_BACKEND_SRCS ${YIRAGE_MPS_BACKEND_SRCS})
    
    # Enable Objective-C++ for MPS backend
    set_source_files_properties(${YIRAGE_MPS_BACKEND_SRCS} PROPERTIES
        COMPILE_FLAGS "-x objective-c++"
    )
endif()

# Add common backend sources
file(GLOB YIRAGE_COMMON_BACKEND_SRCS
    src/backend/*.cc
    src/backend/*.cpp
)
list(APPEND YIRAGE_BACKEND_SRCS ${YIRAGE_COMMON_BACKEND_SRCS})

# Backend summary
message(STATUS "YiRage Backend Configuration:")
message(STATUS "  CUDA Backend: ${YIRAGE_USE_CUDA}")
message(STATUS "  CPU Backend:  ${YIRAGE_USE_CPU}")
message(STATUS "  MPS Backend:  ${YIRAGE_USE_MPS}")

# Validate that at least one backend is enabled
if(NOT YIRAGE_USE_CUDA AND NOT YIRAGE_USE_CPU AND NOT YIRAGE_USE_MPS)
    message(FATAL_ERROR "At least one backend must be enabled!")
endif()
