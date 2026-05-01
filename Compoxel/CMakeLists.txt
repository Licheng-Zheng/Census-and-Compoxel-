cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Define the project and required languages
project(Compoxel LANGUAGES CXX CUDA)

# FlameGPU2 requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Automatically compile for the specific GPU attached to your Colab instance (T4, V100, A100, etc.)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ---------------------------------------------------------
# 1. Download and Configure FlameGPU2
# ---------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    flamegpu2
    GIT_REPOSITORY https://github.com/FLAMEGPU/FLAMEGPU2.git
    GIT_TAG        master # You can lock this to a specific release tag later if needed
)

# Crucial for Colab: Disable building FlameGPU's built-in examples and tests to save compilation time
set(FLAMEGPU_BUILD_EXAMPLES OFF CACHE BOOL "Disable building FLAMEGPU2 examples" FORCE)
set(FLAMEGPU_BUILD_TESTS OFF CACHE BOOL "Disable building FLAMEGPU2 tests" FORCE)

# Make FlameGPU2 available to your project
FetchContent_MakeAvailable(flamegpu2)

# ---------------------------------------------------------
# 2. Build Your Executable
# ---------------------------------------------------------
# Point it to your C++ entry point
add_executable(CompoxelVis 
    src/main.cu
)


target_include_directories(CompoxelVis PRIVATE 
    ${flamegpu2_SOURCE_DIR}/include
)


# Link the FlameGPU2 library to your executable
target_link_libraries(CompoxelVis PRIVATE FLAMEGPU2::flamegpu2)

