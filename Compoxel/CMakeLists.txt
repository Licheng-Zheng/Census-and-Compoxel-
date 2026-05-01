cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Compoxel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include(FetchContent)

# ---------------------------------------------------------
# 1. Fetch Jitify FIRST at the top level so it's visible everywhere
# ---------------------------------------------------------
FetchContent_Declare(
    jitify
    GIT_REPOSITORY https://github.com/NVIDIA/jitify.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(jitify)

# ---------------------------------------------------------
# 2. Download and Configure FlameGPU2
# ---------------------------------------------------------
FetchContent_Declare(
    flamegpu2
    GIT_REPOSITORY https://github.com/FLAMEGPU/FLAMEGPU2.git
    GIT_TAG        master
)
set(FLAMEGPU_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FLAMEGPU_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(flamegpu2)

# ---------------------------------------------------------
# 3. Build Your Executable
# ---------------------------------------------------------
add_executable(CompoxelVis
    src/main.cu
)

target_include_directories(CompoxelVis PRIVATE
    ${flamegpu2_SOURCE_DIR}/include
    ${jitify_SOURCE_DIR}          # jitify.hpp sits directly in the repo root
)

target_link_libraries(CompoxelVis PRIVATE flamegpu2)