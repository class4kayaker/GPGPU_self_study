cmake_minimum_required (VERSION 3.0)
project (ADVECTION_CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
find_package(CUDA QUIET REQUIRED)

# Use HDF5
set (LIB_TYPE SHARED) # or STATIC
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)

find_package(HDF5 COMPONENTS C HL CXX REQUIRED)
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIRS})

set (LINK_LIBS ${LINK_LIBS} ${HDF5_LIBRARIES})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
 
# Specify binary name and source file to build it from
cuda_add_executable(
    advection_CUDA
    advection.cpp
    ../utils/advection_utils.cpp
    ../utils/advection_output.cpp
    advection_kernels.cu)
target_link_libraries(advection_CUDA ${LINK_LIBS})
