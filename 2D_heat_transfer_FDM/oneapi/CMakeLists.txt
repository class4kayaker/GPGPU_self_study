cmake_minimum_required (VERSION 3.0)
project (HEAT_ONEAPI_DPCPP)
if(WIN32)
    set(CMAKE_CXX_COMPILER "dpcpp-cl")
else()
	set(CMAKE_CXX_COMPILER "dpcpp")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

OPTION(SHARED_KERNEL "Use SLM Kernel Version - Only for GPU" OFF)
if(SHARED_KERNEL)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_SHARED")
endif(SHARED_KERNEL)

# Use HDF5
set (LIB_TYPE SHARED) # or STATIC
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)

find_package(HDF5 COMPONENTS C HL CXX REQUIRED)
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIRS})

set (LINK_LIBS ${LINK_LIBS} ${HDF5_LIBRARIES} OpenCL sycl ${HDF5_LIBRARIES})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
add_executable (heat_OneAPI ../heat_solve.cpp ../utils/model_data.cpp ../utils/model_utils.cpp)
target_link_libraries(heat_OneAPI ${LINK_LIBS})
