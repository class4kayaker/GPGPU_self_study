cmake_minimum_required (VERSION 3.0)
project (ADVECTION_SYCL)

list(APPEND CMAKE_MODULE_PATH ~/self-study/computecpp-sdk/cmake/Modules/)

find_package(ComputeCpp REQUIRED)

# Use HDF5
set (LIB_TYPE SHARED) # or STATIC
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)

find_package(HDF5 COMPONENTS C HL CXX REQUIRED)
INCLUDE_DIRECTORIES(${COMPUTECPP_INCLUDE_DIRECTORY} ${HDF5_INCLUDE_DIRS} ${ComputeCpp_DIR}/include)

set (LINK_LIBS ${LINK_LIBS} ${HDF5_LIBRARIES} ${OpenCL_LIBRARY} ${COMPUTECPP_RUNTIME_LIBRARY} ${HDF5_LIBRARIES})

add_executable (advection_SYCL advection.cpp ../utils/advection_utils.cpp ../utils/advection_output.cpp)
add_sycl_to_target(TARGET advection_SYCL SOURCES advection.cpp ../utils/advection_utils.cpp ../utils/advection_output.cpp ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(advection_SYCL ${LINK_LIBS})
