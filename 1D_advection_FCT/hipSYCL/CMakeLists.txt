cmake_minimum_required (VERSION 3.0)
project (ADVECTION_HIPSYCL)

list(APPEND CMAKE_PREFIX_PATH /opt/hipSYCL/lib)

find_package(hipSYCL CONFIG REQUIRED)
find_package(CUDA)

# Use HDF5
set (LIB_TYPE SHARED) # or STATIC
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)

find_package(HDF5 COMPONENTS C HL CXX REQUIRED)
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})

link_libraries (${LINK_LIBS} ${HDF5_LIBRARIES} ${HDF5_LIBRARIES})
add_executable (advection_hipSYCL advection.cpp ../utils/advection_utils.cpp ../utils/advection_output.cpp)
add_sycl_to_target(TARGET advection_hipSYCL SOURCES advection.cpp ../utils/advection_utils.cpp ../utils/advection_output.cpp ${CMAKE_CURRENT_BINARY_DIR})
# target_link_libraries(advection_hipSYCL ${LINK_LIBS})
