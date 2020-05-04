cmake_minimum_required (VERSION 3.10)

find_package(RAJA REQUIRED)

project (RAJA_AdvectionFCT C CXX)

set (LIB_TYPE SHARED) # or SHARED
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)

find_package(HDF5 COMPONENTS C HL CXX REQUIRED)
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIRS})

set (LINK_LIBS ${LINK_LIBS} ${HDF5_LIBRARIES})

add_executable(advection_RAJA advection.cpp ../utils/advection_utils.cpp ../utils/advection_output.cpp)
target_link_libraries(advection_RAJA PUBLIC ${LINK_LIBS} RAJA)
