cmake_minimum_required (VERSION 3.0)
project (ADVECTION_OPENACC)

find_package(OpenACC REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS}")

# Use HDF5
set (LIB_TYPE SHARED) # or STATIC
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)

find_package(HDF5 COMPONENTS C HL CXX REQUIRED)
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})

set (LINK_LIBS ${LINK_LIBS} ${HDF5_LIBRARIES} ${HDF5_LIBRARIES})

add_executable (advection_OPENACC advection.cpp ../utils/advection_utils.cpp ../utils/advection_output.cpp)
target_link_libraries(advection_OPENACC ${LINK_LIBS})
