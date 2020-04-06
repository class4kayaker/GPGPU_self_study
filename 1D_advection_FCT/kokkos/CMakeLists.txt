cmake_minimum_required (VERSION 3.10)

set(SPACK_CXX $ENV{SPACK_CXX})
if(SPACK_CXX)
  message("found spack compiler ${SPACK_CXX}")
  set(CMAKE_CXX_COMPILER ${SPACK_CXX} CACHE STRING "the C++ compiler" FORCE)  
  set(ENV{CXX} ${SPACK_CXX})
endif()

cmake_policy(SET CMP0074 NEW)
find_package(Kokkos REQUIRED)

project (KokkosAdvectionFCT C CXX)

set (LIB_TYPE SHARED) # or SHARED
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)

find_package(HDF5 COMPONENTS C HL CXX REQUIRED)
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIRS})

set (LINK_LIBS ${LINK_LIBS} ${HDF5_LIBRARIES} Kokkos::kokkos)

add_executable(advection_K advection.cpp ../utils/advection_utils.cpp ../utils/advection_output.cpp)
target_link_libraries(advection_K ${LINK_LIBS})
