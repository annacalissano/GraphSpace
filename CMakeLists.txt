cmake_minimum_required(VERSION 3.5)
project(GraphSpace)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(pybind11)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

pybind11_add_module(matcherc src/matcher.cpp)
pybind11_add_module(meanc src/mean_aacc.cpp)
pybind11_add_module(gpcc src/gpc_aacc.cpp)




