cmake_minimum_required(VERSION 3.5)
project(test)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#set (PYTHONPATH=/usr/lib/python3/dist-packages/)


find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


find_package(pybind11 REQUIRED) 

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

add_executable(test_gpc test_gpc.cpp)
target_link_libraries(test_gpc PRIVATE pybind11::embed)

add_executable(test_mean test_mean.cpp)
target_link_libraries(test_mean PRIVATE pybind11::embed)










