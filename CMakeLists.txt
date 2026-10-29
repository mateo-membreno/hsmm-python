cmake_minimum_required(VERSION 3.18)
project(hsmm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC src/bindings.cpp)
target_include_directories(_core PRIVATE src)
# No -ffast-math: the DP relies on exact -inf / 0 handling.
target_compile_options(_core PRIVATE -O3 -Wall -Wextra)

install(TARGETS _core LIBRARY DESTINATION hsmm)
