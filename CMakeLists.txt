cmake_minimum_required(VERSION 2.6)

project(arcface)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

find_package(PythonLibs 3.6 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(arcface-r100 ${PROJECT_SOURCE_DIR}/arcface-r100.cpp)
target_link_libraries(arcface-r100 nvinfer)
target_link_libraries(arcface-r100 cudart)
target_link_libraries(arcface-r100 ${OpenCV_LIBS})


SET(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/ndarray_converter.cpp
)

add_subdirectory(${PROJECT_SOURCE_DIR}/pybind11)
include_directories(${pybind11_INCLUDE_DIR})

cuda_add_library(arcface_module SHARED main.cpp prelu.cu ndarray_converter.cpp ndarray_converter.hpp prelu.h)
target_link_libraries(arcface_module nvinfer)
target_link_libraries(arcface_module cudart)
target_link_libraries(arcface_module ${OpenCV_LIBS})
target_link_libraries(arcface_module ${PYTHON_LIBRARIES})
set_target_properties(arcface_module PROPERTIES PREFIX "")

add_definitions(-O2 -pthread)
