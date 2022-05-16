CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
project(test_nms)
enable_language(CUDA)
# 指定cuda的compute architectures，主要是用于编译电脑的显卡比需要部署的显卡更好的时候，如果采用默认的计算架构的话，在上代GPU上会无法运行
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

# opencv
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#// C++和CUDA的编译参数
# The easiest way to enable support for a specific C++ standard in CMake is by using the CMAKE_CXX_STANDARD variable
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# add the executable
add_executable(nms_cpu nms_cpu.cpp)
add_executable(nms_gpu nms_gpu.cu)

target_link_libraries(nms_cpu ${OpenCV_LIBRARIES})
target_link_libraries(nms_gpu ${OpenCV_LIBRARIES})