CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
project(test_sort)
enable_language(CUDA)
# 指定cuda的compute architectures，主要是用于编译电脑的显卡比需要部署的显卡更好的时候，如果采用默认的计算架构的话，在上代GPU上会无法运行
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

#// C++和CUDA的编译参数
# The easiest way to enable support for a specific C++ standard in CMake is by using the CMAKE_CXX_STANDARD variable
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# add the executable
add_executable(merge_sort merge_sort.cu)
add_executable(sum_merge sum_merge.cu)