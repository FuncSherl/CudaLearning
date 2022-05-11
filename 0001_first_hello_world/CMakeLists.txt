CMAKE_MINIMUM_REQUIRED(VERSION 3.8)
#// 项目名称， 新版cmake不推荐使用FindCuda这个宏，取代的是在project的LANGUAGES加载cuda或者enable_language(CUDA),后续就同cpp一样使用cu文件就行了
project(helloworld LANGUAGES CUDA)

# 指定cuda的compute architectures，主要是用于编译电脑的显卡比需要部署的显卡更好的时候，如果采用默认的计算架构的话，在上代GPU上会无法运行
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

#// C++和CUDA的编译参数
# The easiest way to enable support for a specific C++ standard in CMake is by using the CMAKE_CXX_STANDARD variable
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# set(CMAKE_CXX_STANDARD 11)
# SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};) # -gencode arch=compute_61,code=sm_61;

#两种包含路径的不同：
#include_directories：
#当前CMakeList.txt中的所有目标以及所有在其调用点之后添加的子目录中的所有目标将具有此头文件搜索路径，如果有不同目录相同名称的头文件会产生影响，
#所以这里建议针对特定的target进行添加头文件的操作，不要使用include_directories
#target_include_directories：
#指定目标包含的头文件路径

#// 头文件路径
INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/include
)
#// 库文件路径，按需
# LINK_DIRECTORIES(/usr/lib  /usr/local/lib)

# add the executable
add_executable(hello hello.cu)
add_executable(clock clock.cu)