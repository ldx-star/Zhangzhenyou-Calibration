cmake_minimum_required(VERSION 3.16)
project(Zhangzhenyou_Calibration)

set(CMAKE_CXX_STANDARD 17)

add_library(lib)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Zhangzhenyou_Calibration main.cpp)

target_link_libraries(${PROJECT_NAME} lib ${OpenCV_LIBS})
