cmake_minimum_required(VERSION 3.0)
project(mainproject)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
find_package(spdlog REQUIRED)
find_package(OpenVINO REQUIRED)
set(OpenCV_HOME opencv-build)
set(EIGEN_HOME include/eigen-build)
include_directories("include" "${OpenCV_HOME}/include/opencv4" "${EIGEN_HOME}/include/eigen3")
link_directories(opencv-build/lib)
file(GLOB_RECURSE SRC "src/*.cpp")
file(GLOB_RECURSE TESTS "src/tests/*.cpp")
file(GLOB_RECURSE UTILS "src/utils/*.cpp")
add_library(obdet SHARED ${SRC})
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)
target_link_libraries(obdet ${OpenCV_LIBS} openvino spdlog)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



