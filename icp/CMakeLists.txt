cmake_minimum_required(VERSION 3.10)
project(pointcloud_example)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3 찾기
find_package(Eigen3 REQUIRED)

add_executable(pointcloud_example
    src/icp.cpp
)

target_include_directories(pointcloud_example PRIVATE
    ${EIGEN3_INCLUDE_DIR}
)
