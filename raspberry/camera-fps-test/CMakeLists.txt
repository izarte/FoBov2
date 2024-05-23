cmake_minimum_required(VERSION 3.4)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -Wno-psabi")

include_directories(/usr/include/c)

project(Test_C)
add_executable( fps fps.c )
target_link_libraries( fps ArducamDepthCamera2c)
