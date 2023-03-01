cmake_minimum_required(VERSION 3.0)
project(tfcpp)

set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
set(CMAKE_CXX_STANDART 17)

################################################################################
# OPENCV
find_package(OpenCV REQUIRED)

################################################################################
# BOOST
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

################################################################################
# TENSORFLOW
# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
set(TensorFlow_INCLUDE_DIR /usr/include/tensorflow)
set(TensorFlow_LIBRARY
    /usr/lib/libtensorflow.so
    /usr/lib/libtensorflow_cc.so
    /usr/lib/libtensorflow_framework.so)

################################################################################
# PROJECT
set(PROJECT_INCLUDE_DIRS
    ${TensorFlow_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS})
set(PROJECT_LIBRARIES
    ${TensorFlow_LIBRARY}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS})

include_directories(${PROJECT_INCLUDE_DIRS})

add_executable(test test/test.cpp)
target_link_libraries(test ${PROJECT_LIBRARIES})

add_executable(main src/main.cpp)
target_link_libraries(main ${PROJECT_LIBRARIES})
