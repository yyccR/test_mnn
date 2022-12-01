cmake_minimum_required(VERSION 3.17)
project(test_mnn)

set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mnn-2.2.0/include)
include_directories(${MKLML_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/yolox)

add_library(mnn SHARED IMPORTED)
set_target_properties(mnn PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/mnn-2.2.0/libMNN.dylib)
add_library(mnn_express SHARED IMPORTED)
set_target_properties(mnn_express PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/mnn-2.2.0/libMNN_Express.dylib)

add_executable(test_mnn ./yolox/main.cpp ./yolox/YOLOXMNN.cpp)
target_link_libraries(test_mnn mnn)
target_link_libraries(test_mnn mnn_express)
target_link_libraries(test_mnn ${OpenCV_LIBS})
