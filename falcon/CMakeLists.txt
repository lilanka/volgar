cmake_minimum_required(VERSION 3.20)

include(${CMAKE_CURRENT_LIST_DIR}/autograd/CMakeLists.txt)

target_include_directories(falcon
  ${PROJECT_SOURCE_DIR}/autograd)
