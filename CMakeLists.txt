cmake_minimum_required(VERSION 3.24)
project(Falcon)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE
  ${PROJECT_SOURCE_DIR}/falcon/tensor/tensor.cc
  ${PROJECT_SOURCE_DIR}/falcon/functional/functional.cc
  ${PROJECT_SOURCE_DIR}/test/test_autodiff.cc
  #${PROJECT_SOURCE_DIR}/test/test_tensor.cc
)

add_executable(falcon ${SOURCE})

find_package(ArrayFire)
target_link_libraries(falcon ArrayFire::afcpu)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)