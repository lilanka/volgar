cmake_minimum_required(VERSION 3.22)
project(Falcon)

set(SOURCE
  ${PROJECT_SOURCE_DIR}/falcon/tensor/Tensor.cc
  ${PROJECT_SOURCE_DIR}/falcon/autograd/Functional.cc
  ${PROJECT_SOURCE_DIR}/falcon/nn/Linear.cc
  ${PROJECT_SOURCE_DIR}/test/linear.cc
  #${PROJECT_SOURCE_DIR}/test/operations.cc
)
add_executable(falcon ${SOURCE})

find_package(ArrayFire)
target_link_libraries(falcon ArrayFire::afcpu)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)
