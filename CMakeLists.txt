cmake_minimum_required(VERSION 3.20)
project(Falcon)

set(SOURCE
  ${PROJECT_SOURCE_DIR}/falcon/autograd/Tensor.cc
  ${PROJECT_SOURCE_DIR}/falcon/autograd/operators/Add.cc
  ${PROJECT_SOURCE_DIR}/falcon/autograd/operators/Sub.cc
  ${PROJECT_SOURCE_DIR}/falcon/autograd/operators/Mul.cc
  ${PROJECT_SOURCE_DIR}/falcon/autograd/operators/Div.cc
  ${PROJECT_SOURCE_DIR}/falcon/autograd/operators/Matmul.cc
  ${PROJECT_SOURCE_DIR}/test.cc
)
add_executable(falcon ${SOURCE})

find_package(ArrayFire)
target_link_libraries(falcon ArrayFire::afcpu)

target_include_directories(falcon
  PUBLIC
    ${PROJECT_SOURCE_DIR}/falcon
    ${PROJECT_SOURCE_DIR}/falcon/autograd
    ${PROJECT_SOURCE_DIR}/falcon/autograd/operators
)
