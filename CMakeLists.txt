cmake_minimum_required(VERSION 3.20)
project(Falcon)

set(SOURCE
  ${PROJECT_SOURCE_DIR}/falcon/autograd/Tensor.cc
  ${PROJECT_SOURCE_DIR}/test.cc
)
add_executable(falcon ${SOURCE})

find_package(ArrayFire)
target_link_libraries(falcon ArrayFire::afcpu)

target_include_directories(falcon
  PRIVATE
    ${PROJECT_SOURCE_DIR}/falcon
)
