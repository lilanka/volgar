cmake_minimum_required(VERSION 3.20)

set(
  AUTOGRAD_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/Tensor.cc
)

target_sources(
  falcon
  PRIVATE
  ${AUTOGRAD_SOURCES}
)

target_include_directories(falcon
  ${CMAKE_CURRENT_LIST_DIR}/operators)
