cmake_minimum_required(VERSION 3.20)

include(pybind11)
pybind11_add_module(pyfalcon
  ${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt)
