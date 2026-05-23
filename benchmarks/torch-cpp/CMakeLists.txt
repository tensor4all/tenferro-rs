cmake_minimum_required(VERSION 3.18)
project(tenferro_torch_cpu_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_executable(torch_cpu_bench torch_cpu_bench.cpp)
target_link_libraries(torch_cpu_bench "${TORCH_LIBRARIES}")

if(NOT MSVC)
  target_compile_options(torch_cpu_bench PRIVATE -O3)
endif()
