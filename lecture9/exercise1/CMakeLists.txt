cmake_minimum_required(VERSION 3.11)
include(FetchContent)

project(kokkos_vecadd CXX)

FetchContent_Declare(kokkos
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
)

FetchContent_MakeAvailable(kokkos)
add_executable(vecadd vecadd.cpp)
add_dependencies(vecadd kokkos)
target_include_directories(vecadd PRIVATE ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(vecadd PRIVATE kokkos)
