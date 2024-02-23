cmake_minimum_required(VERSION 3.11)
include(FetchContent)

project(kokkos_vecadd CXX)

FetchContent_Declare(kokkos
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
)

FetchContent_MakeAvailable(kokkos)
add_executable(vecadd vecadd.cpp)
add_executable(vecadd_3d_layoutleft vecadd_3d_layoutleft.cpp)
add_executable(vecadd_3d_layoutright vecadd_3d_layoutright.cpp)
add_dependencies(vecadd kokkos)
add_dependencies(vecadd_3d_layoutleft kokkos)
add_dependencies(vecadd_3d_layoutright kokkos)
target_include_directories(vecadd PRIVATE ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(vecadd PRIVATE kokkos)
target_include_directories(vecadd_3d_layoutleft PRIVATE ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(vecadd_3d_layoutleft PRIVATE kokkos)
target_include_directories(vecadd_3d_layoutright PRIVATE ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(vecadd_3d_layoutright PRIVATE kokkos)
