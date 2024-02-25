cmake_minimum_required(VERSION 3.11)
include(FetchContent)

project(kokkos_sparsemv CXX)

FetchContent_Declare(kokkos
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
)
FetchContent_MakeAvailable(kokkos)

FetchContent_Declare(kokkos-kernels
    GIT_REPOSITORY "https://github.com/kokkos/kokkos-kernels.git"
)
FetchContent_MakeAvailable(kokkos-kernels)

add_executable(sparsemv sparsemv.cpp)
add_dependencies(sparsemv kokkos)
target_include_directories(sparsemv PRIVATE ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(sparsemv PRIVATE kokkos)
target_link_libraries(sparsemv PRIVATE Kokkos::kokkoskernels)