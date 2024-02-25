cmake_minimum_required(VERSION 3.11)
include(FetchContent)

project(kokkos_matvecmul CXX)
FetchContent_Declare(kokkos
    GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
)

FetchContent_MakeAvailable(kokkos)

FetchContent_Declare(kokkos-kernels
    GIT_REPOSITORY "https://github.com/kokkos/kokkos-kernels.git"
)

FetchContent_MakeAvailable(kokkos-kernels)
add_executable(matvecmul matvecmul.cpp)
add_dependencies(matvecmul kokkos)
target_include_directories(matvecmul PRIVATE ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(matvecmul PRIVATE kokkos)
target_link_libraries(matvecmul PRIVATE Kokkos::kokkoskernels)
