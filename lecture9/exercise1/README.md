# Lecture 10 - Exercise 1
## Usage

First of all to build the different executables perform following camke build steps:

```bash
mkdir build_gpu
cd build_gpu
cmake -DKokkos_ENABLE_CUDA ..
make

mkdir build_cpu
cd build_cpu
cmake ..
make
```

Then perform following steps for building the kokkos-tools in the root folder of this exercise

```bash
git clone https://github.com/kokkos/kokkos-tools
cd kokkos-tools/
mkdir build
cd build
cmake ../
make
```
