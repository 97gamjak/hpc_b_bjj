# Lecture 10 Exercise 1

## Usage

First of all to build the different executables perform following camke build steps:

```bash
#for compilation on gpu3
module purge
module load cuda
module load cmake
module load gnu8

mkdir build
cd build
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON ..
make
cd ..
```

Then perform following steps for building the kokkos-tools in the root folder of this exercise

```bash

#for compilation on gpu3
module purge
module load gnu7

git clone https://github.com/kokkos/kokkos-tools
cd kokkos-tools/profiling/simple-kernel-timer/
make
cd ../../..
```

## Output
```bash
 (Type)   Total Time, Call Count, Avg. Time per Call, %Total Time in Kernels, %Total Program Time
-------------------------------------------------------------------------

Regions: 

- KokkosBlas::dot[ETI]
 (REGION)   0.005058 1 0.005058 25.599720 7.521263

-------------------------------------------------------------------------
Kernels: 

- dot_product
 (ParRed)   0.005043 1 0.005043 25.523699 7.498928
- KokkosBlas::dot<1D>
 (ParRed)   0.005039 1 0.005039 25.503186 7.492901
- fill_vec
 (ParFor)   0.004840 1 0.004840 24.496814 7.197226
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.002424 1 0.002424 12.268317 3.604463
- Kokkos::View::initialization [y] via memset
 (ParFor)   0.002412 1 0.002412 12.207983 3.586736

-------------------------------------------------------------------------
Summary:

Total Execution Time (incl. Kokkos + non-Kokkos):                   0.06725 seconds
Total Time in Kokkos kernels:                                       0.01976 seconds
   -> Time outside Kokkos kernels:                                  0.04749 seconds
   -> Percentage in Kokkos kernels:                                   29.38 %
Total Calls to Kokkos Kernels:                                            5

```