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
 (REGION)   0.010096 1 0.010096 24.445099 6.570528

-------------------------------------------------------------------------
Kernels: 

- fill_vec
 (ParFor)   0.010949 1 0.010949 26.510573 7.125701
- KokkosBlas::dot<1D>
 (ParRed)   0.010092 1 0.010092 24.435285 6.567890
- dot_product
 (ParRed)   0.010089 1 0.010089 24.427781 6.565873
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.005090 1 0.005090 12.324149 3.312573
- Kokkos::View::initialization [y] via memset
 (ParFor)   0.005081 1 0.005081 12.302213 3.306676

-------------------------------------------------------------------------
Summary:

Total Execution Time (incl. Kokkos + non-Kokkos):                   0.15366 seconds
Total Time in Kokkos kernels:                                       0.04130 seconds
   -> Time outside Kokkos kernels:                                  0.11236 seconds
   -> Percentage in Kokkos kernels:                                   26.88 %
Total Calls to Kokkos Kernels:                                            5

-------------------------------------------------------------------------
```