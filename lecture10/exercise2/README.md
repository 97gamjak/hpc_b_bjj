# Lecture 10 - Exercise 2
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

## Ouput
 (Type)   Total Time, Call Count, Avg. Time per Call, %Total Time in Kernels, %Total Program Time
-------------------------------------------------------------------------

Regions: 

- KokkosBlas::gemv[TPL_CUBLAS,double]
 (REGION)   0.419821 1 0.419821 1257.288007 77.998548

-------------------------------------------------------------------------
Kernels: 

- InitA
 (ParFor)   0.012683 1 0.012683 37.983749 2.356403
- MatVecProduct
 (ParFor)   0.010646 1 0.010646 31.883158 1.977940
- Kokkos::View::initialization [A] via memset
 (ParFor)   0.009600 1 0.009600 28.750036 1.783570
- Kokkos::View::initialization [y_1_mirror] via memset
 (ParFor)   0.000208 1 0.000208 0.622626 0.038626
- Kokkos::View::initialization [y_2_mirror] via memset
 (ParFor)   0.000198 1 0.000198 0.592637 0.036766
- InitX
 (ParFor)   0.000019 1 0.000019 0.057122 0.003544
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.000014 1 0.000014 0.042127 0.002613
- Kokkos::View::initialization [y_1] via memset
 (ParFor)   0.000012 1 0.000012 0.035701 0.002215
- Kokkos::View::initialization [y_2] via memset
 (ParFor)   0.000011 1 0.000011 0.032845 0.002038

-------------------------------------------------------------------------
Summary:

Total Execution Time (incl. Kokkos + non-Kokkos):                   0.53824 seconds
Total Time in Kokkos kernels:                                       0.03339 seconds
   -> Time outside Kokkos kernels:                                  0.50485 seconds
   -> Percentage in Kokkos kernels:                                    6.20 %
Total Calls to Kokkos Kernels:                                            9

-------------------------------------------------------------------------