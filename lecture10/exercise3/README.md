# Lecture 9 - Exercise 2
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
## Performance of spmv

```bash
 (Type)   Total Time, Call Count, Avg. Time per Call, %Total Time in Kernels, %Total Program Time
-------------------------------------------------------------------------

Regions: 

- KokkosSparse::spmv[TPL_CUSPARSE,double]
 (REGION)   0.370761 1 0.370761 237417.404580 99.821166

-------------------------------------------------------------------------
Kernels: 

- fill_crs
 (ParFor)   0.000056 1 0.000056 35.877863 0.015085
- Kokkos::ViewFill-1D
 (ParFor)   0.000032 1 0.000032 20.458015 0.008601
- Kokkos::View::initialization [row_ptr] via memset
 (ParFor)   0.000023 1 0.000023 14.809160 0.006226
- Kokkos::View::initialization [y] via memset
 (ParFor)   0.000011 1 0.000011 7.022901 0.002953
- Kokkos::View::initialization [col_ind] via memset
 (ParFor)   0.000010 1 0.000010 6.412214 0.002696
- Kokkos::View::initialization [values] via memset
 (ParFor)   0.000010 1 0.000010 6.412214 0.002696
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.000009 1 0.000009 5.801527 0.002439
- Kokkos::View::initialization [y_mirror] via memset
 (ParFor)   0.000005 1 0.000005 3.206107 0.001348

-------------------------------------------------------------------------
Summary:

Total Execution Time (incl. Kokkos + non-Kokkos):                   0.37143 seconds
Total Time in Kokkos kernels:                                       0.00016 seconds
   -> Time outside Kokkos kernels:                                  0.37127 seconds
   -> Percentage in Kokkos kernels:                                    0.04 %
Total Calls to Kokkos Kernels:                                            8

-------------------------------------------------------------------------
```

## Performance of cusparse
```bash
0.0112292 s
```