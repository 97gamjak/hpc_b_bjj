# Lecture 9 - Exercise 1
## Usage

First of all to build the different executables perform following camke build steps:

```bash
#for compilation on gpu3
module purge
module load cuda
module load cmake
module load gnu7

mkdir build_gpu
cd build_gpu
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON ..
make
cd ..

mkdir build_cpu
cd build_cpu
cmake ..
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

## Performance
### Overview Total Execution Time

| Layout | Time |
| :--- | :---: |
| GPU layout left:  |  0.50849 seconds |
| GPU layout right: |  0.64102 seconds |
| CPU layout left:  | 52.59279 seconds |
| CPU layout right: | 13.28428 seconds |

### Performance in GB/s of the <*vecadd*> kernel

total of n=2⁹\*2⁹\*2⁹ double entries -> 1.073741824 GB

| Layout | Performance |
| :--- | :---: |
|GPU layout left: |  280.937 GB/s|
|GPU layout right:|   19.622 GB/s|
|CPU layout left: |    0.034 GB/s|
|CPU layout right:|    0.149 GB/s|

### GPU Layout Left
```bash
(Type)   Total Time, Call Count, Avg. Time per Call, %Total Time in Kernels, %Total Program Time

Kernels: 

- Kokkos::View::initialization [z_mirror] via memset
 (ParFor)   0.261811 1 0.261811 95.975231 51.487857
- vecadd
 (ParFor)   0.003822 1 0.003822 1.401109 0.751653
- fill_vec
 (ParFor)   0.003510 1 0.003510 1.286703 0.690278
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.001222 1 0.001222 0.447925 0.240298
- Kokkos::View::initialization [y] via memset
 (ParFor)   0.001213 1 0.001213 0.444691 0.238564
- Kokkos::View::initialization [z] via memset
 (ParFor)   0.001212 1 0.001212 0.444342 0.238376

Summary:

Total Execution Time (incl. Kokkos + non-Kokkos):                   0.50849 seconds
Total Time in Kokkos kernels:                                       0.27279 seconds
   -> Time outside Kokkos kernels:                                  0.23570 seconds
   -> Percentage in Kokkos kernels:                                   53.65 %
Total Calls to Kokkos Kernels:                                            6
```
### GPU Layout Right
```bash
(Type)   Total Time, Call Count, Avg. Time per Call, %Total Time in Kernels, %Total Program Time

Kernels: 

- Kokkos::View::initialization [z_mirror] via memset
 (ParFor)   0.261437 1 0.261437 64.580354 40.784639
- fill_vec
 (ParFor)   0.085018 1 0.085018 21.001192 13.262950
- vecadd
 (ParFor)   0.054720 1 0.054720 13.517017 8.536445
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.001221 1 0.001221 0.301657 0.190506
- Kokkos::View::initialization [z] via memset
 (ParFor)   0.001215 1 0.001215 0.300126 0.189539
- Kokkos::View::initialization [y] via memset
 (ParFor)   0.001213 1 0.001213 0.299654 0.189242

Summary:

Total Execution Time (incl. Kokkos + non-Kokkos):                   0.64102 seconds
Total Time in Kokkos kernels:                                       0.40482 seconds
   -> Time outside Kokkos kernels:                                  0.23619 seconds
   -> Percentage in Kokkos kernels:                                   63.15 %
Total Calls to Kokkos Kernels:                                            6
```

### CPU Layout Left
```bash
 (Type)   Total Time, Call Count, Avg. Time per Call, %Total Time in Kernels, %Total Program Time

Kernels: 

- vecadd
 (ParFor)   31.345315 1 31.345315 59.608129 59.600021
- fill_vec
 (ParFor)   20.442267 1 20.442267 38.874240 38.868952
- Kokkos::View::initialization [z] via memset
 (ParFor)   0.266110 1 0.266110 0.506051 0.505982
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.266008 1 0.266008 0.505857 0.505788
- Kokkos::View::initialization [y] via memset
 (ParFor)   0.265938 1 0.265938 0.505724 0.505655

Summary:

Total Execution Time (incl. Kokkos + non-Kokkos):                  52.59279 seconds
Total Time in Kokkos kernels:                                      52.58564 seconds
   -> Time outside Kokkos kernels:                                  0.00715 seconds
   -> Percentage in Kokkos kernels:                                   99.99 %
Total Calls to Kokkos Kernels:                                            5
```

### CPU Layout Right

```bash
 (Type)   Total Time, Call Count, Avg. Time per Call, %Total Time in Kernels, %Total Program Time


Kernels: 

- vecadd
 (ParFor)   7.193347 1 7.193347 54.177811 54.149319
- fill_vec
 (ParFor)   5.293292 1 5.293292 39.867249 39.846283
- Kokkos::View::initialization [x] via memset
 (ParFor)   0.264874 1 0.264874 1.994940 1.993890
- Kokkos::View::initialization [z] via memset
 (ParFor)   0.263151 1 0.263151 1.981962 1.980920
- Kokkos::View::initialization [y] via memset
 (ParFor)   0.262630 1 0.262630 1.978039 1.976998

Summary:

Total Execution Time (incl. Kokkos + non-Kokkos):                  13.28428 seconds
Total Time in Kokkos kernels:                                      13.27729 seconds
   -> Time outside Kokkos kernels:                                  0.00699 seconds
   -> Percentage in Kokkos kernels:                                   99.95 %
Total Calls to Kokkos Kernels:                                            5
```