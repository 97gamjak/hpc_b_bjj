#!/bin/bash
#SBATCH -p nvltv # partition (query with sinfo)
#SBATCH -o job-%j.out # stdout and stderr is written to this file
#SBATCH -N 1 # 1 node
#SBATCH -t 00:05:00 # maximal run time of the job
module load cuda

export PATH_KP=kokkos-tools/profiling/simple-kernel-timer
export KOKKOS_PROFILE_LIBRARY=$PATH_KP/libkp_kernel_timer.so
export PATH=$PATH:$PATH_KP

build_gpu/vecadd_3d_layoutleft

build_gpu/vecadd_3d_layoutright

build_cpu/vecadd_3d_layoutleft

build_cpu/vecadd_3d_layoutright
