#!/bin/bash
#SBATCH -p nvltv # partition (query with sinfo)
#SBATCH -o job-%j.out # stdout and stderr is written to this file
#SBATCH -N 1 # 1 node
#SBATCH -t 00:05:00 # maximal run time of the job
module load cuda
module load gnu8

export PATH_KP=kokkos-tools/profiling/simple-kernel-timer
export KOKKOS_PROFILE_LIBRARY=$PATH_KP/kp_kernel_timer.so
export PATH=$PATH:$PATH_KP

build/sparsemv
