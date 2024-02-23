CC=gcc
CC_FLAGS=-O3 -Wall -Wextra -pedantic -std=c99
NVCC=nvcc
NVCC_FLAGS=-O3

all: heat_stencil_omp heat_stencil_cuda

heat_stencil_omp: heat_stencil_omp.c
	@$(CC) $(CC_FLAGS) heat_stencil_omp.c -o heat_stencil_omp -fopenmp

heat_stencil_cuda: heat_stencil_cuda.cu
	@$(NVCC) $(NVCC_FLAGS) heat_stencil_cuda.cu -o heat_stencil_cuda

.PHONEY: clean
clean:
	@rm heat_stencil_omp heat_stencil_cuda
	
