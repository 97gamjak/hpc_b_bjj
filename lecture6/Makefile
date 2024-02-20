CC=gcc
CC_FLAGS=-O3 -Wall -Wextra -pedantic
NVCC=nvcc
NVCC_FLAGS=-O3

exercise-multiplegpu: exercise-multiplegpu.cu ../cuda_error_check.h
	@$(NVCC) $(NVCC_FLAGS) exercise-multiplegpu.cu -o exercise-multiplegpu

.PHONEY: clean
clean:
	@rm exercise-multiplegpu
	
