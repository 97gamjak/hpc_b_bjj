CC=gcc
CC_FLAGS=-O3 -Wall -Wextra -pedantic
NVCC=nvcc
NVCC_FLAGS=-O3 -arch=sm_35 -rdc=true -lcudadevrt -lgomp -lpng

all: mandelbrot mandelbrot_dyn

mandelbrot: mandelbrot.cu
	@$(NVCC) $(NVCC_FLAGS) mandelbrot.cu -o mandelbrot

mandelbrot_dyn: mandelbrot_dyn.cu
	@$(NVCC) $(NVCC_FLAGS) mandelbrot_dyn.cu -o mandelbrot_dyn

.PHONEY: clean
clean:
	@rm mandelbrot
	@rm mandelbrot_dyn
