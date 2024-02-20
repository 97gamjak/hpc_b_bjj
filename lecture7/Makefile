NVCC=nvcc
NVCC_FLAGS=-O3 -arch=sm_35 -rdc=true -lcudadevrt -lgomp -lpng

all: mandelbrot mandelbrot_dyn

mandelbrot: mandelbrot.cu
	@$(NVCC) $(NVCC_FLAGS) mandelbrot.cu -o mandelbrot

mandelbrot_dyn: mandelbrot-dyn.cu
	@$(NVCC) $(NVCC_FLAGS) mandelbrot-dyn.cu -o mandelbrot-dyn

clean:
	@rm mandelbrot
	@rm mandelbrot_dyn
