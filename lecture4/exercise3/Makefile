NVCC = nvcc
NVCC_FLAGS = -g -G

all: stencil

stencil: stencil.cu
	$(NVCC) $(NVCC_FLAGS) -o stencil stencil.cu

clean:
	rm -f stencil