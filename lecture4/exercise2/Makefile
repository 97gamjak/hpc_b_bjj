NVCC = nvcc
NVCC_FLAGS = -g -G 

all: syncthreads

syncthreads: syncthreads.cu
	$(NVCC) $(NVCC_FLAGS) -o syncthreads syncthreads.cu

clean:
	rm -f syncthreads