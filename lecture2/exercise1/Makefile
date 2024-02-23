NVCC = nvcc
NVCC_FLAGS = -std=c++11 

all: syncthreads

syncthreads: syncthreads.cu
	$(NVCC) $(NVCC_FLAGS) syncthreads.cu -o syncthreads

clean:
	rm -f syncthreads