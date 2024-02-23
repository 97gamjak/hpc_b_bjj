NVCC = nvcc
NVCC_FLAGS = -std=c++11 

all: syncthreads syncthreads_solution

syncthreads: syncthreads.cu
	$(NVCC) $(NVCC_FLAGS) syncthreads.cu -o syncthreads

syncthreads_solution: syncthreads_solution.cu
	$(NVCC) $(NVCC_FLAGS) syncthreads_solution.cu -o syncthreads_solution

clean:
	rm -f syncthreads syncthreads_solution