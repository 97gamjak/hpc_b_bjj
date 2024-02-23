NVCC = nvcc
NVCC_FLAGS = -g -g

all: vector

vector: vector.cu
	$(NVCC) $(NVCC_FLAGS) -o vector vector.cu

clean:
	rm -f vector
