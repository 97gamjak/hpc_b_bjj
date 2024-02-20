NVCC = nvcc

all: matmul

matmul: matmul.cu
	$(NVCC) -o matmul matmul.cu

clean:
	rm -f matmul
