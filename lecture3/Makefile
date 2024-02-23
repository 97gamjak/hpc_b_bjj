NVCC = nvcc 

all: exercise-efficiency

exercise-efficiency: exercise-efficiency.cu
	$(NVCC) -o $@ $<

clean:
	rm -f exercise-efficiency result.data