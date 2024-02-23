NVCC = nvcc 

all: exercise-efficiency exercise-efficiency_original

exercise-efficiency: exercise-efficiency.cu
	$(NVCC) -o $@ $<

exercise-efficiency_original: exercise-efficiency_original.cu
	$(NVCC) -o $@ $<

clean:
	rm -f exercise-efficiency exercise-efficiency_original
