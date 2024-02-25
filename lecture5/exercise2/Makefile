NVCC = nvcc
NVCCFLAGS = -O3
LDFLAGS = -lcufft

all: cufft

cufft: cufft.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f cufft
