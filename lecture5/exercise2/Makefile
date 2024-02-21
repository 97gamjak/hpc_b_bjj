NVCC = nvcc
NVCCFLAGS = -O3
LDFLAGS = -lcufft

all: cufft

cufft: cufft.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f cufft