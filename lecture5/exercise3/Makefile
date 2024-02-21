NVCC = nvcc
NVCCFLAGS = -O3
#NVCCFLAGS = -lineinfo #for debugging
LDFLAGS = -lcusparse

all: cusparse

cusparse: cusparse.cu ../../cuda_error_check.h
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f cusparse
