NVCC = nvcc
NVCCFLAGS = -O3
LDFLAGS = -lcusparse

all: cusparse

cusparse: cusparse.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f cusparse