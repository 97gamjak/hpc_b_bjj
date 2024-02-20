CXX = nvcc
CXXFLAGS = -O3
LDFLAGS = -lcublas

all: matvec

matvec: matvec.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f matvec