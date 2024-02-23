#include <stdio.h>
#include <stdlib.h>

int main()
{
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice %ld: \"%s\"\n\t> Max gridsize: %ld\n\t> Max threadsPerBlock: %ld\n", dev, deviceProp.name, deviceProp.maxGridSize, deviceProp.maxThreadsPerBlock);
    }
    return 0;
}