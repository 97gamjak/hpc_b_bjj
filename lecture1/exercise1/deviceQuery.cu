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
        printf("\nDevice %d: \"%s\"\n\t> Max gridsize: %d\n\t> Max threadsPerBlock: %d\n", dev, deviceProp.name, deviceProp.maxGridSize, deviceProp.maxThreadsPerBlock);
    }
    return 0;
}