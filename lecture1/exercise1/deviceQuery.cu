#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        cout << "-------------------------------------------------------------" << endl;
        cout << "Device " << dev << ": \"" << deviceProp.name << "\"" << endl;
        cout << "\tCUDA Driver and Runtime version: " << deviceProp.major << "." << deviceProp.minor << endl;
        cout << "\tTotal amount of global memory: " << deviceProp.totalGlobalMem / 1e9 << " GB" << endl;
        cout << "\tMaximum number of threads per block: " << deviceProp.maxThreadsPerBlock << endl;
        cout << "\tUVA support: " << (deviceProp.unifiedAddressing ? "Yes" : "No") << endl;
        cout << "-------------------------------------------------------------" << endl;
        
    }
    return EXIT_SUCCESS;
}