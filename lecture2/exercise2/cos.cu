#include <stdio.h>

#include "../../cuda_error_check.h"

__global__ void mathKernel(float *a) {
    a[1] = __cosf(a[0]);  // a fast implementation of the cosine function
    a[2] = cos(a[0]);
    return;
}

int main(int argc, char **argv) {
    float *data;

    if (argc != 2) {
        printf("usage: %s <value>\n", argv[0]);
        return 1;
    }

    gpuErrorCheck(cudaMallocManaged(&data, 3 * sizeof(float)), true);

    data[0] = atof(argv[1]);

    mathKernel<<<1, 1>>>(data);

    cudaDeviceSynchronize();

    printf("GPU: __cosf(%.15g) = %.15e\n", data[0], data[1]);
    printf("GPU: cos(%.15g) = %.15e\n", data[0], data[2]);
    printf("CPU: cosf(%.15g) = %.15e\n", data[0], cosf(data[0]));
    printf("CPU: cos(%.15g) = %.15e\n", data[0], cos(data[0]));

    cudaFree(data);

    return 0;
}
