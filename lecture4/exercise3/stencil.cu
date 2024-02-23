#include <stdio.h>

#define RADIUS 2
#define N 1024

__global__ void applyStencil1D(int sIdx, int eIdx, const float *weights, float *in, float *out) {
    int i = sIdx + blockIdx.x*blockDim.x + threadIdx.x;
    if (i < eIdx) {
        out[i] = 0;
        //loop over all elements in the stencil
        for(int j = -RADIUS; j <= RADIUS; j++) {
            out[i] += weights[j + RADIUS] * in[i + j];
        }
        out[i] = out[i] / (2 * RADIUS + 1);
    }
}

void initializeWeights(float *weights, int radius) {
    for(int i = 0; i < radius*2+1; ++i) {
        weights[i] = 1.0;
    }
}

void initializeArray(float *array, int n) {
    for(int i = 0; i < N; ++i) {
        array[i] = 0.0;
    }

    array[N/2] = 35.0;
}

int main() {
    int size = N * sizeof(float);
    int wsize = (2 * RADIUS + 1) * sizeof(float);

    //allocate resources
    float *weights = (float *)malloc(wsize);
    float *in = (float *)malloc(size);
    float *out= (float *)malloc(size);

    initializeWeights(weights, RADIUS);
    initializeArray(in, N);

    float *d_weights; cudaMalloc(&d_weights, wsize);
    float *d_in; cudaMalloc(&d_in, size); // <- wsize would be wrong here
    float *d_out; cudaMalloc(&d_out, size); // <- wsize would be wrong here

    cudaMemcpy(d_weights, weights, wsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice); // <- wsize would be wrong here

    applyStencil1D<<<N/512, 512>>>(RADIUS, N-RADIUS, d_weights, d_in, d_out);

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost); // <- wsize would be wrong here

    //free resources
    free(weights); free(in); free(out);
    cudaFree(d_weights); cudaFree(d_in); cudaFree(d_out);

    return 0;
}
