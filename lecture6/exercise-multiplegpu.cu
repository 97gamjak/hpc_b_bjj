#include <cmath>
#include <iostream>

#include "cuda_error_check.h"

// index column major, so that arr[x,y] and arr[x, y+1] are consecutive in memory
#define index(i_x, i_y, n_y) ((i_y) + (i_x) * (n_y))

using namespace std;

// split domain along x direction

__global__ void k_upwind(long nx, long ny, double *in, double *out) {
    long ix = threadIdx.x + blockDim.x * blockIdx.x;
    long iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (iy >= ny || ix >= nx) return;

    // Upwind scheme with CFL number 0.5
    long i = index(ix, iy, ny);

    // periodic boundary conditions in y
    long _im1_y = iy - 1;
    long im1_y = _im1_y + (_im1_y < 0) * ny;
    long im1 = index(ix, im1_y, ny);

    out[i] = in[i] - 0.5 * (in[i] - in[im1]);
}

__global__ void k_init(long nx, long ny, double hx, double hy, double *in, int block_x_offset) {
    long ix = threadIdx.x + blockDim.x * blockIdx.x;
    long iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (iy >= ny || ix >= nx) return;

    double x = -1.0 + (ix + block_x_offset * blockDim.x) * hx;
    double y = -1.0 + iy * hy;

    in[index(ix, iy, ny)] = exp(-50.0 * x * x - 50.0 * y * y);
}

int main() {
    int numGPUs;
    gpuErrorCheck(cudaGetDeviceCount(&numGPUs), true);

    if (numGPUs < 2) {
        printf("This program requires 2 GPUs to execute, found only %d.\nExiting...\n", numGPUs);
    }

    long nx = 4096;
    long ny = 4096;
    long N = nx * ny;

    // Grid spacing (domain [-1,1]x[-1,1]).
    double hx = 2.0 / double(nx);
    double hy = 2.0 / double(ny);

    long nx_local = nx / numGPUs;
    cudaStream_t streams[numGPUs];
    float time[numGPUs];
    cudaEvent_t start[numGPUs], stop[numGPUs];

    int threads = 128;

    dim3 threads_per_block(1, threads, 1);
    dim3 num_blocks(nx_local, ny / threads + 1, 1);

    printf("Lauch config:\n");
    printf("threads_per_block(%d,%d)\n", threads_per_block.x, threads_per_block.y);
    printf("num_blocks(%d,%d)\n\n", num_blocks.x, num_blocks.y);

    double *d_in[numGPUs];
    double *d_out[numGPUs];

    for (int dev = 0; dev < numGPUs; dev++) {
        gpuErrorCheck(cudaSetDevice(dev), true);

        gpuErrorCheck(cudaStreamCreate(&streams[dev]), true);

        printf("Length of local buffer: %ld\n", nx_local * ny);

        // Initialization.
        gpuErrorCheck(cudaMalloc(&d_in[dev], sizeof(double) * nx_local * ny), true);
        gpuErrorCheck(cudaMalloc(&d_out[dev], sizeof(double) * nx_local * ny), true);

        int block_x_offset = nx_local * dev;

        k_init<<<num_blocks, threads_per_block, 0, streams[dev]>>>(nx_local, ny, hx, hy, d_in[dev], block_x_offset);

        gpuErrorCheck(cudaEventCreate(&start[dev]), true);
        gpuErrorCheck(cudaEventCreate(&stop[dev]), true);

        // Do the actual computation.
        gpuErrorCheck(cudaEventRecord(start[dev], streams[dev]), true);

        for (long k = 0; k < 2 * ny; k++) {
            k_upwind<<<num_blocks, threads_per_block, 0, streams[dev]>>>(nx, ny, d_in[dev], d_out[dev]);
            swap(d_in[dev], d_out[dev]);
        }

        gpuErrorCheck(cudaEventRecord(stop[dev], streams[dev]), true);
    }

    // Check the result.
    double *h_in, *h_out;
    gpuErrorCheck(cudaMallocHost(&h_in, sizeof(double) * N), true);
    gpuErrorCheck(cudaMallocHost(&h_out, sizeof(double) * N), true);

    for (int dev = 0; dev < numGPUs; dev++) {
        gpuErrorCheck(cudaSetDevice(dev), true);
        gpuErrorCheck(cudaStreamSynchronize(streams[dev]), true);
        gpuErrorCheck(cudaEventElapsedTime(&time[dev], start[dev], stop[dev]), true);
        cout << "Runtime GPU " << dev << ": " << time[dev] * 1e-3 << " s" << endl;

        int block_x_offset = nx_local * dev;

        k_init<<<num_blocks, threads_per_block, 0, streams[dev]>>>(nx_local, ny, hx, hy, d_out[dev], block_x_offset);
    }

    for (int dev = 0; dev < numGPUs; dev++) {
        gpuErrorCheck(cudaSetDevice(dev), true);
        gpuErrorCheck(cudaStreamSynchronize(streams[dev]), true);
        gpuErrorCheck(cudaMemcpy(h_in + N / numGPUs * dev, d_in[dev], sizeof(double) * N / numGPUs, cudaMemcpyDeviceToHost), true);
        gpuErrorCheck(cudaMemcpy(h_out + N / numGPUs * dev, d_out[dev], sizeof(double) * N / numGPUs, cudaMemcpyDeviceToHost), true);
    }

    double error = 0.0;
    for (long i = 0; i < N; i++) {
        error = max(error, fabs(h_in[i] - h_out[i]));
    }
    cout << "Error: " << error << endl;

    return EXIT_SUCCESS;
}
