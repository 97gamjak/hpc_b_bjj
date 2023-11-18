#include <stdio.h>
#include <stdlib.h>

#include "../../cuda_error_check.h"

#define IND(x, y, N) ((y) * (N) + (x))

#define CLAMP(val, lo, hi) (max((lo), min((hi), (val))))

// swap matrices (just pointers, not content)
#define SWAP(A, B)       \
    {                    \
        float* temp = A; \
        A = B;           \
        B = temp;        \
    }

void printTemperature(float* m, int N, int M);
void printCurrentState(float*, int, int);

void propagate_gpu(float*, float*, int, int, int, int, int);
__global__ void propagate_step_gpu(float*, float*, int, int, int);

// ----------------------

int main(int argc, char** argv) {
    // 'parsing' optional input parameter = problem size
    int N = 512;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    int T = N * 300;
    printf("Computing heat-distribution for room size N=%d for T=%d timesteps\n", N, T);

    // ---------- setup ----------

    // create a buffer for storing temperature fields
    float* A;
    gpuErrorCheck(cudaMallocManaged(&A, N * N * sizeof(float)), true);

    // create a second buffer for the computation
    float* B;
    gpuErrorCheck(cudaMallocManaged(&B, N * N * sizeof(float)), true);

    // set up initial conditions in A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[IND(i, j, N)] = B[IND(i, j, N)] = 273;  // temperature is 0° C everywhere (273 K)
        }
    }

    // and there is a heat source in one corner
    int source_x = N / 4;
    int source_y = N / 4;
    A[IND(source_x, source_y, N)] = B[IND(source_x, source_y, N)] = 273 + 60;

    printf("Initial:\n");
    printTemperature(A, N, N);

    // ---------- compute ----------

    // -- BEGIN ASSIGNMENT --

    propagate_gpu(A, B, N, source_x, source_y, T, 1000);

    // -- END ASSIGNMENT --

    cudaFree(B);

    // ---------- check ----------

    printf("Final:\n");
    printTemperature(A, N, N);

    bool success = true;
    for (long long i = 0; i < N; i++) {
        for (long long j = 0; j < N; j++) {
            float temp = A[IND(i, j, N)];
            if (273 <= temp && temp <= 273 + 60) continue;
            success = false;
            break;
        }
    }

    printf("Verification: %s\n", success ? "OK" : "FAILED");

    // ---------- cleanup ----------

    cudaFree(A);

    // done
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

__global__ void propagate_step_gpu(float* A, float* B, int N, int source_x, int source_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Disregard indices outside array boundaries and indices of heat source
    if (i >= N || j >= N || (i == source_x && j == source_y)) return;

    // get current temperature at (i,j)
    float tc = A[IND(i, j, N)];

    // get temperatures left/right and up/down
    float tl = A[IND(i, CLAMP(j - 1, 0, N - 1), N)];
    float tr = A[IND(i, CLAMP(j + 1, 0, N - 1), N)];
    float tu = A[IND(CLAMP(i - 1, 0, N - 1), j, N)];
    float td = A[IND(CLAMP(i + 1, 0, N - 1), j, N)];

    float incr = 0.2 * (tl + tr + tu + td + (-4 * tc));

    // update temperature at current point
    B[IND(i, j, N)] = tc + incr;

    return;
}

void propagate_gpu(float* A, float* B, int N, int source_x, int source_y, int T, int outfreq) {
    // determine launch configuration
    dim3 block_size(32, 32);  // threads per block
    int grid_size_xy = (int)ceil(double(N) / double(block_size.x));
    dim3 grid_size(grid_size_xy, grid_size_xy);

    // for each time step ..
    for (int t = 0; t < T; t++) {
        // launch kernel to propagate for a single time step
        propagate_step_gpu<<<grid_size, block_size>>>(A, B, N, source_x, source_y);

        gpuErrorCheck(cudaDeviceSynchronize(), true);

        SWAP(A, B);

        if (!(t % outfreq)) printCurrentState(A, N, t);
    }
    return;
}

void printTemperature(float* m, int N, int M) {
    const char* colors = " .-:=+*#%@";
    const int numColors = 10;

    // boundaries for temperature (for simplicity hard-coded)
    const float max = 273 + 30;
    const float min = 273 + 0;

    // set the 'render' resolution
    int H = 30;
    int W = 50;

    // step size in each dimension
    int sH = N / H;
    int sW = M / W;

    // upper wall
    for (int i = 0; i < W + 2; i++) {
        printf("X");
    }
    printf("\n");

    // room
    for (int i = 0; i < H; i++) {
        // left wall
        printf("X");
        // actual room
        for (int j = 0; j < W; j++) {
            // get max temperature in this tile
            float max_t = 0;
            for (int x = sH * i; x < sH * i + sH; x++) {
                for (int y = sW * j; y < sW * j + sW; y++) {
                    max_t = (max_t < m[x * N + y]) ? m[x * N + y] : max_t;
                }
            }
            float temp = max_t;

            // pick the 'color'
            int c = ((temp - min) / (max - min)) * numColors;
            c = (c >= numColors) ? numColors - 1 : ((c < 0) ? 0 : c);

            // print the average temperature
            printf("%c", colors[c]);
        }
        // right wall
        printf("X\n");
    }

    // lower wall
    for (int i = 0; i < W + 2; i++) {
        printf("X");
    }
    printf("\n");
}

void printCurrentState(float* A, int N, int t) {
    printf("Step t=%d:\n", t);
    printTemperature(A, N, N);
    return;
}
