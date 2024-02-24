#include <cufft.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda.h>
#include "../../cuda_error_check.h"

using namespace std;

#define CUFFT_ASSERT(x) if ((x) != CUFFT_SUCCESS) { \
    cout << "Error at " << __FILE__ << ":" << __LINE__ << endl; \
    exit(1); \
}

void initialize_x(double *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = i * (1.0 / (n)) * (2 * M_PI);
    }
}

void initialize_f_hat(cufftComplex *f, double *x, int n) {
    for (int i = 0; i < n; i++) {
        f[i].x = sin(x[i]);
    }
}

void init_k(double *k, int n) {
    for (int i = 1; i < n/2; i++) {
        k[i] = i;
    }
    for (int i = n/2; i < n; i++) {
        k[i] = i - n;
    }
}


// ------------------ Main ------------------
int main(int argc, char **argv) {
    int n = 1 << 3;
    double *x;
    cufftComplex *f, *f_hat, *u, *u_hat;

    gpuErrorCheck(cudaMallocManaged(&x, n * sizeof(double)));
    gpuErrorCheck(cudaMallocManaged(&f, n * sizeof(cufftComplex)));
    gpuErrorCheck(cudaMallocManaged(&f_hat, n * sizeof(cufftComplex)));
    gpuErrorCheck(cudaMallocManaged(&u, n * sizeof(cufftComplex)));
    gpuErrorCheck(cudaMallocManaged(&u_hat, n * sizeof(cufftComplex)));

    initialize_x(x, n);
    initialize_f_hat(f, x, n);

    for (int i = 0; i < n; i++) {
        cout << f[i].x << " ";
    }
    cout << endl;

    cufftHandle handle;
    CUFFT_ASSERT(cufftPlan1d(&handle, n, CUFFT_C2C, 1));

    // transform f to f_hat
    CUFFT_ASSERT(cufftExecC2C(handle, f, f_hat, CUFFT_FORWARD));
    cudaDeviceSynchronize();

    // Initialize k vector
    double *k;
    gpuErrorCheck(cudaMallocManaged(&k, n * sizeof(double)));
    init_k(k, n);

    for (int i = 1; i < n; i++) {
        u_hat[i].x = -f_hat[i].x / (4 * M_PI * M_PI * k[i] * k[i]);
        u_hat[i].y = -f_hat[i].y / (4 * M_PI * M_PI * k[i] * k[i]);
    }

    // transform u_hat to u
    CUFFT_ASSERT(cufftExecC2C(handle, u_hat, u, CUFFT_INVERSE));
    cudaDeviceSynchronize();

    // Print u values
    for (int i = 0; i < n; i++) {
        cout << u[i].x << " ";
    }
    cout << endl;

    // Free the memory
    gpuErrorCheck(cudaFree(x));
    gpuErrorCheck(cudaFree(f));
    gpuErrorCheck(cudaFree(f_hat));
    gpuErrorCheck(cudaFree(u));
    gpuErrorCheck(cudaFree(u_hat));

    // Destroy handle
    CUFFT_ASSERT(cufftDestroy(handle));

    return EXIT_SUCCESS;
}