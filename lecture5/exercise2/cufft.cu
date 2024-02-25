#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <cufft.h>
#include "../../cuda_error_check.h"

using namespace std;

#define CUFFT_ASSERT(x) if ((x) != CUFFT_SUCCESS) { \
    cout << "Error at " << __FILE__ << ":" << __LINE__ << endl; \
    exit(1); \
}

void initialize_x(double *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = i * (1.0 / double(n)) * (2 * M_PI);
    }
}

void initialize_f(cufftDoubleComplex *f, double *x, int n) {
    for (int i = 0; i < n; i++) {
        f[i].x = cos(x[i]);
    }
}

void init_k(double *k, int n) {
    for (int i = 0; i <= n/2; i++) {
        k[i] = i;
    }
    for (int i = n/2 + 1; i < n; i++) {
        k[i] = (i - n);
    }
}

void verify_u(cufftDoubleComplex *u, double *x, int n) {
    for (int i = 0; i < n; i++) {
        double u_exact = cos(x[i]);
        double u_approx = u[i].x; // -cos(x[i]);
        if (fabs(u_exact + u_approx) > 1e-6) {
            cout << "Verification failed at i = " << i << endl;
            cout << "Exact: " << u_exact << " Approx: " << u_approx << endl;
            exit(1);
        }
    }
    cout << "Verification passed!" << endl;
}

// ------------------ Main ------------------
int main(int argc, char **argv) {
    int n = 1 << 28;
    double *x;
    cufftDoubleComplex *f_hat, *u_hat;
    cufftDoubleComplex *f, *u;

    gpuErrorCheck(cudaMallocManaged(&x, n * sizeof(double)));
    gpuErrorCheck(cudaMallocManaged(&f, n * sizeof(cufftDoubleComplex)));
    gpuErrorCheck(cudaMallocManaged(&f_hat, n * sizeof(cufftDoubleComplex)));
    gpuErrorCheck(cudaMallocManaged(&u, n * sizeof(cufftDoubleComplex)));
    gpuErrorCheck(cudaMallocManaged(&u_hat, n * sizeof(cufftDoubleComplex)));

    initialize_x(x, n);
    initialize_f(f, x, n);

    cufftHandle handle;
    CUFFT_ASSERT(cufftPlan1d(&handle, n, CUFFT_Z2Z, 1));

    // transform f to f_hat
    CUFFT_ASSERT(cufftExecZ2Z(handle, f, f_hat, CUFFT_FORWARD));
    cudaDeviceSynchronize();

    // Initialize k vector
    double *k;
    gpuErrorCheck(cudaMallocManaged(&k, n * sizeof(double)));
    init_k(k, n);

    for (int i = 1; i < n; i++) {
        u_hat[i].x = -f_hat[i].x / (k[i] * k[i] * n);
        u_hat[i].y = -f_hat[i].y / (k[i] * k[i] * n);
    }

    u_hat[0].x = f_hat[0].x;
    u_hat[0].y = f_hat[0].y;

    cudaDeviceSynchronize();

    // transform u_hat to u
    CUFFT_ASSERT(cufftExecZ2Z(handle, u_hat, u, CUFFT_INVERSE));
    cudaDeviceSynchronize();

    // Verify u
    verify_u(u, x, n);

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