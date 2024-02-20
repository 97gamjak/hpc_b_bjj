#include <cufft.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda.h>

using namespace std;

void initialize_x(double *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = i * (1.0 / (n + 1));
    }
}

void initialize_f_hat(cufftComplex *f_hat, double *x, int n) {
    for (int i = 0; i < n; i++) {
        f_hat[i].x = sin(x[i] * 2 * M_PI);
    }
}

void check_plan(cufftHandle plan) {
    if (plan == CUFFT_SETUP_FAILED) {
        cout << "ERROR: plan creation failed." << endl;
        exit(1);
    }
}


// ------------------ Main ------------------
int main() {
    int n = 1 << 3;
    double *x;
    cufftComplex *f_hat, *u_hat;

    cudaMallocManaged(&x, n * sizeof(double));
    cudaMallocManaged(&f_hat, n * sizeof(cufftComplex));
    cudaMallocManaged(&u_hat, n * sizeof(cufftComplex));

    initialize_x(x, n);
    initialize_f_hat(f_hat, x, n);

    for (int i = 0; i < n; i++) {
        cout << f_hat[i].x << " ";
    }
    cout << endl;

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_D2Z, 1);
    check_plan(plan);

    // transform f to f_hat
    cufftExecC2C(plan, f_hat, f_hat, CUFFT_FORWARD);

    // Initialize k vector
    double *k;
    cudaMallocManaged(&k, n * sizeof(double));
    for (int i = 0; i <= n/2 - 1; i++) {
        k[i] = i;
    }
    for (int i = n/2; i < n; i++) {
        k[i] = i - n;
    }

    for (int i = 0; i < n; i++) {
        if (k[i] != 0) {
            u_hat[i].x = -f_hat[i].x / (4 * M_PI * M_PI * k[i] * k[i]);
            u_hat[i].y = -f_hat[i].y / (4 * M_PI * M_PI * k[i] * k[i]);
        } else {
            u_hat[i].x = 0;
            u_hat[i].y = 0;
        }
    }

    // transform u_hat to u
    cufftExecC2C(plan, u_hat, u_hat, CUFFT_INVERSE);

    for (int i = 0; i < n; i++) {
        cout << u_hat[i].x << " ";
    }
    cout << endl;

    // Free the memory
    cudaFree(x);
    cudaFree(f_hat);
    cudaFree(u_hat);

    // Destroy plan
    cufftDestroy(plan);

    return 0;
}