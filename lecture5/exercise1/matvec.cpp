
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../../cuda_error_check.h"

using namespace std;

#define CUBLAS_ASSERT(x) if ((x) != CUBLAS_STATUS_SUCCESS) { \
    cout << "Error at " << __FILE__ << ":" << __LINE__ << endl; \
    exit(1); \
}

void initialize_matrix(double *A, int n);
void initialize_vector(double *x, int n);
void verify_result(double *y, int n);
void check(int argc, char **argv);
int read_args(int argc, char **argv);

// ------------------ Main ------------------
int main(int argc, char **argv)
{
    int n = read_args(argc, argv);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    CUBLAS_ASSERT(cublasCreate(&handle));

    size_t size_A = n * n * sizeof(double);
    size_t size_x = n * sizeof(double);
    double *h_A = (double *)malloc(size_A);
    double *h_x = (double *)malloc(size_x);
    double *h_y = (double *)malloc(size_x);

    initialize_matrix(h_A, n);
    initialize_vector(h_x, n);

    double *d_A;
    gpuErrorCheck(cudaMalloc(&d_A, size_A));
    gpuErrorCheck(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));

    double *d_x;
    gpuErrorCheck(cudaMalloc(&d_x, size_x));
    gpuErrorCheck(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));

    double *d_y;
    gpuErrorCheck(cudaMalloc(&d_y, size_x));

    double alpha = 1.0;
    double beta = 0.0; // if beta=0 then d_y can be uninitialized
    cublasOperation_t op;
    op = CUBLAS_OP_N; // (no transpose)
    // op = CUBLAS_OP_T (tranpose of A)
    // op = CUBLAS_OP_C (conjugate transpose of A).
    
    // Start the timer
    cudaEventRecord(start, 0);  

    // Usually lda=n, incx=1, and incy=1
    CUBLAS_ASSERT(cublasDgemv(handle, op, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y, 1));
    cudaDeviceSynchronize();

    // Stop the timer
    cudaEventRecord(stop, 0);

    gpuErrorCheck(cudaMemcpy(h_y, d_y, size_x, cudaMemcpyDeviceToHost));

    verify_result(h_y, n);

    // Destroy the handle
    CUBLAS_ASSERT(cublasDestroy(handle));

    // Free the memory
    gpuErrorCheck(cudaFree(d_A)); 
    gpuErrorCheck(cudaFree(d_x)); 
    gpuErrorCheck(cudaFree(d_y));
    free(h_A); 
    free(h_x); 
    free(h_y);

    // Calculate the time
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Elapsed time: " << elapsedTime << " ms" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// ------------------ Functions ------------------

void initialize_matrix(double *A, int n)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            A[i + j * n] = 1.0;

    return;
}

void initialize_vector(double *x, int n)
{
    for (int i = 0; i < n; i++)
        x[i] = 1.0;
    return;
}

void verify_result(double *y, int n)
{
    double expected = n;
    for (int i = 0; i < n; i++)
    {
        if (y[i] != expected)
        {
            cout << "ERROR: mismatch at position " << i << " expected " << expected << " but got " << y[i] << endl;
            return;
        }
    }
    // cout << "PASSED!" << endl;
    return;
}

void check(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " m" << endl;
        cout << "m: matrix dimension will be 2^m x 2^m" << endl;
        exit(1);
    }
    return ;
}

int read_args(int argc, char **argv)
{
    int m;
    check(argc, argv);
    m = atoi(argv[1]);
    return 1 << m;
}