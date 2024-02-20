
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

void initialize_matrix(double *A, int n);
void initialize_vector(double *x, int n);
void verify_result(double *y, int n);
void check(int argc, char **argv);
int read_args(int argc, char **argv);

// ------------------ Main ------------------
int main(int argc, char **argv)
{
    int n = read_args(argc, argv);

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        cout << "ERROR: cublasCreate failed. Error code: " << status << endl;
        exit(1);
    }

    size_t size_A = n * n * sizeof(double);
    size_t size_x = n * sizeof(double);
    double *h_A = (double *)malloc(size_A);
    double *h_x = (double *)malloc(size_x);
    double *h_y = (double *)malloc(size_x);

    initialize_matrix(h_A, n);
    initialize_vector(h_x, n);

    double *d_A;
    cudaMalloc(&d_A, size_A);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

    double *d_x;
    cudaMalloc(&d_x, size_x);
    cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice);

    double *d_y;
    cudaMalloc(&d_y, size_x);

    double alpha = 1.0;
    double beta = 0.0; // if beta=0 then d_y can be uninitialized
    cublasOperation_t op;
    op = CUBLAS_OP_N; // (no transpose)
    // op = CUBLAS_OP_T (tranpose of A)
    // op = CUBLAS_OP_C (conjugate transpose of A).
    
    // Start the timer
    double a = clock();   

    // Usually lda=n, incx=1, and incy=1
    cublasDgemv(handle, op, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y, 1);
    cudaDeviceSynchronize();

    // Stop the timer
    double b = clock();

    cudaMemcpy(h_y, d_y, size_x, cudaMemcpyDeviceToHost);

    verify_result(h_y, n);

    // Destroy the handle
    cublasDestroy(handle);

    // Free the memory
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
    free(h_A); free(h_x); free(h_y);

    cout << n << " in " << (b - a) / CLOCKS_PER_SEC * 1e6 << " µs" << endl;

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