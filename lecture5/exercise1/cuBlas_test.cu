#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cublas_v2.h>

void handle_error(cudaError_t error_id, std::string msg) {
    if (error_id != cudaSuccess) {
        printf("Encountered cuda error %d. Msg: \"%s\"\n", error_id, msg.c_str());
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    int n = 10000;

    if (argc > 1) {
        int tmp = atoi(argv[1]);
        if (tmp > 0) {
            n = tmp;
        }
    }

    printf("Using arrays with %d elements.\n", n);

    // cudaSetDevice(0);

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle); 
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: cublasCreate failed. Error code: %d\n", status);
        exit(1);
    }

    // - - - - - -

    size_t bytes_A = (n * n) * sizeof(double);
    double* h_A = (double*)malloc(bytes_A); 
    for(int j=0;j<n;j++)
        for(int i=0;i<n;i++) 
            h_A[i + n*j] = i-j;
    
    double *d_A;
    handle_error(cudaMalloc(&d_A, bytes_A), "allocate d_A");
    handle_error(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice), "copy h_A to device");

    // - - - - - - -

    size_t bytes_xy = n * sizeof(double);

    double *h_x = (double *)malloc(bytes_xy);
    for(int j=0; j<n; j++)
        h_x[j] = j;

    double *d_x;
    handle_error(cudaMalloc(&d_x, bytes_xy), "allocate d_x");
    handle_error(cudaMemcpy(d_x, h_x, bytes_xy, cudaMemcpyHostToDevice), "copy h_x to device");

    double *h_y = (double *)malloc(bytes_xy);
    for(int j=0; j<n; j++)
        h_y[j] = 0;

    double *d_y;
    handle_error(cudaMalloc(&d_y, bytes_xy), "allocate d_y");
    handle_error(cudaMemcpy(d_y, h_y, bytes_xy, cudaMemcpyHostToDevice), "copy h_y to device");

    // - - - - - - -

    double alpha = 1.0;
    double beta = 0.0; // if beta=0 then d_y can be uninitialized 
    cublasDgemv(handle, CUBLAS_OP_N, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y, 1);

    // - - - - - - -

    handle_error(cudaMemcpy(h_y, d_y, bytes_xy, cudaMemcpyDeviceToHost), "copy d_y to host");

    free(h_A);
    free(h_x);
    free(h_y);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    cublasDestroy(handle);

    return EXIT_SUCCESS;
}