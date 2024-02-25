#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../cuda_error_check.h"

using std::cout;
using std::endl;

#define USE_DOUBLE_PRECISION

#ifdef USE_DOUBLE_PRECISION
#define REAL double
#define CUDA_REAL CUDA_R_64F
#else
#define REAL float
#define CUDA_REAL CUDA_R_32F
#endif

#define CUSPARSE_ASSERT(X){\
        cusparseStatus_t ___resCuda = (X);\
        if (CUSPARSE_STATUS_SUCCESS != ___resCuda){\
            printf("Error: fails, %s (%s line %d)\nbCols", cusparseGetErrorString(___resCuda), __FILE__, __LINE__);\
            exit(1);\
        }\
    }

// create 3 point stencil matrix in CSR format
void create3PointStencil(int n, int* rows, int* columns, REAL* values) {
    int index = 0;
    for (int i = 0; i < n; i++) {
        rows[i] = index;
        if (i > 0) {
            columns[index] = i - 1;
            values[index] = 1.0;
            index++;
        }
        columns[index] = i;
        values[index] = -2.0;
        index++;
        if (i < n - 1) {
            columns[index] = i + 1;
            values[index] = 1.0;
            index++;
        }
    }
    rows[n] = index;
}

template<typename T>
void print_vec(T* vec, int len, const char* text = "vec") {
    cout << text << ": [ ";
    for (int i = 0; i < len; i++)
        cout << vec[i] << " ";
    cout << "]\n";
}

void print_mat_csr(int A_num_rows,
    int A_nnz, int* A_csr_offsets, int* A_csr_col_idx, REAL* A_values) {
    print_vec(A_csr_offsets, A_num_rows, "A_csr_offsets");
    print_vec(A_csr_col_idx, A_nnz, "A_col_idx");
    print_vec(A_values, A_nnz, "A_values");
}

// initialize the dense vector
void initDenseVector(int n, REAL* x) {
    for (int i = 0; i < n; i++) {
        x[i] = sin(i / (double)n * M_PI * 2.0);
    }
}

int main(void) {
    int n = 1 << 27;
    int A_num_rows = n;
    int A_num_cols = n;
    int A_nnz = 3 * n - 2;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cout << "nrows: " << A_num_rows << "\nncols: " << A_num_cols << endl;

    // start timing for SpMV
    cudaEventRecord(start, 0);

    // init library
    cusparseHandle_t handle;
    CUSPARSE_ASSERT(cusparseCreate(&handle));

    // allocate array for CSR format offsets on GPU and CPU
    int* h_A_csr_offsets, * d_A_csr_offsets;
    h_A_csr_offsets = (int*)calloc(A_num_rows + 1, sizeof(int));
    gpuErrorCheck(cudaMalloc(&d_A_csr_offsets, (A_num_rows + 1) * sizeof(int)));

    // allocate array for CSR format column pointer on GPU and CPU
    int* h_A_csr_col_idx, * d_A_csr_col_idx;
    h_A_csr_col_idx = (int*)calloc(A_nnz, sizeof(int));
    gpuErrorCheck(cudaMalloc(&d_A_csr_col_idx, A_nnz * sizeof(int)));

    // allocate array for CSR format nonzero values on GPU and CPU
    REAL* h_A_values, * d_A_values;
    h_A_values = (REAL*)calloc(A_nnz, sizeof(REAL));
    gpuErrorCheck(cudaMalloc(&d_A_values, A_nnz * sizeof(REAL)));

    // fill CSR matrix arrays with values for stencil
    create3PointStencil(n, h_A_csr_offsets, h_A_csr_col_idx, h_A_values);
    // print_mat_csr(A_num_rows, A_nnz, h_A_csr_offsets, h_A_csr_col_idx, h_A_values);

    // Copy CSR matrix individual arrays to GPU
    gpuErrorCheck(cudaMemcpy(d_A_csr_offsets, h_A_csr_offsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_A_csr_col_idx, h_A_csr_col_idx, A_nnz * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_A_values, h_A_values, A_nnz * sizeof(REAL), cudaMemcpyHostToDevice));

    // build cusparse internal CSR matrix data structure from individual arrays
    cusparseSpMatDescr_t d_matdesc_A;
    CUSPARSE_ASSERT(
        cusparseCreateCsr(
            &d_matdesc_A,
            A_num_rows,
            A_num_cols,
            A_nnz,
            d_A_csr_offsets,
            d_A_csr_col_idx,
            d_A_values,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_REAL
        )
    );

    // setup the dense vectors for input(x) and output(y) on host and device + cusparse data structure
    REAL* h_x, * d_x;
    h_x = (REAL*)calloc(A_num_cols, sizeof(REAL));
    initDenseVector(A_num_cols, h_x);
    gpuErrorCheck(cudaMalloc(&d_x, A_num_cols * sizeof(REAL)));
    gpuErrorCheck(cudaMemcpy(d_x, h_x, A_num_cols * sizeof(REAL), cudaMemcpyHostToDevice));
    cusparseDnVecDescr_t d_vecdesc_x;
    CUSPARSE_ASSERT(cusparseCreateDnVec(&d_vecdesc_x, A_num_cols, d_x, CUDA_REAL));

    REAL* h_y, * d_y;
    h_y = (REAL*)calloc(A_num_rows, sizeof(REAL));
    gpuErrorCheck(cudaMalloc(&d_y, A_num_rows * sizeof(REAL)));
    gpuErrorCheck(cudaMemcpy(d_y, h_y, A_num_rows * sizeof(REAL), cudaMemcpyHostToDevice));
    cusparseDnVecDescr_t d_vecdesc_y;
    CUSPARSE_ASSERT(cusparseCreateDnVec(&d_vecdesc_y, A_num_rows, d_y, CUDA_REAL));

    // print_vec(h_x, A_num_cols, "X");

    // determine if we need some external storage space
    void* d_buffer;
    size_t bufferSize;

    REAL alpha = 1;
    REAL beta = 0; // just Mat * Vec, summand = 0

    CUSPARSE_ASSERT(
        cusparseSpMV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            d_matdesc_A,
            d_vecdesc_x,
            &beta,
            d_vecdesc_y,
            CUDA_REAL,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &bufferSize
        )
    );

    // cout << "Requierd buffersize: " << bufferSize << " B" << endl;

    gpuErrorCheck(cudaMalloc(&d_buffer, bufferSize));

    // computes y = alpha op(A) x + beta y
    CUSPARSE_ASSERT(
        cusparseSpMV(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            d_matdesc_A,
            d_vecdesc_x,
            &beta,
            d_vecdesc_y,
            CUDA_REAL,
            CUSPARSE_SPMV_ALG_DEFAULT,
            d_buffer
        )
    );

    // stop timing for SpMV
    cudaEventRecord(stop, 0);

    gpuErrorCheck(cudaDeviceSynchronize());

    CUSPARSE_ASSERT(cusparseDestroySpMat(d_matdesc_A));
    CUSPARSE_ASSERT(cusparseDestroyDnVec(d_vecdesc_x));
    CUSPARSE_ASSERT(cusparseDestroyDnVec(d_vecdesc_y));
    CUSPARSE_ASSERT(cusparseDestroy(handle));

    // copy result back to host and print
    gpuErrorCheck(cudaMemcpy(h_y, d_y, A_num_rows * sizeof(REAL), cudaMemcpyDeviceToHost));
    // print_vec(h_y, A_num_rows, "Y");

    // free memory
    free(h_A_csr_offsets);
    free(h_A_csr_col_idx);
    free(h_A_values);
    free(h_x);
    free(h_y);

    // free device memory
    gpuErrorCheck(cudaFree(d_A_csr_offsets));
    gpuErrorCheck(cudaFree(d_A_csr_col_idx));
    gpuErrorCheck(cudaFree(d_A_values));
    gpuErrorCheck(cudaFree(d_x));
    gpuErrorCheck(cudaFree(d_y));
    gpuErrorCheck(cudaFree(d_buffer));

    // print timing
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout << "TIME: " << time * 1e-3 << " s" << endl;

    // destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}