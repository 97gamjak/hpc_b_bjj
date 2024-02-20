#include <cusparse.h>
#include <iostream>
#include <cuda_runtime.h>

// create sparse 3 point stencil matrix
void create3PointStencil(int n, int* rows, int* columns, double* values) {
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

// initialize the dense vector
void initDenseVector(int n, double* x) {
    for (int i = 0; i < n; i++) {
        x[i] = sin(double (i) / n * M_PI);
    }
}

// --------- main ---------
int main(int argc, char* argv[]) {

    int n = 1 << 3;
    int A_num_rows = n;
    int A_num_cols = n;
    int A_nnz = 3 * n - 2;

    int* A_rows_csr;
    int* A_columns;
    double* A_values;

    A_rows_csr = (int*)malloc( (A_num_rows + 1) * sizeof(int));
    A_columns = (int*)malloc(A_nnz * sizeof(int));
    A_values = (double*)malloc(A_nnz * sizeof(double));
    

    create3PointStencil(n, A_rows_csr, A_columns, A_values);

    // init
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    if (handle == nullptr) {
        std::cerr << "Failed to create handle" << std::endl;
        return 1;
    }

    // setup the CSR matrix
    cusparseSpMatDescr_t d_A;
    cusparseCreateCsr(&d_A, A_num_rows, A_num_cols, A_nnz,
        A_rows_csr, A_columns, A_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // setup the dense vector
    double *h_x, *h_y;
    h_x = (double*)malloc(A_num_cols * sizeof(double));
    h_y = (double*)calloc(A_num_rows, sizeof(double));

    initDenseVector(A_num_cols, h_x);

    // print the dense vector
    for (int i = 0; i < A_num_cols; i++) {
        std::cout << h_x[i] << " ";
    }
    std::cout << std::endl;

    // setup the dense vector
    cusparseDnVecDescr_t d_x, d_y;
    cusparseCreateDnVec(&d_x, A_num_cols, h_x, CUDA_R_64F);
    cusparseCreateDnVec(&d_y, A_num_rows, h_y, CUDA_R_64F);

    // determine if we need some external storage space
    void *d_buffer;
    size_t bufferSize;

    double alpha = 1.0;
    double beta = 0.0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, d_A, d_x, &beta, d_y, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, &bufferSize);

    cudaMalloc(&d_buffer, bufferSize);

    // computes y = alpha op(A) x + beta y
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, d_A, d_x, &beta, d_y, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    
    cudaDeviceSynchronize();

    // copy result back to host
    cusparseDnVecGetValues(d_y,(void **) &h_y);

    // print the result
    for (int i = 0; i < A_num_rows; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    // free
    free(A_rows_csr); free(A_columns); free(A_values);
    free(h_x); free(h_y);
    cudaFree(d_buffer);

    // destroy
    cusparseDestroy(handle);

    return 0;
}