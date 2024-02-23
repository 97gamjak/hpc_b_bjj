#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <functional>
#include <string>

__global__ void vec_add(double *a, double *b, double *c, long n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        c[id] = a[id] + b[id];
}

void initialize(double *arr, int size, const double std::function<double(int)> &fun) {
    for (int i = 0; i < size; i++) {
        arr[i] = fun(i);
    }
}

void handle_error(cudaError_t error_id, std::string msg) {
    if (error_id != cudaSuccess) {
        printf("Encountered cuda error %d. Msg: \"%s\"\n", error_id, msg.c_str());
        exit(EXIT_FAILURE);
    }
}

void compute_max_err(double *result, int n) {
    double max_error = 0.0;
    int err_index = -1;
    for (int i = 0; i < n; i++) {
        double err = fabs(1 - result[i]);
        if (err >= max_error) {
            max_error = err;
            err_index = i;
        }
    }
    printf("Maximum error was %.3e at index %d\n", max_error, err_index);
}

int main(int argc, char *argv[]) {
    int n = 100000000;  // approx 2.4 GiB for a,b and c

    if (argc > 1) {
        int tmp = atoi(argv[1]);
        if (tmp > 0) {
            n = tmp;
        }
    }

    printf("Using arrays with %d elements.\n", n);

    // cudaSetDevice(0);

    size_t bytes = n * sizeof(double);
    double *h_a = (double *)malloc(bytes);
    double *d_a;
    handle_error(cudaMalloc(&d_a, bytes), "allocate d_a");

    double *h_b = (double *)malloc(bytes);
    double *d_b;
    handle_error(cudaMalloc(&d_b, bytes), "allocate d_b");

    double *h_c = (double *)malloc(bytes);
    double *d_c;
    handle_error(cudaMalloc(&d_c, bytes), "allocate d_c");

    initialize(h_a, n, [](int i) { double d = sin(i); return d*d; });
    initialize(h_b, n, [](int i) { double d = cos(i); return d*d; });

    handle_error(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "copy h_a to device");
    handle_error(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "copy h_b to device");

    int block_size = 1024;  // threads per block
    int grid_size = (int)ceil(double(n) / double(block_size));

    vec_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    handle_error(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "copy d_c to host");

    compute_max_err(h_c, n);

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return EXIT_SUCCESS;
}