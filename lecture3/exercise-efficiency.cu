#include <iostream>
#include <cmath>
#include <fstream>
using namespace std;


__global__
void k_matvecmul(long n, double* in, double* out, double* mat) {
    long i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i>0 && i < n-1)
        out[i] = mat[0]*in[i] + mat[1]*in[i+1] + mat[2]*in[i-1];
}

int main() {
    long n = 1e6;
    long time_steps = 100;
    double *h_in, *d_mat;
    double *d_in, *d_out;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // To be more realistic, we do not time memory allocation.
    cudaMallocHost(&h_in, sizeof(double) * n);
    cudaMalloc(&d_in, sizeof(double) * n);
    cudaMalloc(&d_out, sizeof(double) * n);

    cudaEventRecord(start, 0);

    // intialize vector and copy to GPU
    for (long i = 0; i < n; i++)
        h_in[i] = sin(double(i) / double(n - 1) * M_PI);

    cudaMemcpy(d_in, h_in, sizeof(double) * n, cudaMemcpyHostToDevice);

    // initialize matrix and copy to GPU
    double matrix_row[3] = {1.0 - 0.25 * 2.0, 0.25, 0.25};
    cudaMalloc(&d_mat, sizeof(double) * 3);
    cudaMemcpy(d_mat, matrix_row, sizeof(double) * 3, cudaMemcpyHostToDevice);

    int threads_per_block = 64;
    int blocks_per_grid = n/threads_per_block + 1;

    // repeated matrix-vector multiplication (i.e. time integration)
    for(long k=0;k<time_steps;k++) {
        k_matvecmul<<<blocks_per_grid, threads_per_block>>>(n, d_in, d_out, d_mat);

        cudaMemcpy(d_in, d_out, sizeof(double) * n, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(stop, 0);

    // Write result to a file (we do not time this).
    cudaMemcpy(h_in, d_in, sizeof(double) * n, cudaMemcpyDeviceToHost);
    ofstream fs("result.data");
    for (long i = 0; i < n; i++)
        fs << h_in[i] << endl;
    fs.close();

    // Compare to the exact solution (we do not time this).
    double error = 0.0;
    double decay = exp(-0.25 / pow(double(n - 1), 2) * time_steps * pow(M_PI, 2));
    for (long i = 0; i < n; i++)
        error = max(error, fabs(h_in[i] - decay * sin(double(i) / double(n - 1) * M_PI)));
    cout << "Numerical error: " << error << endl;

    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout << time * 1e-3 << " s" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}
