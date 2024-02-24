#include <iostream>
#include <cmath>
using namespace std;


__global__
void k_sum(long n, double* vec, double* result) {
    extern __shared__ float sdata[];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load the data into shared memory
    sdata[tid] = (i < n) ? vec[i] : 0;
    __syncthreads();

    // Do the reduction in shared memory
    for(int s=blockDim.x/2; s>0; s>>=1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if(tid == 0) {
        result[blockIdx.x] = sdata[0];
    }

    __syncthreads();
}

int main() {
    long n = 8*1024*1024; // must be a power of 2
    double h_result = 0.0;
    double *h_vec, *d_vec, *d_tmp;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMallocHost(&h_vec, sizeof(double)*n);
    cudaMalloc(&d_vec, sizeof(double)*n);
    cudaMalloc(&d_tmp, sizeof(double)*(n/256));

    // Initialie vec and copy to GPU.
    for(long i=0;i<n;i++)
        h_vec[i] = 1.0/pow(double(i+1),2);
    cudaMemcpy(d_vec, h_vec, sizeof(double)*n, cudaMemcpyHostToDevice);

    // calculate number of blocks and threads per block
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEventRecord(start, 0);
    while(numBlocks > 1) {
        k_sum<<<numBlocks, blockSize, blockSize*sizeof(double)>>>(n, d_vec, d_tmp);
        cudaDeviceSynchronize();
        n = numBlocks;
        d_vec = d_tmp;
        numBlocks = (n + blockSize - 1) / blockSize;
    }
    cudaEventRecord(stop, 0);

    k_sum<<<1, blockSize, blockSize*sizeof(double)>>>(n, d_vec, d_tmp);

    // Copy the result back to the host.
    cudaMemcpy(&h_result, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    
    // Check the result.
    cout << "Result: " << h_result << endl;
    if(fabs(h_result - pow(M_PI,2)/6.0) < 1e-5) {
        cout << "Correct!" << endl;
    } else {
        cout << "The computed result does not match with the expected result ("
             << pow(M_PI,2)/6.0 << ")" << endl;
    }

    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cout << time * 1e-3 << " s" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free the memory.
    cudaFreeHost(h_vec);
    cudaFree(d_vec);
    cudaFree(d_tmp);
    return 0;
}

