#include <iostream>
#include <vector>
using namespace std;

__global__
void matmul(long n, float* A, float* B, float* C) {
    long i = blockIdx.x*blockDim.x + threadIdx.x;
    long j = blockIdx.y*blockDim.y + threadIdx.y;

    float val=0.0;
    for(long k=0;k<n;k++)
        val += A[i+k*n]*B[k+j*n];

    C[i+j*n] = val;
}

const long BS = 32;

__global__
void matmul_fast(long n, float* A, float* B, float* C) {
    long i = threadIdx.x; long bi = blockIdx.x;
    long j = threadIdx.y; long bj = blockIdx.y;
    
    // loop over all sub-matrices
    double val = 0.0;
    for(long m=0;m<n/BS;m++) {
        __shared__ double block_A[BS*BS];
        __shared__ double block_B[BS*BS];
        // load block into shared memory
        block_A[i+BS*j] = A[bi*BS+i + n*(m*BS+j)];
        block_B[i+BS*j] = B[m*BS+i + n*(bj*BS+j)];
        
        // wait until all threads have caught up
        __syncthreads();
        
        // compute the (sub-)matrix-matrix product
        for(long k=0;k<BS;k++)
            val += block_A[i+BS*k]*block_B[k+BS*j];


        // make sure that all threads are finished before
        // next loop iteration starts.
        __syncthreads();
    }
    // update result in global memory
    C[bi*BS+i + n*(bj*BS+j)] = val;
}

int main() {
    long n = 8192;  // must be a power of 2

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float)*n*n);
    cudaMalloc(&d_B, sizeof(float)*n*n);
    cudaMalloc(&d_C, sizeof(float)*n*n);

    vector<float> h_A(n*n), h_B(n*n), h_C(n*n);

    for(long j=0;j<n;j++)
        for(long i=0;i<n;i++) {
            h_A[i + n*j] = float(i)/float(n);
            h_B[i + n*j] = float(j)/float(n);
        }

    cudaMemcpy(d_A, &h_A[0], sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &h_B[0], sizeof(float)*n*n, cudaMemcpyHostToDevice);

    cudaEvent_t t1, t2, t3;
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    cudaEventCreate(&t3);

    cudaEventRecord(t1);
    matmul<<<dim3(n/BS,n/BS,1),dim3(BS,BS,1)>>>(n, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaEventRecord(t2);
    matmul_fast<<<dim3(n/BS,n/BS,1),dim3(BS,BS,1)>>>(n, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaEventRecord(t3);

    cudaEventSynchronize(t3);
    float time_mm, time_mm_fast;
    cudaEventElapsedTime(&time_mm,      t1, t2);
    cudaEventElapsedTime(&time_mm_fast, t2, t3);
    cout << "Time: " << time_mm*1e-3 << " " << time_mm_fast*1e-3 << endl;

    // check the result
    cudaMemcpy(&h_C[0], d_C, sizeof(float)*n*n, cudaMemcpyDeviceToHost);

    for(long j=0;j<n;j++) {
        for(long i=0;i<n;i++) {
            float exact = float(i)*float(j)/float(n);
            if(fabs((h_C[i + n*j] - exact)/exact) > 1e-3) {
                cout << "ERROR: value not correct: " << i << " " << j << " " << h_C[i+n*j] << " vs " << exact << endl;
                exit(1);
            }
        }
    }

    return 0;
}
