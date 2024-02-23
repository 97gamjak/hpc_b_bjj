#include <stdio.h>

#define ARRAYDIM 64
#define THREADS_PER_BLOCK 64

__global__ void globFunction(int *arr) {
    __shared__ int local_array[THREADS_PER_BLOCK];  //local block memory cache           
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= ARRAYDIM) {
        return;
    }

    // read values and increase if not already 2
    if(arr[idx] != 2) {
        local_array[threadIdx.x] = arr[idx] + 1;  
    } else {
        local_array[threadIdx.x] = arr[idx];
    }
    __syncthreads(); // <- wait for all threads to finish outside the if statement

    // read the results of neighbor element
    int val = local_array[(threadIdx.x + 1) % THREADS_PER_BLOCK];

    // write back the value to global memory
    arr[idx] = val;

    return;
}

int main() {
    int i, cudaRtrn;
    dim3 thrds_per_block, blcks_per_grid;
    int *a;

    if (cudaRtrn = cudaMallocManaged(&a, ARRAYDIM * sizeof(int)) != 0) {
       printf("*** allocation failed for array a[], %d ***\n", cudaRtrn);
    }

    // first half is 1
    for(i = 0; i < ARRAYDIM/2; ++i) {
        a[i] = 1;
    }

    // second half is 2
    for(i = ARRAYDIM/2; i < ARRAYDIM; ++i) {
        a[i] = 2;
    }

    thrds_per_block.x = THREADS_PER_BLOCK;
    blcks_per_grid.x = 1;

    printf("Running computation...\n");

    globFunction<<<blcks_per_grid, thrds_per_block>>>(a);

    if(cudaRtrn = cudaDeviceSynchronize() != 0) {
        printf("*** error synchronizing device, %d ***\n", cudaRtrn);
    }

    printf("Computation finished!\n");

    // sanity check on the result
    for (i=0; i<ARRAYDIM-1; i++) {
        if(a[i] != 2) {
            printf("ERROR, expected a[%d]=2 but found %d\n", i, a[i]);
        }
    }

    cudaFree(a);

    return(0);
}
