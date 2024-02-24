# Lecture 4 - Exercise 1

## What was the problem?
```bash
========= ERROR SUMMARY: 11 errors
```
9 invalid accesses error (Out of bounds) and 2 kernel launch errors were detected in the program.

## How can we fix it?
We can fix the problem by changing the number of threads from 265 to 256
```c
thrds_per_block.x = 265; // <- should be 256
blcks_per_grid.x = 1;

KrnlDmmy<<<blcks_per_grid, thrds_per_block>>>(a);
```
Additionally, the kernel was adjusted to only access the elements that are within the array bounds.
```c
__global__ void KrnlDmmy(int *x) {
    int i;
    i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i < ARRAYDIM) // <- helps to avoid out-of-bounds access
        x[i] = i; 
    return;
}
```
With 265 theads per block, the kernel was accessing the 256th element of the array, which is out of bounds.