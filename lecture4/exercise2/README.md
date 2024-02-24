# Lecture 4 - Exercise 2

## Result 
```bash
cuda-memcheck --tool synccheck syncthreads
```
```bash
========= ERROR SUMMARY: 32 errors
```
32 invalid __syncthreads() calls were detected in the program (`Barrier error detected. Divergent thread(s) in block`).

## Solution
From:
```c
    // read values and increase if not already 2
    if(arr[idx] != 2) {
        local_array[threadIdx.x] = arr[idx] + 1;  
        __syncthreads();
    } else {
        local_array[threadIdx.x] = arr[idx];
        __syncthreads();
    }
```
To:
```c
    // read values and increase if not already 2
    if(arr[idx] != 2) {
        local_array[threadIdx.x] = arr[idx] + 1;  
    } else {
        local_array[threadIdx.x] = arr[idx];
    }
    __syncthreads();
```
This change ensures that all threads in the block reach the __syncthreads() call, avoiding the barrier error.