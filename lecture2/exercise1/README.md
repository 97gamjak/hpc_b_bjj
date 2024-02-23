# Lecture 2 - Exercise 1

```cpp
if(arr[idx] != 2) {
    local_array[threadIdx.x] = arr[idx] + 1;
    __syncthreads();
} else {
    local_array[threadIdx.x] = arr[idx];
    __syncthreads();
}
```

## What was the problem?
Both `__synchthreads()` could be visible for each thread. This could impact the performance of the kernel. Or even worse, it could lead to a deadlock.

## How can we fix it?
We can remove the `__syncthreads()` from the `if` and `else` blocks, and put it after the `if-else` block.

```cpp
if(arr[idx] != 2) {
    local_array[threadIdx.x] = arr[idx] + 1;
} else {
    local_array[threadIdx.x] = arr[idx];
}
__syncthreads();
```



