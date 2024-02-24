# Lecture 4 - Exercise 3

## Problem
```c
int size = N * sizeof(float);
int wsize = (2 * RADIUS + 1) * sizeof(float);

...

float *d_weights; cudaMalloc(&d_weights, wsize);
float *d_in; cudaMalloc(&d_in, wsize);
float *d_out; cudaMalloc(&d_out, wsize);

cudaMemcpy(d_weights, weights, wsize, cudaMemcpyHostToDevice);
cudaMemcpy(d_in, in, wsize, cudaMemcpyHostToDevice);

...

cudaMemcpy(out, d_out, wsize, cudaMemcpyDeviceToHost);
```
Input and output arrays are allocated with the wrong size. The input and output arrays should be allocated with the size `N * sizeof(float)`, while the weights array should be allocated with the size `(2 * RADIUS + 1) * sizeof(float)`.
Therefore `wsize` should be interchanged with `size` in the `cudaMalloc` and `cudaMemcpy` calls.

## Solution
```c
int size = N * sizeof(float);
int wsize = (2 * RADIUS + 1) * sizeof(float);

...

float *d_weights; cudaMalloc(&d_weights, wsize);
float *d_in; cudaMalloc(&d_in, size);
float *d_out; cudaMalloc(&d_out, size);

cudaMemcpy(d_weights, weights, wsize, cudaMemcpyHostToDevice);
cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

...

cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
```