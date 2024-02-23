# Lecture 2 - Exercise 2
## Output with 5992552
```bash
GPU: __cosf(5992552) = 3.826833069324493e-01
GPU: cos(5992552) = -1.411203294992447e-01
CPU: cosf(5992552) = -1.411203444004059e-01
CPU: cos(5992552) = -1.411203444004059e-01
```
## Interpretation
The result for the `__cosf` function on the GPU is different from the result for both `cos` functions and the `cosf` on the CPU. This is due to the input number being too large for the `__cosf` function to handle. The error of this function is small within the -π to π range. However, the error increases as the input number increases. The `__cosf` function is not suitable for large input numbers. The reason for this behavior is due to the fact that the `__cosf` function is a single-precision intrinsic function and compared to the cosf function on the CPU the main difference is that on a GPU, generally there are no additional instructions related to a floating point operations such as there are on i.e. an x86 architecture of a CPU.

### Example with 1
```bash
GPU: __cosf(1) = 5.403023362159729e-01
GPU: cos(1) = 5.403022766113281e-01
CPU: cosf(1) = 5.403022766113281e-01
CPU: cos(1) = 5.403022766113281e-01
```