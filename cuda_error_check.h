/*

This header file contains code to conveniently handle cuda error codes.

> gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
   function to print error message corresponding to cuda return value and optionally abort program

> gpuErrorCheck(ans, abort)
   macro that calls gpuAssert, to wrap cuda calls that might error, adds source code information

> usage
gpuErrorCheck(cudaMalloc(...));        // if fails, print message and continue
gpuErrorCheck(cudaMalloc(...), true);  // if fails, print message and abort

*/

#define gpuErrorCheck(ans, abort) \
    { gpuAssert((ans), __FILE__, __LINE__, (abort)); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "assert:%s%s%d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}