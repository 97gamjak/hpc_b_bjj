/*

This header file contains code to conveniently handle cuda error codes.

> gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
   function to print error message corresponding to cuda return value and optionally abort program

> gpuErrorCheck(ans[, abort])
   macro that calls gpuAssert, to wrap cuda calls that might error, adds source code information
   optionally, abort can be specified, default = true

> usage
gpuErrorCheck(cudaMalloc(...)); 	      // if fails, print message and abort
gpuErrorCheck(cudaMalloc(...), true);         // if fails, print message and abort
gpuErrorCheck(cudaMalloc(...), false);        // if fails, print message and continue

*/

//#define gpuErrorCheck(ans, abort) \
//    { gpuAssert((ans), __FILE__, __LINE__, (abort)); }
#define gpuErrorCheck(...) \
    { gpuAssert(__VA_ARGS__, __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, bool abort, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA ASSERT: \"%s\"\n\t> %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    return gpuAssert(code, true, file, line);
}
