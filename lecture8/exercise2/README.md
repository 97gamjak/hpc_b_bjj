#Performance V100 vs A100

V100 naive: 0.483495s fast: 0.548329s
A100 naive: 0.465738s fast: 0.389180s

Looking at the results it can be seen that the performance of the V100 and A100 is approximately the same for the naive implementation, while for the shared memory implementation the V100 gets slower and the A100 faster. This observing results probably due to the larger memory of the A100 (80GB) compared to the V100 (16GB) enabling a faster shared memory access. 