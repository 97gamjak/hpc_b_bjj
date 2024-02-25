# Lecture 3
## Original version

```bash
array-size = 10⁶
Numerical error: 6.88338e-15
6.20209 s
```

# Enhanced version

```bash
array-size = 10⁶
Numerical error: 6.56142e-14
0.0228248 s
```

## Performance Analysis

**NOTE** we assumed 5 flops per kernel execution

### V100
```bash
theoretical peak performance = 7.45 TFLOPS/s
measured original performance = 0.25 TFLOPS/s (n=10⁹)
measured enhanced performance = 0.25 TFLOPS/s (n=10⁹)
```
### A100
```bash
theoretical peak performance = 7.45 TFLOPS/s
measured original performance = 0.25 TFLOPS/s (n=10⁹)
measured enhanced performance = 0.42 TFLOPS/s (n=10⁹)
```