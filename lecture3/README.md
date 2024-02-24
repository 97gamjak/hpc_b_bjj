#Lecture 3
##Original version

array-size = 10⁶
Numerical error: 6.88338e-15
6.20209 s

##Enhanced version

array-size = 10⁶
Numerical error: 6.56142e-14
0.0228248 s

##Performance Analysis

###V100

theoretical peak performance = 7.45 TFLOPS/s
measured original performance = 0.25 TFLOPS/s (n=10⁹)
measured enhanced performance = 0.25 TFLOPS/s (n=10⁹)

###A100

theoretical peak performance = 7.45 TFLOPS/s
measured original performance = 0.25 TFLOPS/s (n=10⁹)
measured enhanced performance = 0.42 TFLOPS/s (n=10⁹)