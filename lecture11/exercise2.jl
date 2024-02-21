using CUDA
using LinearAlgebra

function power_iteration_kernel(A, x, n)
    i = threadIdx().x
    sum = 0.0
    for j = 1:n
        sum += A[i, j] * x[j]
    end
    x[i] = sum
    return
end

function power_iteration(A, num_iterations)
    n = size(A, 1)

    A_d = CuArray(A)
    x_d = CuArray(rand(n))

    for _ = 1:num_iterations
        @cuda threads = n power_iteration_kernel(A_d, x_d, n)
        x_d = x_d / norm(x_d)
    end

    x = Array(x_d)

    return x
end

A = rand(3, 3)
x = power_iteration(A, 1000)
println("The final vector is ", x)