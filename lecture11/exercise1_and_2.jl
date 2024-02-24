using LinearAlgebra
using CUDA

function power_iteration_cpu(A, num_iterations)
    n = size(A, 1)
    x = rand(n)

    for _ = 1:num_iterations
        x = A * x  # Multiply A with x
        x = x / norm(x)  # Normalize x
    end

    return x
end

function square_kernel(x, y)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for i = index:stride:length(x)
        CUDA.@atomic y[1] += x[i]^2
    end
    return
end

function normalize_kernel(x, norm)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for i = index:stride:length(x)
        x[i] = x[i] / norm
    end
    return
end

function power_iteration_gpu(A, num_iterations)
    n = size(A, 1)

    A_d = CuArray(A)
    x_d = CUDA.rand(n)

    for _ = 1:num_iterations
        x_d = A_d * x_d
        x_d = x_d / CUDA.norm(x_d)
    end

    return Array(x_d)
end

function power_iteration_gpu2(A, num_iterations)
    n = size(A, 1)

    A_d = CuArray(A)
    x_d = CUDA.rand(n)

    threads = 256
    numblocks = ceil(Int, n / threads)

    for _ = 1:num_iterations
        x_d = A_d * x_d
        sum_d = CuArray([0.0])
        @cuda threads = threads blocks = numblocks square_kernel(x_d, sum_d)
        norm = sqrt(sum_d[1])
        @cuda threads = threads blocks = numblocks normalize_kernel(x_d, norm)
    end

    return Array(x_d)
end

function bench_gpu(A, num_iterations)
    @benchmark power_iteration_gpu($A, $num_iterations)
end

function bench_gpu2(A, num_iterations)
    @benchmark power_iteration_gpu2($A, $num_iterations)
end

function bench_cpu(A, num_iterations)
    @benchmark power_iteration_cpu($A, $num_iterations)
end

n = 1 << 12
A = rand(n, n)
x_cpu = power_iteration_cpu(A, 1000)
x_gpu = power_iteration_gpu(A, 1000)
x_gpu2 = power_iteration_gpu2(A, 1000)

println("The final cpu vector is ", x_cpu)
println("The final gpu vector is ", x_gpu)
println("The final gpu2 vector is ", x_gpu2)

@assert isapprox(x_cpu, x_gpu, atol=1e-6)
@assert isapprox(x_cpu, x_gpu2, atol=1e-6)
benchmark = eigvals(A)[end]
if typeof(benchmark) == Complex
    benchmark = real(benchmark)
end
@assert isapprox(norm(A * x_cpu), eigvals(A)[end], atol=1e-6)
@assert isapprox(norm(A * x_gpu), eigvals(A)[end], atol=1e-6)
@assert isapprox(norm(A * x_gpu2), eigvals(A)[end], atol=1e-6)
