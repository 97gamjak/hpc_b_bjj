using LinearAlgebra

function power_iteration(A, num_iterations)
    n = size(A, 1)
    x = rand(n)  # Initialize a random vector

    for _ = 1:num_iterations
        x = A * x  # Multiply A with x
        x = x / norm(x)  # Normalize x
    end

    return x
end

# Test the function with a random matrix
A = rand(3, 3)
x = power_iteration(A, 1000)
println("The final vector is ", x)