using FiniteDiff

# Define the function to compute the gradient of
f(x) = x^2 + sin(x)

# Define a point at which to compute the gradient
x = 2.0

# Compute the gradient using FiniteDiff
fx = f(x)
grad_fd = FiniteDiff.finite_difference_gradient(f, x, fx=fx)

println("FiniteDiff gradient: ", grad_fd)

# using FiniteDiff

# List all the functions defined in the PackageName package
# functions_list = names(FiniteDiff, all=true, imported=false)

# # Print the list of functions
# println(functions_list)