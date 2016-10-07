import numpy as np

"""

This file contains functions that are very useful during debugging. 
We don't know what the exact gradient is, so many times the backpropagation may be incorrect.
Thus, by using the numerical_gradient_check_* functions we can do testing in a notebook.

The backbone of these functions is the derivative definition using 2*epsilon:

For single variable:
dF/dx = (f(x+epsilon) - f(x-epsilon)) / 2*epsilon

For multivariate:
dF/dx_i = (f([x_0, x_1, ... x_i+epsilon ... x_n]) - f([x_0, x_1, ... x_i-epsilon ... x_n])) / 2*epsilon

Functions:
numerical_gradient_check_scalar(function, input x)

"""

# For pure scalars -> scalars. R -> R
def numerical_gradient_check_scalar(function, x, eps = 1e-5):
    return (function(x+eps) - function(x-eps)) / (2*eps)

# For multivariable -> multivariable. R^n -> R^n, where n can be 1
def numerical_gradient_check_multivar(function, x, eps = 1e-5):
    # Unroll the vector.
    dims = x.shape
    x = x.ravel()
    grad = np.zeros_like(x)
    epsilon_map = np.zeros_like(x, dtype=np.float32)
    for idx, num in enumerate(x):
        epsilon_map[idx] = eps
        deriv = (function((x+epsilon_map).reshape(dims)) - function((x-epsilon_map).reshape(dims))) / (2*eps)
        grad[idx] += np.sum(deriv)
        epsilon_map[idx] = 0
    
    return grad.reshape(dims)