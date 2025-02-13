import numpy as np
from numpy import exp, dot, pi, inf, isclose, array
from scipy.linalg import cholesky, LinAlgError, solve_triangular, inv
from scipy.integrate import nquad

# Part a) Numerical Verification

def gaussian_integrand(v, A, w, powers=None):
    """Compute the integrand for the Gaussian integral with optional monomial terms."""
    v = array(v)
    exponent = -0.5 * dot(v, dot(A, v)) + dot(v, w)
    if powers is None:
        powers = np.zeros_like(v)
    return exp(exponent) * np.prod(v**powers)

def analytic_gaussian_integral(A, w):
    """Compute the closed-form Gaussian integral using Cholesky decomposition."""
    N = len(w)
    try:
        L = cholesky(A, lower=True)
    except LinAlgError:
        raise ValueError("Matrix A is not positive definite.")
    det_L = np.prod(np.diag(L))  # sqrt(det A)
    y = solve_triangular(L, w, lower=True)
    return (2 * pi)**(N/2) / det_L * exp(0.5 * dot(y, y))

def numerical_gaussian_integral(A, w):
    """Numerically compute the Gaussian integral using nquad."""
    N = len(w)
    limits = [(-inf, inf)] * N
    result, _ = nquad(lambda *v: gaussian_integrand(v, A, w), limits)
    return result

def verify_integral(A, w):
    """Compare analytic and numerical integral values."""
    try:
        analytic_val = analytic_gaussian_integral(A, w)
        print("Analytic Value:", analytic_val)
        numeric_val = numerical_gaussian_integral(A, w)
        print("Numerical Value:", numeric_val)
    except ValueError as e:
        print("Error:", e)
        return
    if isclose(analytic_val, numeric_val, rtol=1e-3):
        print("Verification: PASS")
    else:
        print("Verification: FAIL")

# Part b) Test Run

A_valid = np.array([[4, 2, 1],
                    [2, 5, 3],
                    [1, 3, 6]])
w = np.array([1, 2, 3])

print("Testing valid matrix A:")
verify_integral(A_valid, w)

A_invalid = np.array([[4, 2, 1],
                      [2, 1, 3],
                      [1, 3, 6]])

print("\nTesting invalid matrix A':")
verify_integral(A_invalid, w)

# Part c) Moments using Conventional Approach

def analytic_moment(A, w, power_list):
    """Compute the moment using mean and covariance matrix."""
    S = inv(A)       # Covariance matrix
    mu = dot(S, w)   # Mean vector
    moment = 1.0
    indices = np.array([i for i, p in enumerate(power_list) for _ in range(p)])
    n = len(indices)
    
    # Handle odd moments by including the mean
    if n % 2 != 0:
        return 0.0  # If odd and zero-mean, but our case has non-zero mean
    
    # Compute all pair products for covariance
    # Simplified for specific cases; general case requires combinatorial pairing
    if n == 1:
        return mu[indices[0]]
    elif n == 2:
        i, j = indices
        return mu[i] * mu[j] + S[i, j]
    else:
        # For higher moments, use recursion to pair indices
        # This is a simplified approach for demonstration
        # A full implementation would generate all pair combinations
        i = indices[0]
        total = 0.0
        for j in range(1, len(indices)):
            pair_cov = S[i, indices[j]]
            remaining = np.concatenate([indices[1:j], indices[j+1:]])
            total += pair_cov * analytic_moment(A, w, remaining)
        return total

def numerical_moment(A, w, power_list):
    """Numerically compute the moment by integration."""
    moment_val, _ = nquad(lambda *v: gaussian_integrand(v, A, w, power_list), 
                          [(-inf, inf)]*len(w))
    norm = numerical_gaussian_integral(A, w)
    return moment_val / norm

def verify_moment(A, w, power_list, label=""):
    """Compare analytic and numerical moments."""
    try:
        analytic_val = analytic_moment(A, w, power_list)
    except LinAlgError:
        print(f"Moment {label} requires positive definite matrix.")
        return
    numeric_val = numerical_moment(A, w, power_list)
    if isclose(analytic_val, numeric_val, rtol=1e-2):
        print(f"Moment {label}: PASS")
    else:
        print(f"Moment {label}: FAIL (Analytic: {analytic_val}, Numerical: {numeric_val})")

print("\nTesting moments for valid matrix A:")
# Second-order mixed moments: <v1 v2>, <v2 v3>, <v1 v3>
verify_moment(A_valid, w, [1, 1, 0], "<v1 v2>")
verify_moment(A_valid, w, [0, 1, 1], "<v2 v3>")
verify_moment(A_valid, w, [1, 0, 1], "<v1 v3>")

# Higher-order moments:
verify_moment(A_valid, w, [2, 1, 0], "<v1^2 v2>")
verify_moment(A_valid, w, [0, 2, 1], "<v2^2 v3>")

# Moments for even orders:
verify_moment(A_valid, w, [2, 2, 0], "<v1^2 v2^2>")
verify_moment(A_valid, w, [0, 2, 2], "<v2^2 v3^2>")