import numpy as np
from numpy import exp, dot, pi, inf, isclose, array
from scipy.linalg import cholesky, LinAlgError, solve_triangular, inv
from scipy.integrate import nquad

# a) Numerical Verification

def gaussian_integrand(v, A, w, powers=None):
    """compute the integrand for the gaussian integral with optional monomial terms."""
    v = array(v)  # convert input to numpy array
    exponent = -0.5 * dot(v, dot(A, v)) + dot(v, w)  # compute exponent of gaussian function
    if powers is None:
        powers = np.zeros_like(v)  # default power terms are zero (i.e., standard gaussian integral)
    return exp(exponent) * np.prod(v**powers)  # return evaluated integrand

def analytic_gaussian_integral(A, w):
    """compute the closed-form gaussian integral using cholesky decomposition."""
    N = len(w)  # determine dimensionality of the integral
    try:
        L = cholesky(A, lower=True)  # compute cholesky factorization (A = L L^T)
    except LinAlgError:
        raise ValueError("matrix A' is not positive definite.")  # handle non-positive definite matrices
    det_L = np.prod(np.diag(L))  # compute determinant of L (sqrt(det A))
    y = solve_triangular(L, w, lower=True)  # solve L y = w
    return (2 * pi)**(N/2) / det_L * exp(0.5 * dot(y, y))  # return closed-form integral result

def numerical_gaussian_integral(A, w):
    """numerically compute the gaussian integral using nquad."""
    N = len(w)  # determine dimensionality of the integral
    limits = [(-inf, inf)] * N  # set integration limits for each variable
    result, _ = nquad(lambda *v: gaussian_integrand(v, A, w), limits)  # perform numerical integration
    return result

def verify_integral(A, w):
    """compare analytic and numerical integral values with detailed output."""
    try:
        analytic_val = analytic_gaussian_integral(A, w)  # compute analytic integral
        numeric_val = numerical_gaussian_integral(A, w)  # compute numerical integral
        print("\nAnalytic Value:", analytic_val)
        print("Numerical Value:", numeric_val)
        diff = abs(analytic_val - numeric_val)  # compute absolute difference
        print(f"Absolute Difference: {diff:.6e}")
        if isclose(analytic_val, numeric_val, rtol=1e-3):  # check if values are close within tolerance
            print("Verification: PASS")
        else:
            print("Verification: FAIL")
    except ValueError as e:
        print("Error:", e)  # handle errors

# b) Test run

# define a valid positive definite matrix A
A_valid = np.array([[4, 2, 1],
                    [2, 5, 3],
                    [1, 3, 6]])
w = np.array([1, 2, 3])  # define vector w

print("Testing valid matrix A:")
verify_integral(A_valid, w)  # run verification for valid A

# define an invalid (non-positive definite) matrix A'
A_invalid = np.array([[4, 2, 1],
                      [2, 1, 3],
                      [1, 3, 6]])

print("\nTesting invalid matrix A':")
verify_integral(A_invalid, w)  # run verification for invalid A

# c) Correlation function and Moments of a multivariate normal distribution

def analytic_moment(A, w, power_list):
    """compute the moment using mean and covariance matrix."""
    S = inv(A)  # compute covariance matrix (inverse of A)
    mu = dot(S, w)  # compute mean vector
    indices = np.array([i for i, p in enumerate(power_list) for _ in range(p)])  # generate index list based on powers
    n = len(indices)  # determine number of variables in moment
    
    if n % 2 != 0:
        return 0.0  # return zero for odd moments (simplified assumption)
    
    if n == 1:
        return mu[indices[0]]  # return mean for single variable moment
    elif n == 2:
        i, j = indices
        return mu[i] * mu[j] + S[i, j]  # use expectation formula for second-order moments
    else:
        i = indices[0]  # pick first index
        total = 0.0
        for j in range(1, len(indices)):
            pair_cov = S[i, indices[j]]  # extract covariance term
            remaining = np.concatenate([indices[1:j], indices[j+1:]])  # generate remaining indices
            total += pair_cov * analytic_moment(A, w, remaining)  # compute recursive moment sum
        return total

def numerical_moment(A, w, power_list):
    """numerically compute the moment by integration."""
    moment_val, _ = nquad(lambda *v: gaussian_integrand(v, A, w, power_list), 
                          [(-inf, inf)]*len(w))  # integrate to compute moment
    norm = numerical_gaussian_integral(A, w)  # compute normalization constant
    return moment_val / norm  # return normalized moment value

def verify_moment(A, w, power_list, label=""):
    """compare analytic and numerical moments with detailed output."""
    try:
        analytic_val = analytic_moment(A, w, power_list)  # compute analytic moment
        numeric_val = numerical_moment(A, w, power_list)  # compute numerical moment
        print(f"\nMoment {label}:")
        print(f"  Analytic: {analytic_val:.6f}")
        print(f"  Numerical: {numeric_val:.6f}")
        diff = analytic_val - numeric_val  # compute difference
        print(f"  Difference: {diff:.6e}")
        if isclose(analytic_val, numeric_val, rtol=1e-2):  # check if values are close
            print("  Verification: PASS")
        else:
            print("  Verification: FAIL")
    except LinAlgError:
        print(f"Moment {label} requires positive definite matrix.")  # handle errors

print("\nTesting moments for valid matrix A:")
verify_moment(A_valid, w, [1, 1, 0], "<v1 v2>")
verify_moment(A_valid, w, [0, 1, 1], "<v2 v3>")
verify_moment(A_valid, w, [1, 0, 1], "<v1 v3>")
verify_moment(A_valid, w, [2, 1, 0], "<v1^2 v2>")
verify_moment(A_valid, w, [0, 2, 1], "<v2^2 v3>")
verify_moment(A_valid, w, [2, 2, 0], "<v1^2 v2^2>")
verify_moment(A_valid, w, [0, 2, 2], "<v2^2 v3^2>")

# Results:


# Testing valid matrix A:

# Analytic Value: 4.2758236590115155
# Numerical Value: 4.275823659021463
# Absolute Difference: 9.947598e-12
# Verification: PASS

# Testing invalid matrix A':
# Error: matrix A' is not positive definite.

# Testing moments for valid matrix A:

# Moment <v1 v2>:
#   Analytic: -0.124972
#   Numerical: -0.124972
#   Difference: -8.225642e-13
#   Verification: PASS

# Moment <v2 v3>:
#   Analytic: -0.104032
#   Numerical: -0.104032
#   Difference: 4.642592e-12
#   Verification: PASS

# Moment <v1 v3>:
#   Analytic: 0.053687
#   Numerical: 0.053687
#   Difference: -2.001371e-12
#   Verification: PASS

# Moment <v1^2 v2>:
#   Analytic: 0.000000
#   Numerical: 0.009526
#   Difference: -9.525773e-03
#   Verification: FAIL

# Moment <v2^2 v3>:
#   Analytic: 0.000000
#   Numerical: 0.122123
#   Difference: -1.221227e-01
#   Verification: FAIL

# Moment <v1^2 v2^2>:
#   Analytic: -0.039170
#   Numerical: 0.144919
#   Difference: -1.840896e-01
#   Verification: FAIL

# Moment <v2^2 v3^2>:
#   Analytic: -0.013447
#   Numerical: 0.168498
#   Difference: -1.819449e-01
#   Verification: FAIL