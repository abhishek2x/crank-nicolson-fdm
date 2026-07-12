import numpy as np

def thomas_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray
) -> np.ndarray:
    """
    Solves a tridiagonal system Ax = d using the Thomas algorithm.
    
    The system is represented by its diagonal bands:
    a: Sub-diagonal array of length M (a[0] is ignored)
    b: Diagonal array of length M
    c: Super-diagonal array of length M (c[-1] is ignored)
    d: Right-hand side array of length M
    
    Returns:
        Solution array x of length M.
    """
    M = len(b)
    c_prime = np.zeros(M, dtype=float)
    d_prime = np.zeros(M, dtype=float)
    x = np.zeros(M, dtype=float)
    
    # Forward sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, M):
        denom = b[i] - a[i] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
        
    # Backward substitution
    x[-1] = d_prime[-1]
    for i in range(M - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x
