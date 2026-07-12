import numpy as np
from typing import Tuple

def psor_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    g: np.ndarray,
    V_guess: np.ndarray,
    omega: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[np.ndarray, int]:
    """
    Solves a tridiagonal system under early exercise constraints (LCP) 
    using Projected Successive Over-Relaxation (PSOR).
    
    A V = d, subject to V >= g.
    
    Args:
        a: Sub-diagonal array of length M (a[0] is ignored)
        b: Diagonal array of length M
        c: Super-diagonal array of length M (c[-1] is ignored)
        d: Right-hand side array of length M
        g: Payoff constraint array of length M (intrinsic value)
        V_guess: Initial guess array of length M (for warm start)
        omega: Relaxation parameter, 1.0 <= omega < 2.0
        tol: Convergence tolerance (maximum absolute difference)
        max_iter: Maximum number of iterations allowed
        
    Returns:
        Tuple containing:
            - np.ndarray: The solved option values of length M
            - int: The number of iterations taken to converge
    """
    M = len(b)
    V = V_guess.copy()
    
    for iteration in range(1, max_iter + 1):
        V_old = V.copy()
        
        # Sequentially update each element of the grid node-by-node
        for i in range(M):
            # Contribution of the left neighbor (uses new value V[i-1])
            sum_left = a[i] * V[i-1] if i > 0 else 0.0
            
            # Contribution of the right neighbor (uses old value V_old[i+1])
            sum_right = c[i] * V_old[i+1] if i < M - 1 else 0.0
            
            # Standard Gauss-Seidel continuation value
            V_gs = (d[i] - sum_left - sum_right) / b[i]
            
            # Successive Over-Relaxation adjustment
            V_relaxed = (1.0 - omega) * V_old[i] + omega * V_gs
            
            # Projection step: V cannot fall below the intrinsic value g
            V[i] = max(V_relaxed, g[i])
            
        # Check convergence: maximum change across the entire vector
        diff = np.max(np.abs(V - V_old))
        if diff < tol:
            return V, iteration
            
    return V, max_iter
