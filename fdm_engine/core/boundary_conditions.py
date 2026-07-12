import numpy as np

def apply_boundary_conditions(
    V: np.ndarray,
    S: np.ndarray,
    K: float,
    r: float,
    t_remaining: float,
    q: float = 0.0,
    is_call: bool = False,
    is_american: bool = False,
    boundary_type: str = "dirichlet"
) -> np.ndarray:
    """
    Applies boundary conditions to the option value array.
    
    Args:
        V: Current option values on the grid
        S: Array of asset prices
        K: Strike price
        r: Risk-free rate
        t_remaining: Time to maturity
        q: Continuous dividend yield
        is_call: True for call, False for put
        is_american: True for American, False for European
        boundary_type: Upper boundary condition type ("dirichlet" or "linearity")
        
    Returns:
        V with boundary conditions applied.
    """
    # Lower Boundary (S -> 0)
    if is_call:
        V[0] = 0.0
    else:
        # For Put: V = K * exp(-r*t) - S * exp(-q*t) for European (generalizing for S[0] > 0)
        # For American Put: V = K - S[0]
        if is_american:
            V[0] = max(K - S[0], 0.0)
        else:
            V[0] = max(K * np.exp(-r * t_remaining) - S[0] * np.exp(-q * t_remaining), 0.0)
            
    # Upper Boundary (S -> Smax)
    if boundary_type == "linearity":
        # Linearity boundary condition (d2V/dS2 = 0)
        V[-1] = 2.0 * V[-2] - V[-3]
    else: # Dirichlet boundary condition
        if is_call:
            # Call: V = S * exp(-q*t) - K * exp(-r*t)
            V[-1] = max(S[-1] * np.exp(-q * t_remaining) - K * np.exp(-r * t_remaining), 0.0)
        else:
            # Put: V -> 0
            V[-1] = 0.0
            
    return V

