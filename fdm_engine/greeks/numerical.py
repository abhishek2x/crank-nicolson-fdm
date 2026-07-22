import numpy as np
from scipy.interpolate import interp1d
from fdm_engine.core.grid import Grid
from fdm_engine.greeks.analytical import Greeks

def calculate_fdm_greeks(
    grid: Grid,
    V_t0: np.ndarray,
    V_t1: np.ndarray,
    S0: float
) -> Greeks:
    """
    Extracts option Greeks (Delta, Gamma, Theta) from the finite difference grid.
    Uses log-space central finite differences for spatial derivatives (Delta, Gamma)
    and time difference for Theta.
    
    Args:
        grid: Grid object containing discretized space (x, S) and time (dt, dx)
        V_t0: Option price array at t=0 (time to maturity T, length Ns)
        V_t1: Option price array at t=dt (time to maturity T - dt, length Ns)
        S0: Target asset price for interpolation
        
    Returns:
        Greeks dataclass containing (delta, gamma, theta) at S0.
    """
    dx = grid.dx
    dt = grid.dt
    S_int = grid.S[1:-1]
    
    # 1. Delta: dV/dS = (1/S) * (dV/dx)
    # Central difference in uniform log-space x
    dV_dx = (V_t0[2:] - V_t0[:-2]) / (2.0 * dx)
    delta_grid = dV_dx / S_int
    
    # 2. Gamma: d2V/dS2 = (1/S^2) * (d2V/dx2 - dV/dx)
    d2V_dx2 = (V_t0[2:] - 2.0 * V_t0[1:-1] + V_t0[:-2]) / (dx ** 2)
    gamma_grid = (d2V_dx2 - dV_dx) / (S_int ** 2)
    
    # 3. Theta: dV/dt = (V(t=dt) - V(t=0)) / dt
    # Negative for long options (time decay loss as calendar time advances)
    theta_grid = (V_t1[1:-1] - V_t0[1:-1]) / dt
    
    # Interpolate each Greek at S0 using cubic spline
    interp_delta = interp1d(S_int, delta_grid, kind="cubic", fill_value="extrapolate")
    interp_gamma = interp1d(S_int, gamma_grid, kind="cubic", fill_value="extrapolate")
    interp_theta = interp1d(S_int, theta_grid, kind="cubic", fill_value="extrapolate")
    
    delta = float(interp_delta(S0))
    gamma = float(interp_gamma(S0))
    theta = float(interp_theta(S0))
    
    return Greeks(delta=delta, gamma=gamma, theta=theta)
