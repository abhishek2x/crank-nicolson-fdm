import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Grid:
    """
    Representation of the computational grid in log-space.
    
    Attributes:
        x: Array of spatial points in log-space (ln(S))
        S: Array of spatial points in price-space
        t: Array of time points from 0 to T
        dt: Time step size
        dx: Spatial step size in log-space
    """
    x: np.ndarray
    S: np.ndarray
    t: np.ndarray
    dt: float
    dx: float

def construct_log_grid(
    S0: float,
    K: float,
    T: float,
    Ns: int,
    Nt: int,
    Smax_mult: float = 3.0,
    align_strike: bool = True
) -> Grid:
    """
    Constructs a uniform grid in log-space x = ln(S).
    
    If align_strike is True, adjusts the domain boundaries slightly
    to ensure that ln(K) is exactly a grid node.
    
    Args:
        S0: Current asset price
        K: Strike price
        T: Time to maturity
        Ns: Number of spatial points
        Nt: Number of time steps
        Smax_mult: Multiple of strike to define Smax
        align_strike: If True, aligns the grid such that ln(K) is a node
        
    Returns:
        Grid object containing discretized space and time arrays.
    """
    Smax = Smax_mult * K
    x_max_raw = np.log(Smax)
    x_min_raw = np.log(K / Smax_mult)
    
    if align_strike:
        # Place ln(K) at the middle index
        j = Ns // 2
        # Ideal step size
        dx = (x_max_raw - x_min_raw) / (Ns - 1)
        # Shift boundaries so that x_j = ln(K)
        x_target = np.log(K)
        x = x_target + (np.arange(Ns) - j) * dx
        dx = x[1] - x[0]
        S = np.exp(x)
    else:
        x = np.linspace(x_min_raw, x_max_raw, Ns)
        dx = x[1] - x[0]
        S = np.exp(x)
        
    t = np.linspace(0, T, Nt + 1)
    dt = T / Nt
    
    return Grid(x=x, S=S, t=t, dt=dt, dx=dx)


def get_payoff(S: np.ndarray, K: float, is_call: bool = False) -> np.ndarray:
    """
    Calculates the terminal payoff for an option.
    
    Args:
        S: Array of asset prices
        K: Strike price
        is_call: True for call, False for put
        
    Returns:
        Payoff array
    """
    if is_call:
        return np.maximum(S - K, 0.0)
    else:
        return np.maximum(K - S, 0.0)
