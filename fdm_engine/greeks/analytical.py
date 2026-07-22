import numpy as np
from scipy.stats import norm
from dataclasses import dataclass

@dataclass(frozen=True)
class Greeks:
    """Dataclass holding option risk sensitivities (Greeks)."""
    delta: float
    gamma: float
    theta: float

def black_scholes_greeks(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    is_call: bool = True
) -> Greeks:
    """
    Calculates exact analytical Black-Scholes Greeks (Delta, Gamma, Theta) for European options.
    
    Args:
        S0: Current asset price
        K: Strike price
        r: Risk-free rate (annualized)
        q: Continuous dividend yield
        sigma: Volatility (annualized)
        T: Time to maturity (years)
        is_call: True for Call, False for Put
        
    Returns:
        Greeks object containing delta, gamma, and theta.
    """
    if T <= 1e-15 or sigma <= 1e-15:
        delta = (1.0 if S0 > K else 0.0) if is_call else (-1.0 if S0 < K else 0.0)
        return Greeks(delta=delta, gamma=0.0, theta=0.0)

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if is_call:
        delta = float(np.exp(-q * T) * norm.cdf(d1))
    else:
        delta = float(-np.exp(-q * T) * norm.cdf(-d1))
        
    # Gamma (same for Call and Put)
    gamma = float(np.exp(-q * T) * norm.pdf(d1) / (S0 * sigma * np.sqrt(T)))
    
    # Theta (per year)
    term1 = -(S0 * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2.0 * np.sqrt(T))
    if is_call:
        term2 = q * S0 * np.exp(-q * T) * norm.cdf(d1)
        term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = float(term1 + term2 + term3)
    else:
        term2 = -q * S0 * np.exp(-q * T) * norm.cdf(-d1)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = float(term1 + term2 + term3)
        
    return Greeks(delta=delta, gamma=gamma, theta=theta)
