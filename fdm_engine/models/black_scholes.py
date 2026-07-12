import numpy as np
from scipy.stats import norm

def black_scholes_price(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    is_call: bool = True
) -> float:
    """
    Calculates the analytical Black-Scholes price for European options.
    
    Args:
        S0: Current asset price
        K: Strike price
        r: Risk-free rate (annualized)
        q: Continuous dividend yield
        sigma: Volatility (annualized)
        T: Time to maturity (years)
        is_call: True for Call, False for Put
        
    Returns:
        The analytical option price.
    """
    # Handle boundary case where time to maturity is 0
    if T <= 1e-15:
        if is_call:
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)
            
    # Handle boundary case where volatility is 0
    if sigma <= 1e-15:
        pv_K = K * np.exp(-r * T)
        pv_S = S0 * np.exp(-q * T)
        if is_call:
            return max(pv_S - pv_K, 0.0)
        else:
            return max(pv_K - pv_S, 0.0)

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if is_call:
        price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
        
    return float(price)
