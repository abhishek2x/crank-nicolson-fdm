import numpy as np

def price_crr_binomial_tree(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    steps: int = 500,
    is_call: bool = True,
    is_american: bool = True
) -> float:
    """
    Prices an option using the Cox-Ross-Rubinstein (CRR) Binomial Tree model.
    Supports continuous dividend yield q and American early exercise.
    
    Args:
        S0: Current asset price
        K: Strike price
        r: Risk-free interest rate (annualized)
        q: Continuous dividend yield (annualized)
        sigma: Volatility (annualized)
        T: Time to maturity (years)
        steps: Number of time steps in the binomial tree
        is_call: True for Call, False for Put
        is_american: True for American option, False for European option
        
    Returns:
        Option price at t=0 (float).
    """
    if steps < 1:
        raise ValueError("Number of steps must be at least 1.")
        
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    
    # Risk-neutral probability incorporating continuous dividend yield q
    p = (np.exp((r - q) * dt) - d) / (u - d)
    df = np.exp(-r * dt)  # Discount factor per step
    
    # Stock prices at maturity (step N)
    j = np.arange(steps + 1)
    S_T = S0 * (u ** j) * (d ** (steps - j))
    
    # Terminal payoffs
    if is_call:
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)
        
    # Backward induction through the tree
    for i in range(steps - 1, -1, -1):
        # Discounted expected value
        V = df * (p * V[1:] + (1.0 - p) * V[:-1])
        
        if is_american:
            # Underlying stock prices at step i
            j_i = np.arange(i + 1)
            S_i = S0 * (u ** j_i) * (d ** (i - j_i))
            
            # Intrinsic value
            if is_call:
                intrinsic = np.maximum(S_i - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_i, 0.0)
                
            # Early exercise constraint
            V = np.maximum(V, intrinsic)
            
    return float(V[0])
