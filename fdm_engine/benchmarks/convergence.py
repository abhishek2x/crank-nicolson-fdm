import numpy as np
from typing import List, Tuple

def calculate_convergence_rate(errors: List[float], grid_spacings: List[float]) -> float:
    """
    Calculates the empirical order of convergence p from a series of errors
    and grid spacings h via log-log linear regression: log(Error) = p * log(h) + C.
    
    Args:
        errors: List of absolute pricing errors at different resolutions.
        grid_spacings: List of grid step sizes (e.g., dx or 1/Ns).
        
    Returns:
        Empirical convergence order p (float).
    """
    log_h = np.log(np.array(grid_spacings))
    log_err = np.log(np.array(errors))
    
    # Linear fit: log(Error) = slope * log(h) + intercept
    slope, _ = np.polyfit(log_h, log_err, 1)
    return float(slope)
