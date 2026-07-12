import logging
import sys
from dataclasses import dataclass

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("fdm_engine")

@dataclass(frozen=True)
class OptionParameters:
    """Standard parameters for option pricing."""
    S0: float = 100.0        # Current asset price
    K: float = 100.0         # Strike price
    r: float = 0.05          # Risk-free rate (annualized)
    q: float = 0.0           # Continuous dividend yield
    sigma: float = 0.2       # Volatility (annualized)
    T: float = 1.0           # Time to maturity (years)

@dataclass(frozen=True)
class GridParameters:
    """Parameters for the finite difference grid."""
    Ns: int = 200            # Number of spatial points
    Nt: int = 100            # Number of time steps
    Smax_mult: float = 3.0   # Multiple of K for Smax
    align_strike: bool = True # Force a grid node at the strike price K
    boundary_type: str = "dirichlet" # Boundary condition type at Smax: "dirichlet" or "linearity"
