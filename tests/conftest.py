import pytest
from fdm_engine.config import OptionParameters, GridParameters

@pytest.fixture
def default_option_params():
    """Standard ATM option parameters."""
    return OptionParameters(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0
    )

@pytest.fixture
def default_grid_params():
    """Standard grid parameters."""
    return GridParameters(
        Ns=100,
        Nt=50
    )
