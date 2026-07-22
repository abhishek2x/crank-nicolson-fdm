import numpy as np
import pytest
from fdm_engine.config import OptionParameters, GridParameters
from fdm_engine.solvers.crank_nicolson import price_european, price_american
from fdm_engine.models.black_scholes import black_scholes_price
from fdm_engine.benchmarks.binomial_tree import price_crr_binomial_tree
from fdm_engine.benchmarks.convergence import calculate_convergence_rate

def test_crr_vs_analytical_european(default_option_params):
    """
    Validates CRR Binomial Tree solver against analytical Black-Scholes for European Call and Put.
    """
    bs_call = black_scholes_price(
        S0=default_option_params.S0, K=default_option_params.K, r=default_option_params.r,
        q=default_option_params.q, sigma=default_option_params.sigma, T=default_option_params.T, is_call=True
    )
    crr_call = price_crr_binomial_tree(
        S0=default_option_params.S0, K=default_option_params.K, r=default_option_params.r,
        q=default_option_params.q, sigma=default_option_params.sigma, T=default_option_params.T,
        steps=500, is_call=True, is_american=False
    )
    
    assert abs(crr_call - bs_call) / bs_call < 0.001

def test_fdm_vs_crr_american(default_option_params):
    """
    Compares FDM American Option price against CRR Binomial Tree for American Put.
    """
    grid_params = GridParameters(Ns=300, Nt=150, Smax_mult=3.0, align_strike=True)
    fdm_put, *_ = price_american(default_option_params, grid_params, is_call=False, rannacher_steps=4)
    
    crr_put = price_crr_binomial_tree(
        S0=default_option_params.S0, K=default_option_params.K, r=default_option_params.r,
        q=default_option_params.q, sigma=default_option_params.sigma, T=default_option_params.T,
        steps=1000, is_call=False, is_american=True
    )
    
    # Assert FDM and CRR American Put prices match within $0.02
    assert abs(fdm_put - crr_put) < 0.02

def test_empirical_order_of_convergence(default_option_params):
    """
    Verifies that Crank-Nicolson FDM exhibits ~O(h^2) second-order spatial convergence.
    """
    bs_call = black_scholes_price(
        S0=default_option_params.S0, K=default_option_params.K, r=default_option_params.r,
        q=default_option_params.q, sigma=default_option_params.sigma, T=default_option_params.T, is_call=True
    )
    
    Ns_list = [50, 100, 200, 400]
    errors = []
    spacings = []
    
    for Ns in Ns_list:
        grid_params = GridParameters(Ns=Ns, Nt=Ns, Smax_mult=3.0, align_strike=True)
        fdm_call, grid, *_ = price_european(default_option_params, grid_params, is_call=True, rannacher_steps=4)
        errors.append(abs(fdm_call - bs_call))
        spacings.append(grid.dx)
        
    rate = calculate_convergence_rate(errors, spacings)
    
    # Empirical convergence order should be close to 2.0 (>= 1.8)
    assert rate >= 1.8
