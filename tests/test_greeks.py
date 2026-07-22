import numpy as np
import pytest
from fdm_engine.config import OptionParameters, GridParameters
from fdm_engine.solvers.crank_nicolson import price_european, price_american
from fdm_engine.greeks.analytical import black_scholes_greeks
from fdm_engine.greeks.numerical import calculate_fdm_greeks

def test_european_call_greeks(default_option_params):
    """
    Compares FDM numerical Greeks against analytical Black-Scholes Greeks for European Call.
    """
    grid_params = GridParameters(Ns=200, Nt=100, Smax_mult=3.0, align_strike=True)
    
    price, grid, V_t0, V_t1 = price_european(default_option_params, grid_params, is_call=True)
    
    fdm_greeks = calculate_fdm_greeks(grid, V_t0, V_t1, default_option_params.S0)
    bs_greeks = black_scholes_greeks(
        S0=default_option_params.S0, K=default_option_params.K, r=default_option_params.r,
        q=default_option_params.q, sigma=default_option_params.sigma, T=default_option_params.T, is_call=True
    )
    
    # Delta comparison (error < 0.5%)
    assert abs(fdm_greeks.delta - bs_greeks.delta) / bs_greeks.delta < 0.005
    # Gamma comparison (error < 1%)
    assert abs(fdm_greeks.gamma - bs_greeks.gamma) / bs_greeks.gamma < 0.01
    # Theta comparison (error < 1%)
    assert abs(fdm_greeks.theta - bs_greeks.theta) / abs(bs_greeks.theta) < 0.01

def test_european_put_greeks(default_option_params):
    """
    Compares FDM numerical Greeks against analytical Black-Scholes Greeks for European Put.
    """
    grid_params = GridParameters(Ns=200, Nt=100, Smax_mult=3.0, align_strike=True)
    
    price, grid, V_t0, V_t1 = price_european(default_option_params, grid_params, is_call=False)
    
    fdm_greeks = calculate_fdm_greeks(grid, V_t0, V_t1, default_option_params.S0)
    bs_greeks = black_scholes_greeks(
        S0=default_option_params.S0, K=default_option_params.K, r=default_option_params.r,
        q=default_option_params.q, sigma=default_option_params.sigma, T=default_option_params.T, is_call=False
    )
    
    assert abs(fdm_greeks.delta - bs_greeks.delta) / abs(bs_greeks.delta) < 0.005
    assert abs(fdm_greeks.gamma - bs_greeks.gamma) / bs_greeks.gamma < 0.01
    assert abs(fdm_greeks.theta - bs_greeks.theta) / abs(bs_greeks.theta) < 0.01

def test_greeks_with_dividends():
    """
    Verifies Greeks extraction for dividend-paying European options.
    """
    opt_div = OptionParameters(S0=100.0, K=100.0, r=0.05, q=0.03, sigma=0.2, T=1.0)
    grid_params = GridParameters(Ns=200, Nt=100, Smax_mult=3.0, align_strike=True)
    
    # Call
    price_call, grid_c, V_c0, V_c1 = price_european(opt_div, grid_params, is_call=True)
    fdm_call_greeks = calculate_fdm_greeks(grid_c, V_c0, V_c1, opt_div.S0)
    bs_call_greeks = black_scholes_greeks(
        S0=opt_div.S0, K=opt_div.K, r=opt_div.r, q=opt_div.q, sigma=opt_div.sigma, T=opt_div.T, is_call=True
    )
    assert abs(fdm_call_greeks.delta - bs_call_greeks.delta) < 0.005
    assert abs(fdm_call_greeks.gamma - bs_call_greeks.gamma) < 0.005
    
    # Put
    price_put, grid_p, V_p0, V_p1 = price_european(opt_div, grid_params, is_call=False)
    fdm_put_greeks = calculate_fdm_greeks(grid_p, V_p0, V_p1, opt_div.S0)
    bs_put_greeks = black_scholes_greeks(
        S0=opt_div.S0, K=opt_div.K, r=opt_div.r, q=opt_div.q, sigma=opt_div.sigma, T=opt_div.T, is_call=False
    )
    assert abs(fdm_put_greeks.delta - bs_put_greeks.delta) < 0.005

def test_american_greeks_bounds(default_option_params):
    """
    Sanity checks for American option numerical Greeks:
    - Delta in [0, 1] for Call, [-1, 0] for Put.
    - Gamma > 0 for both long Call and Put.
    """
    grid_params = GridParameters(Ns=200, Nt=100, Smax_mult=3.0, align_strike=True)
    
    # American Put
    price_put, grid_p, V_p0, V_p1 = price_american(default_option_params, grid_params, is_call=False)
    put_greeks = calculate_fdm_greeks(grid_p, V_p0, V_p1, default_option_params.S0)
    
    assert -1.0 <= put_greeks.delta <= 0.0
    assert put_greeks.gamma > 0.0
    
    # American Call
    price_call, grid_c, V_c0, V_c1 = price_american(default_option_params, grid_params, is_call=True)
    call_greeks = calculate_fdm_greeks(grid_c, V_c0, V_c1, default_option_params.S0)
    
    assert 0.0 <= call_greeks.delta <= 1.0
    assert call_greeks.gamma > 0.0
