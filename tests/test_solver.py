import numpy as np
import pytest
from fdm_engine.config import OptionParameters, GridParameters
from fdm_engine.solvers.tridiagonal import thomas_solve
from fdm_engine.solvers.crank_nicolson import price_european, price_american
from fdm_engine.models.black_scholes import black_scholes_price

def test_thomas_solver():
    """
    Validates the Thomas algorithm against standard matrix inversion using numpy.linalg.solve.
    """
    M = 5
    np.random.seed(42)
    
    # Generate a diagonally dominant tridiagonal system for stability
    b = np.random.uniform(5.0, 10.0, M)
    a = np.random.uniform(1.0, 2.0, M)
    c = np.random.uniform(1.0, 2.0, M)
    d = np.random.uniform(10.0, 20.0, M)
    
    # First and last elements
    a[0] = 0.0
    c[-1] = 0.0
    
    # Construct full matrix for numpy solver
    A = np.zeros((M, M))
    for i in range(M):
        A[i, i] = b[i]
        if i > 0:
            A[i, i-1] = a[i]
        if i < M - 1:
            A[i, i+1] = c[i]
            
    expected_x = np.linalg.solve(A, d)
    actual_x = thomas_solve(a, b, c, d)
    
    np.testing.assert_allclose(actual_x, expected_x, rtol=1e-12, atol=1e-12)

def test_european_pricing_vs_analytical(default_option_params):
    """
    Compares Crank-Nicolson FDM pricing against closed-form Black-Scholes prices
    for European Call and Put options.
    """
    # ATM Option: S0 = 100, K = 100, r = 0.05, sigma = 0.2, T = 1.0
    grid_params = GridParameters(Ns=200, Nt=100, Smax_mult=3.0, align_strike=True)
    
    # 1. European Call (no dividends)
    fdm_call, _, _ = price_european(default_option_params, grid_params, is_call=True, rannacher_steps=4)
    bs_call = black_scholes_price(
        S0=default_option_params.S0,
        K=default_option_params.K,
        r=default_option_params.r,
        q=default_option_params.q,
        sigma=default_option_params.sigma,
        T=default_option_params.T,
        is_call=True
    )
    
    # 2. European Put (no dividends)
    fdm_put, _, _ = price_european(default_option_params, grid_params, is_call=False, rannacher_steps=4)
    bs_put = black_scholes_price(
        S0=default_option_params.S0,
        K=default_option_params.K,
        r=default_option_params.r,
        q=default_option_params.q,
        sigma=default_option_params.sigma,
        T=default_option_params.T,
        is_call=False
    )
    
    # Pricing targets: error < 0.1% of option price or absolute difference < 0.01
    assert abs(fdm_call - bs_call) / bs_call < 0.001
    assert abs(fdm_put - bs_put) / bs_put < 0.001

def test_european_pricing_with_dividends():
    """
    Verifies European pricing correctness when continuous dividend yield q > 0.
    """
    opt_params = OptionParameters(S0=100.0, K=100.0, r=0.05, q=0.03, sigma=0.2, T=1.0)
    grid_params = GridParameters(Ns=200, Nt=100, Smax_mult=3.0, align_strike=True)
    
    # Call with dividends
    fdm_call, _, _ = price_european(opt_params, grid_params, is_call=True, rannacher_steps=4)
    bs_call = black_scholes_price(
        S0=opt_params.S0, K=opt_params.K, r=opt_params.r, q=opt_params.q,
        sigma=opt_params.sigma, T=opt_params.T, is_call=True
    )
    
    # Put with dividends
    fdm_put, _, _ = price_european(opt_params, grid_params, is_call=False, rannacher_steps=4)
    bs_put = black_scholes_price(
        S0=opt_params.S0, K=opt_params.K, r=opt_params.r, q=opt_params.q,
        sigma=opt_params.sigma, T=opt_params.T, is_call=False
    )
    
    assert abs(fdm_call - bs_call) / bs_call < 0.001
    assert abs(fdm_put - bs_put) / bs_put < 0.001

def test_linearity_boundary(default_option_params):
    """
    Verifies that the linearity boundary condition works and matches the Dirichlet condition closely.
    """
    grid_dirichlet = GridParameters(Ns=200, Nt=100, Smax_mult=3.0, align_strike=True, boundary_type="dirichlet")
    grid_linearity = GridParameters(Ns=200, Nt=100, Smax_mult=3.0, align_strike=True, boundary_type="linearity")
    
    price_dir, _, _ = price_european(default_option_params, grid_dirichlet, is_call=True)
    price_lin, _, _ = price_european(default_option_params, grid_linearity, is_call=True)
    
    # Linearity boundary should be extremely close to Dirichlet boundary (error < 0.01%)
    assert abs(price_lin - price_dir) < 1e-4

def test_convergence(default_option_params):
    """
    Verifies convergence: increasing Ns reduces pricing error.
    """
    bs_call = black_scholes_price(
        S0=default_option_params.S0, K=default_option_params.K, r=default_option_params.r,
        q=default_option_params.q, sigma=default_option_params.sigma, T=default_option_params.T, is_call=True
    )
    
    errors = []
    for Ns in [50, 100, 200]:
        grid_params = GridParameters(Ns=Ns, Nt=100, Smax_mult=3.0, align_strike=True)
        fdm_call, _, _ = price_european(default_option_params, grid_params, is_call=True, rannacher_steps=0)
        errors.append(abs(fdm_call - bs_call))
        
    # Check that error strictly decreases
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]

def test_american_pricing_relationships(default_option_params):
    """
    Verifies American option pricing relationships:
    1. American Put >= European Put (strictly > for standard parameter regimes).
    2. American Call == European Call when q = 0 (no early exercise benefit).
    3. American Call > European Call when dividend yield q > 0.
    """
    grid_params = GridParameters(Ns=200, Nt=100, Smax_mult=3.0, align_strike=True)
    
    # 1. American Put vs European Put (no dividends)
    price_eur_put, _, _ = price_european(default_option_params, grid_params, is_call=False)
    price_am_put, _, _ = price_american(default_option_params, grid_params, is_call=False)
    
    assert price_am_put >= price_eur_put
    assert price_am_put - price_eur_put > 0.1  # Significant early exercise premium for ATM Put
    
    # 2. American Call vs European Call (no dividends: q = 0)
    price_eur_call, _, _ = price_european(default_option_params, grid_params, is_call=True)
    price_am_call, _, _ = price_american(default_option_params, grid_params, is_call=True)
    
    # Early exercise is suboptimal for American call on non-dividend paying stock; prices must match
    assert price_am_call == pytest.approx(price_eur_call, abs=1e-5)
    
    # 3. American Call vs European Call (with dividends: q > 0)
    opt_div = OptionParameters(S0=100.0, K=100.0, r=0.05, q=0.08, sigma=0.2, T=1.0)
    price_eur_call_div, _, _ = price_european(opt_div, grid_params, is_call=True)
    price_am_call_div, _, _ = price_american(opt_div, grid_params, is_call=True)
    
    assert price_am_call_div >= price_eur_call_div
    assert price_am_call_div - price_eur_call_div > 0.02  # Early exercise premium exists due to dividends

def test_american_convergence(default_option_params):
    """
    Verifies convergence of the American option pricer: error decreases as grid size increases.
    """
    prices = []
    for Ns in [50, 100, 200]:
        grid_params = GridParameters(Ns=Ns, Nt=100, Smax_mult=3.0, align_strike=True)
        price_am, _, _ = price_american(default_option_params, grid_params, is_call=False, rannacher_steps=4)
        prices.append(price_am)
        
    # Check that differences are decreasing (convergence)
    diff_1 = abs(prices[1] - prices[0])
    diff_2 = abs(prices[2] - prices[1])
    assert diff_2 < diff_1
