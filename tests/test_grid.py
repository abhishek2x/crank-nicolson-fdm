import numpy as np
import pytest
from fdm_engine.core.grid import construct_log_grid, get_payoff
from fdm_engine.core.boundary_conditions import apply_boundary_conditions

def test_grid_construction(default_option_params, default_grid_params):
    grid = construct_log_grid(
        S0=default_option_params.S0,
        K=default_option_params.K,
        T=default_option_params.T,
        Ns=default_grid_params.Ns,
        Nt=default_grid_params.Nt,
        Smax_mult=default_grid_params.Smax_mult,
        align_strike=False
    )
    
    assert len(grid.x) == default_grid_params.Ns
    assert len(grid.S) == default_grid_params.Ns
    assert len(grid.t) == default_grid_params.Nt + 1
    
    # Check log-space relationship
    np.testing.assert_allclose(np.exp(grid.x), grid.S)
    
    # Check boundaries
    assert grid.S[-1] == pytest.approx(default_option_params.K * default_grid_params.Smax_mult)
    assert grid.S[0] == pytest.approx(default_option_params.K / default_grid_params.Smax_mult)

def test_grid_strike_alignment(default_option_params, default_grid_params):
    # Test log-grid strike alignment (both even and odd Ns)
    for Ns in [100, 101]:
        grid = construct_log_grid(
            S0=default_option_params.S0,
            K=default_option_params.K,
            T=default_option_params.T,
            Ns=Ns,
            Nt=default_grid_params.Nt,
            Smax_mult=default_grid_params.Smax_mult,
            align_strike=True
        )
        # Check that ln(K) is exactly a node in grid.x
        x_target = np.log(default_option_params.K)
        assert np.any(np.isclose(grid.x, x_target, rtol=1e-15, atol=1e-15))
        # Consequently, K should be exactly in grid.S
        assert np.any(np.isclose(grid.S, default_option_params.K, rtol=1e-15, atol=1e-15))

def test_payoff_calculation():
    S = np.array([80, 100, 120])
    K = 100.0
    
    put_payoff = get_payoff(S, K, is_call=False)
    call_payoff = get_payoff(S, K, is_call=True)
    
    np.testing.assert_array_equal(put_payoff, [20, 0, 0])
    np.testing.assert_array_equal(call_payoff, [0, 0, 20])

def test_boundary_conditions(default_option_params):
    S = np.array([0.0, 100.0, 300.0])
    V = np.zeros_like(S)
    K = default_option_params.K
    r = default_option_params.r
    T = default_option_params.T
    
    # European Put (q = 0)
    V_put = apply_boundary_conditions(V.copy(), S, K, r, T, is_call=False, is_american=False)
    assert V_put[0] == pytest.approx(K * np.exp(-r * T))
    assert V_put[-1] == 0.0
    
    # American Put (q = 0)
    V_put_am = apply_boundary_conditions(V.copy(), S, K, r, T, is_call=False, is_american=True)
    assert V_put_am[0] == K
    assert V_put_am[-1] == 0.0
    
    # European Call (q = 0)
    V_call = apply_boundary_conditions(V.copy(), S, K, r, T, is_call=True, is_american=False)
    assert V_call[0] == 0.0
    assert V_call[-1] == pytest.approx(S[-1] - K * np.exp(-r * T))

    # European Call with dividend yield q > 0
    q = 0.03
    V_call_div = apply_boundary_conditions(V.copy(), S, K, r, T, q=q, is_call=True, is_american=False)
    assert V_call_div[0] == 0.0
    assert V_call_div[-1] == pytest.approx(S[-1] * np.exp(-q * T) - K * np.exp(-r * T))

    # European Put with dividend yield q > 0 (and S[0] > 0 as in log-grid)
    S_log = np.array([30.0, 100.0, 300.0])
    V_log = np.zeros_like(S_log)
    V_put_div = apply_boundary_conditions(V_log.copy(), S_log, K, r, T, q=q, is_call=False, is_american=False)
    assert V_put_div[0] == pytest.approx(max(K * np.exp(-r * T) - S_log[0] * np.exp(-q * T), 0.0))
    assert V_put_div[-1] == 0.0

    # Linearity boundary condition for Call
    V_init = np.array([0.0, 50.0, 120.0])  # V[-2] = 50.0, V[-3] = 0.0
    V_lin = apply_boundary_conditions(V_init.copy(), S, K, r, T, is_call=True, boundary_type="linearity")
    # Expected V[-1] = 2 * V[-2] - V[-3] = 2 * 50 - 0 = 100
    assert V_lin[-1] == 100.0

