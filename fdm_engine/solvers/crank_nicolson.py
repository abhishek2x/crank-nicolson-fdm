import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple

from fdm_engine.config import OptionParameters, GridParameters
from fdm_engine.core.grid import Grid, construct_log_grid, get_payoff
from fdm_engine.core.boundary_conditions import apply_boundary_conditions
from fdm_engine.solvers.tridiagonal import thomas_solve
from fdm_engine.solvers.psor import psor_solve

def price_european(
    option_params: OptionParameters,
    grid_params: GridParameters,
    is_call: bool = True,
    rannacher_steps: int = 4
) -> Tuple[float, Grid, np.ndarray, np.ndarray]:
    """
    Prices a European option using the Crank-Nicolson finite difference method in log-space.
    
    Args:
        option_params: OptionParameters containing S0, K, r, q, sigma, T.
        grid_params: GridParameters containing Ns, Nt, Smax_mult, align_strike, boundary_type.
        is_call: True for Call, False for Put.
        rannacher_steps: Number of Rannacher smoothing steps (Implicit Euler) at start.
                          Must be even. Set to 0 to disable.
                          
    Returns:
        Tuple containing:
            - price at S0 (float)
            - Grid object
            - Array of option values at t=0 on the grid (np.ndarray)
            - Array of option values at t=dt on the grid (np.ndarray)
    """
    # 1. Grid Construction
    grid = construct_log_grid(
        S0=option_params.S0,
        K=option_params.K,
        T=option_params.T,
        Ns=grid_params.Ns,
        Nt=grid_params.Nt,
        Smax_mult=grid_params.Smax_mult,
        align_strike=grid_params.align_strike
    )
    
    Ns = len(grid.x)
    Nt = len(grid.t) - 1  # number of time steps
    dx = grid.dx
    dt = grid.dt
    
    # 2. Setup PDE coefficients
    sigma2 = option_params.sigma ** 2
    alpha = 0.5 * sigma2
    beta = option_params.r - option_params.q - 0.5 * sigma2
    
    p_minus = alpha / (dx ** 2) - beta / (2.0 * dx)
    p_zero  = -2.0 * alpha / (dx ** 2) - option_params.r
    p_plus  = alpha / (dx ** 2) + beta / (2.0 * dx)
    
    # 3. Handle Rannacher smoothing steps
    R = 0
    if rannacher_steps > 0:
        R = min(rannacher_steps // 2 * 2, 2 * Nt)
        
    N_cn = Nt - R // 2
    
    V = get_payoff(grid.S, option_params.K, is_call=is_call)
    V_t1 = V.copy()
    
    M = Ns - 2
    
    sub_diag_cn = np.full(M, -0.5 * dt * p_minus)
    diag_cn = np.full(M, 1.0 - 0.5 * dt * p_zero)
    super_diag_cn = np.full(M, -0.5 * dt * p_plus)
    
    dt_ie = 0.5 * dt
    sub_diag_ie = np.full(M, -dt_ie * p_minus)
    diag_ie = np.full(M, 1.0 - dt_ie * p_zero)
    super_diag_ie = np.full(M, -dt_ie * p_plus)
    
    # 4. Backward Marching in Time
    for m in range(1, R + 1):
        if m == R and N_cn == 0:
            V_t1 = V.copy()
            
        tau = m * dt_ie
        d = V[1:-1].copy()
        
        V_bound = np.zeros(Ns)
        V_bound = apply_boundary_conditions(
            V_bound, grid.S, option_params.K, option_params.r, tau, option_params.q,
            is_call=is_call, is_american=False, boundary_type="dirichlet"
        )
        
        d[0] -= -dt_ie * p_minus * V_bound[0]
        
        if grid_params.boundary_type == "linearity":
            diag_mod = diag_ie.copy()
            sub_diag_mod = sub_diag_ie.copy()
            diag_mod[-1] += 2.0 * (-dt_ie * p_plus)
            sub_diag_mod[-1] += -(-dt_ie * p_plus)
            
            V_interior = thomas_solve(sub_diag_mod, diag_mod, super_diag_ie, d)
            V[0] = V_bound[0]
            V[1:-1] = V_interior
            V[-1] = 2.0 * V[-2] - V[-3]
        else:
            d[-1] -= -dt_ie * p_plus * V_bound[-1]
            V_interior = thomas_solve(sub_diag_ie, diag_ie, super_diag_ie, d)
            V[0] = V_bound[0]
            V[1:-1] = V_interior
            V[-1] = V_bound[-1]
            
    for k in range(1, N_cn + 1):
        if k == N_cn:
            V_t1 = V.copy()
            
        tau = R * dt_ie + k * dt
        
        a_rhs = 0.5 * dt * p_minus
        b_rhs = 1.0 + 0.5 * dt * p_zero
        c_rhs = 0.5 * dt * p_plus
        
        d = a_rhs * V[:-2] + b_rhs * V[1:-1] + c_rhs * V[2:]
        
        V_bound = np.zeros(Ns)
        V_bound = apply_boundary_conditions(
            V_bound, grid.S, option_params.K, option_params.r, tau, option_params.q,
            is_call=is_call, is_american=False, boundary_type="dirichlet"
        )
        
        d[0] -= -0.5 * dt * p_minus * V_bound[0]
        
        if grid_params.boundary_type == "linearity":
            diag_mod = diag_cn.copy()
            sub_diag_mod = sub_diag_cn.copy()
            diag_mod[-1] += 2.0 * (-0.5 * dt * p_plus)
            sub_diag_mod[-1] += -(-0.5 * dt * p_plus)
            
            V_interior = thomas_solve(sub_diag_mod, diag_mod, super_diag_cn, d)
            V[0] = V_bound[0]
            V[1:-1] = V_interior
            V[-1] = 2.0 * V[-2] - V[-3]
        else:
            d[-1] -= -0.5 * dt * p_plus * V_bound[-1]
            V_interior = thomas_solve(sub_diag_cn, diag_cn, super_diag_cn, d)
            V[0] = V_bound[0]
            V[1:-1] = V_interior
            V[-1] = V_bound[-1]
            
    # 5. Interpolate to find option price at S0
    interpolator = interp1d(grid.S, V, kind="cubic", fill_value="extrapolate")
    price_at_S0 = float(interpolator(option_params.S0))
    price_at_S0 = max(price_at_S0, 0.0)
    
    return price_at_S0, grid, V, V_t1

def price_american(
    option_params: OptionParameters,
    grid_params: GridParameters,
    is_call: bool = False,
    rannacher_steps: int = 4,
    omega: float = 1.2,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, Grid, np.ndarray, np.ndarray]:
    """
    Prices an American option using the Crank-Nicolson finite difference method in log-space.
    Uses Projected Successive Over-Relaxation (PSOR) to enforce early exercise.
    
    Args:
        option_params: OptionParameters containing S0, K, r, q, sigma, T.
        grid_params: GridParameters containing Ns, Nt, Smax_mult, align_strike, boundary_type.
        is_call: True for Call, False for Put.
        rannacher_steps: Number of Rannacher smoothing steps (Implicit Euler) at start.
        omega: Relaxation parameter for PSOR solver.
        tol: Convergence tolerance for PSOR solver.
        max_iter: Maximum number of iterations allowed per step in PSOR solver.
        
    Returns:
        Tuple containing:
            - price at S0 (float)
            - Grid object
            - Array of option values at t=0 on the grid (np.ndarray)
            - Array of option values at t=dt on the grid (np.ndarray)
    """
    # 1. Grid Construction
    grid = construct_log_grid(
        S0=option_params.S0,
        K=option_params.K,
        T=option_params.T,
        Ns=grid_params.Ns,
        Nt=grid_params.Nt,
        Smax_mult=grid_params.Smax_mult,
        align_strike=grid_params.align_strike
    )
    
    Ns = len(grid.x)
    Nt = len(grid.t) - 1
    dx = grid.dx
    dt = grid.dt
    
    # 2. Setup PDE coefficients
    sigma2 = option_params.sigma ** 2
    alpha = 0.5 * sigma2
    beta = option_params.r - option_params.q - 0.5 * sigma2
    
    p_minus = alpha / (dx ** 2) - beta / (2.0 * dx)
    p_zero  = -2.0 * alpha / (dx ** 2) - option_params.r
    p_plus  = alpha / (dx ** 2) + beta / (2.0 * dx)
    
    # 3. Handle Rannacher smoothing steps
    R = 0
    if rannacher_steps > 0:
        R = min(rannacher_steps // 2 * 2, 2 * Nt)
        
    N_cn = Nt - R // 2
    
    payoff = get_payoff(grid.S, option_params.K, is_call=is_call)
    V = payoff.copy()
    V_t1 = V.copy()
    
    M = Ns - 2
    g = payoff[1:-1]
    
    sub_diag_cn = np.full(M, -0.5 * dt * p_minus)
    diag_cn = np.full(M, 1.0 - 0.5 * dt * p_zero)
    super_diag_cn = np.full(M, -0.5 * dt * p_plus)
    
    dt_ie = 0.5 * dt
    sub_diag_ie = np.full(M, -dt_ie * p_minus)
    diag_ie = np.full(M, 1.0 - dt_ie * p_zero)
    super_diag_ie = np.full(M, -dt_ie * p_plus)
    
    # 4. Backward Marching in Time
    for m in range(1, R + 1):
        if m == R and N_cn == 0:
            V_t1 = V.copy()
            
        tau = m * dt_ie
        d = V[1:-1].copy()
        
        V_bound = np.zeros(Ns)
        V_bound = apply_boundary_conditions(
            V_bound, grid.S, option_params.K, option_params.r, tau, option_params.q,
            is_call=is_call, is_american=True, boundary_type="dirichlet"
        )
        
        d[0] -= -dt_ie * p_minus * V_bound[0]
        
        if grid_params.boundary_type == "linearity":
            diag_mod = diag_ie.copy()
            sub_diag_mod = sub_diag_ie.copy()
            diag_mod[-1] += 2.0 * (-dt_ie * p_plus)
            sub_diag_mod[-1] += -(-dt_ie * p_plus)
            
            V_interior, _ = psor_solve(
                sub_diag_mod, diag_mod, super_diag_ie, d, g, V[1:-1],
                omega=omega, tol=tol, max_iter=max_iter
            )
            V[0] = V_bound[0]
            V[1:-1] = V_interior
            V[-1] = 2.0 * V[-2] - V[-3]
        else:
            d[-1] -= -dt_ie * p_plus * V_bound[-1]
            V_interior, _ = psor_solve(
                sub_diag_ie, diag_ie, super_diag_ie, d, g, V[1:-1],
                omega=omega, tol=tol, max_iter=max_iter
            )
            V[0] = V_bound[0]
            V[1:-1] = V_interior
            V[-1] = V_bound[-1]
            
    for k in range(1, N_cn + 1):
        if k == N_cn:
            V_t1 = V.copy()
            
        tau = R * dt_ie + k * dt
        
        a_rhs = 0.5 * dt * p_minus
        b_rhs = 1.0 + 0.5 * dt * p_zero
        c_rhs = 0.5 * dt * p_plus
        
        d = a_rhs * V[:-2] + b_rhs * V[1:-1] + c_rhs * V[2:]
        
        V_bound = np.zeros(Ns)
        V_bound = apply_boundary_conditions(
            V_bound, grid.S, option_params.K, option_params.r, tau, option_params.q,
            is_call=is_call, is_american=True, boundary_type="dirichlet"
        )
        
        d[0] -= -0.5 * dt * p_minus * V_bound[0]
        
        if grid_params.boundary_type == "linearity":
            diag_mod = diag_cn.copy()
            sub_diag_mod = sub_diag_cn.copy()
            diag_mod[-1] += 2.0 * (-0.5 * dt * p_plus)
            sub_diag_mod[-1] += -(-0.5 * dt * p_plus)
            
            V_interior, _ = psor_solve(
                sub_diag_mod, diag_mod, super_diag_cn, d, g, V[1:-1],
                omega=omega, tol=tol, max_iter=max_iter
            )
            V[0] = V_bound[0]
            V[1:-1] = V_interior
            V[-1] = 2.0 * V[-2] - V[-3]
        else:
            d[-1] -= -0.5 * dt * p_plus * V_bound[-1]
            V_interior, _ = psor_solve(
                sub_diag_cn, diag_cn, super_diag_cn, d, g, V[1:-1],
                omega=omega, tol=tol, max_iter=max_iter
            )
            V[0] = V_bound[0]
            V[1:-1] = V_interior
            V[-1] = V_bound[-1]
            
    # 5. Interpolate to find option price at S0
    interpolator = interp1d(grid.S, V, kind="cubic", fill_value="extrapolate")
    price_at_S0 = float(interpolator(option_params.S0))
    
    intrinsic_at_S0 = max(option_params.S0 - option_params.K if is_call else option_params.K - option_params.S0, 0.0)
    price_at_S0 = max(price_at_S0, intrinsic_at_S0)
    
    return price_at_S0, grid, V, V_t1
