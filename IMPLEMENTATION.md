# Implementation Roadmap

Detailed phase-by-phase breakdown for building the American Option Pricing Engine.

---

## Phase 0: Project Setup & Engineering Foundation

**Duration:** ~1-2 days  
**Objective:** Establish a clean, scalable codebase.

### Tasks

- [ ] Set up repository with modular folder structure
- [ ] Initialize virtual environment and `requirements.txt`
- [ ] Create CLI entry point (`main.py`)
- [ ] Add logging and configuration handling
- [ ] Create `.gitignore`, `setup.py` (or `pyproject.toml`)

### Success Criteria

- Project builds cleanly with `python main.py`
- All imports resolve without errors
- Modular structure ready for extension

### Key Files to Create

- `fdm_engine/__init__.py`
- `fdm_engine/config.py`
- `requirements.txt`

---

## Phase 1: Mathematical Foundation & Grid System

**Duration:** ~2-3 days  
**Objective:** Discretize the Black–Scholes PDE and construct the computational grid.

### Tasks

- [ ] Implement spatial grid:
  - Uniform grid in asset price $S$
  - Optional: Log-space grid for improved coverage
- [ ] Implement time discretization (backward in time from maturity)
- [ ] Define terminal payoff functions:
  - American Put: max($K - S$, 0)
  - American Call: max($S - K$, 0)
- [ ] Implement boundary conditions:
  - Deep ITM: $V \approx$ intrinsic value
  - Deep OTM: $V \approx 0$

### Key Parameters

| Symbol   | Meaning             | Typical Range |
| -------- | ------------------- | ------------- |
| $S_0$    | Current asset price | 100           |
| $K$      | Strike price        | 100           |
| $r$      | Risk-free rate      | 0.05          |
| $\sigma$ | Volatility          | 0.2           |
| $T$      | Time to maturity    | 1 year        |
| $N_S$    | Spatial grid points | 100–500       |
| $N_t$    | Time steps          | 50–250        |

### Success Criteria

- Grid values initialized correctly
- Terminal payoff matches expected intrinsic values
- Boundary values behave as expected

### Key Files

- `fdm_engine/core/grid.py`
- `fdm_engine/core/boundary_conditions.py`

---

## Phase 2: Crank–Nicolson Solver Implementation

**Duration:** ~3-4 days  
**Objective:** Build the core PDE solver.

### Mathematical Background

The Black–Scholes PDE:
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

Crank–Nicolson scheme (second-order accurate):
$$V_i^{n+1} = V_i^n + \text{implicit correction}$$

### Tasks

- [ ] Discretize PDE using Crank–Nicolson finite differences
- [ ] Construct tridiagonal system $AV^{n+1} = BV^n + \text{RHS}$
- [ ] Implement Thomas algorithm (O(N) solution)
- [ ] Verify stability: $r \Delta t \leq \frac{1}{2}$
- [ ] Test against analytical Black–Scholes formula (European options)

### Success Criteria

- European option prices match Black–Scholes formula (error < 0.1%)
- Convergence with increasing grid size
- No numerical instabilities (oscillations)

### Key Files

- `fdm_engine/models/black_scholes.py`
- `fdm_engine/solvers/crank_nicolson.py`
- `fdm_engine/solvers/thomas_solver.py`

---

## Phase 3: American Option Logic (Early Exercise)

**Duration:** ~3-4 days  
**Objective:** Incorporate early exercise constraints.

### Early Exercise Constraint

At each time step:
$$V_i^n = \max(\text{continuation value}, \text{intrinsic value})$$

### Implementation Approaches

**Option A: PSOR (Projected Successive Over-Relaxation)** ⭐ Recommended

- Iterative approach: Solve with relaxation then project
- Convergence tolerance: $\epsilon = 10^{-6}$
- Typical iterations: 5–20 per timestep

**Option B: Brennan–Schwartz Algorithm**

- Direct enforcement before each solve
- Often faster for sparse grids

### Tasks

- [ ] Implement PSOR algorithm
  - Main solver loop with relaxation parameter $\omega$ (typically 1.0–1.5)
  - Convergence check on max residual
  - Projection step: `max(continuation, intrinsic)`
- [ ] Tune convergence tolerance and iteration limits
- [ ] Validate: American Put > European Put (same parameters)

### Success Criteria

- American option prices > European prices (same strike/maturity)
- Early exercise boundary correctly identified
- Convergence achieved within 5–20 iterations per step

### Key Files

- `fdm_engine/solvers/psor.py`
- `fdm_engine/instruments/american_option.py`
- `fdm_engine/instruments/european_option.py`

---

## Phase 4: Greeks & Sensitivity Analysis

**Duration:** ~2-3 days  
**Objective:** Extend solver to compute risk sensitivities.

### Greeks Formulas (Finite Differences)

| Greek            | Formula                                                          | Type       |
| ---------------- | ---------------------------------------------------------------- | ---------- |
| Delta ($\Delta$) | $\frac{V(S + \Delta S) - V(S - \Delta S)}{2\Delta S}$            | 1st deriv  |
| Gamma ($\Gamma$) | $\frac{V(S + \Delta S) - 2V(S) + V(S - \Delta S)}{(\Delta S)^2}$ | 2nd deriv  |
| Theta ($\Theta$) | $\frac{V(t) - V(t + \Delta t)}{-\Delta t}$                       | Time deriv |

### Tasks

- [ ] Compute Greeks from solved grid
  - Use central differences for stability
  - Apply smoothing if needed (e.g., rolling average)
- [ ] Validate against finite difference bumping
- [ ] Plot Greeks as function of spot price $S$
- [ ] Detect and handle numerical noise in derivatives

### Success Criteria

- Greeks smooth and stable near-the-money
- Delta ∈ [-1, 0] for puts, ∈ [0, 1] for calls
- Gamma > 0 for both puts and calls

### Key Files

- `fdm_engine/utils/greeks.py`

---

## Phase 5: Benchmarking & Validation

**Duration:** ~3-4 days  
**Objective:** Validate correctness and compare performance.

### Benchmarks

1. **FDM vs Analytical (European):**
   - Error: |FDM_Price - BS_Price| / BS_Price
   - Target error: < 0.1% for fine grid

2. **FDM vs Binomial Tree (American):**
   - Compare prices on increasingly fine grids
   - Measure convergence rate (should be O(h²))

3. **Performance Metrics:**
   - Runtime vs grid size
   - Memory usage
   - Iterations per timestep

### Tasks

- [ ] Implement binomial tree pricer (Cox-Ross-Rubinstein)
- [ ] Create convergence test suite:
  - Sweep $N_S$ from 50 to 500
  - Plot error vs grid size
- [ ] Generate benchmark table:
  ```
  | Grid | FDM Price | Binomial | Error | Runtime |
  ```
- [ ] Document results in markdown table

### Success Criteria

- FDM converges to binomial tree solution
- Convergence rate ≈ O(h²)
- Reasonable runtime (< 1s for moderate grid)

### Key Files

- `fdm_engine/benchmarks/binomial_tree.py`
- `fdm_engine/benchmarks/convergence_tests.py`

---

## Phase 6: Profiling & Performance Optimization

**Duration:** ~2-3 days  
**Objective:** Identify and optimize bottlenecks.

### Profiling Tools

- `cProfile` for function-level timing
- `line_profiler` for line-by-line breakdown

### Typical Bottlenecks

1. **PSOR iterations:** Main loop with many matrix solves
2. **Thomas solver:** Called $N_t \times$ iterations times
3. **Boundary condition updates:** Repeated every timestep
4. **Greeks computation:** Requires multiple full solves

### Tasks

- [ ] Profile code with `cProfile`:
  ```bash
  python -m cProfile -s cumulative main.py
  ```
- [ ] Identify top 3 bottlenecks
- [ ] Optimize:
  - Vectorize where possible
  - Reduce memory allocations in loops
  - Cache repeated computations
- [ ] Measure speedup after each optimization
- [ ] Document before/after timings

### Success Criteria

- Identified bottlenecks clearly documented
- ≥25% improvement from baseline
- Code remains maintainable

### Key Files

- `fdm_engine/utils/profiler.py`

---

## Phase 7: Acceleration (Numba / C++ Optional)

**Duration:** ~3-5 days  
**Objective:** Improve performance of critical components.

### Option A: Numba JIT Compilation ⭐ Recommended

**Pros:** Easy integration, no external build system, 10x+ speedup possible  
**Cons:** Limited Python compatibility

```python
from numba import njit

@njit
def thomas_solve_numba(a, b, c, d):
    # Tridiagonal solver
    ...
```

**Target functions:**

- Thomas solver
- PSOR inner loop
- Greeks computation

**Expected speedup:** 5–15x

### Option B: C++ with pybind11

**Pros:** Maximum performance, full control  
**Cons:** Build complexity, C++ knowledge required

**Recommended modules to move:**

- Core tridiagonal solver
- PSOR iteration loop

**Expected speedup:** 3–10x (vs pure Python)

### Tasks (Option A)

- [ ] Install Numba: `pip install numba`
- [ ] Decorate hot functions with `@njit`
- [ ] Benchmark compiled vs uncompiled
- [ ] Document speedup achieved
- [ ] Validate numerical accuracy unchanged

### Tasks (Option B)

- [ ] Create C++ module (solver.cc)
- [ ] Build with pybind11
- [ ] Add unit tests
- [ ] Benchmark vs Python

### Success Criteria

- ≥5x speedup on core solver
- Compile/import time negligible
- Results identical to Python version

### Key Files

- `fdm_engine/solvers/thomas_solver.py` (Numba-accelerated)
- `fdm_engine/solvers/psor.py` (Numba-accelerated)
- Optionally: `src/solver.cc`, `setup.py` (C++ build)

---

## Phase 8: Visualization & Demo

**Duration:** ~2-3 days  
**Objective:** Make results interpretable and presentation-ready.

### Visualization Types

1. **Option Price Surface:**
   - 2D heatmap: Price vs (S, t)
   - 3D surface plot

2. **Greeks Curves:**
   - Delta vs Spot
   - Gamma vs Spot
   - Theta vs Spot

3. **Convergence Analysis:**
   - Error vs grid size (log-log plot)
   - Runtime vs grid size

4. **Early Exercise Boundary:**
   - Show optimal exercise frontier for American option

### Tasks

- [ ] Create `notebooks/visualization.ipynb`:
  - Load price & Greeks from solver
  - Generate plots with Matplotlib
  - Add annotations and legends
- [ ] Create comparison plots:
  - FDM vs Binomial Tree
  - American vs European
- [ ] 3D surface visualization (Matplotlib)
- [ ] Export high-quality figures (PNG, PDF)

### Success Criteria

- 4–6 publication-quality plots
- Clear legends and labels
- Interactive notebook runnable end-to-end

### Key Files

- `notebooks/visualization.ipynb`

---

## Phase 9: Final Packaging (Resume Ready)

**Duration:** ~2-3 days  
**Objective:** Present project professionally.

### Tasks

- [ ] **Update README (main):**
  - Problem statement
  - Key results & benchmarks
  - Usage example
  - Link to IMPLEMENTATION.md
- [ ] **Create 1-page summary PDF:**
  - High-level approach
  - Key results table
  - Performance metrics
  - "Portfolio-ready" snapshot

- [ ] **Add BENCHMARKS.md:**
  - Convergence analysis
  - Performance comparison
  - Numerical validation

- [ ] **Optional: QR code linking to:**
  - GitHub repo
  - Demo notebook
  - PDF summary

- [ ] **Polish code:**
  - Add docstrings to public APIs
  - Final linting/formatting
  - Remove debug prints

### Success Criteria

- Repository is "resume-ready"
- Professional appearance
- Results clearly communicated
- Entry point simple (`python main.py`)

### Key Files

- Updated `README.md`
- `BENCHMARKS.md`
- `INVESTMENT_SUMMARY.pdf` (1-page)

---

## Optional Advanced Extensions

Consider these after core implementation is complete:

### Volatility Surface

- Implement $\sigma = f(S, t)$ instead of constant
- Handle local volatility models
- Calibrate to market prices

### Adaptive Grid Refinement

- Concentrate points near strike ($S = K$)
- Refine in regions of high curvature (Gamma)
- Improve accuracy with fewer points

### Early Exercise Boundary Visualization

- Track optimal exercise frontier
- Plot boundary vs time to maturity
- Compare across spot prices

### Stability Analysis

- Compare Crank–Nicolson vs explicit scheme
- Measure error growth over long maturities
- Validate unconditional stability claim

### Multi-dimensional Extensions

- 2D: Multiple underlyings (basket options)
- Stochastic volatility (Heston model)

---

## Key References & Formulas

### Black–Scholes PDE

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

### Crank–Nicolson Discretization

$$\frac{V_i^{n+1} - V_i^n}{\Delta t} = \frac{1}{2}\left[\mathcal{L}(V^{n+1}) + \mathcal{L}(V^n)\right]$$

where $\mathcal{L}$ is the spatial differential operator.

### American Option Constraint

$$V \geq \max(0, S - K) \quad \text{(call)}$$
$$V \geq \max(0, K - S) \quad \text{(put)}$$

---

## Timeline Summary

| Phase         | Duration        | Status        |
| ------------- | --------------- | ------------- |
| 0: Setup      | 1–2 days        | ⬜            |
| 1: Grid       | 2–3 days        | ⬜            |
| 2: Solver     | 3–4 days        | ⬜            |
| 3: American   | 3–4 days        | ⬜            |
| 4: Greeks     | 2–3 days        | ⬜            |
| 5: Benchmark  | 3–4 days        | ⬜            |
| 6: Profile    | 2–3 days        | ⬜            |
| 7: Accelerate | 3–5 days        | ⬜ (optional) |
| 8: Visualize  | 2–3 days        | ⬜            |
| 9: Package    | 2–3 days        | ⬜            |
| **Total**     | **~27–40 days** |               |

---

## Development Workflow

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Development loop
python main.py                    # Test current phase
python -m pytest tests/           # Run unit tests
python -m cProfile -s cumulative main.py  # Profile

# Benchmarking
python fdm_engine/benchmarks/convergence_tests.py

# Final demo
jupyter notebook notebooks/visualization.ipynb
```

---

## Success Criteria Checklist

- [ ] Phase 0: Project builds cleanly
- [ ] Phase 1: Grid tests pass
- [ ] Phase 2: European option prices < 0.1% error vs BS
- [ ] Phase 3: American prices > European, early exercise detected
- [ ] Phase 4: Greeks computed, validated via bumping
- [ ] Phase 5: FDM converges to binomial tree
- [ ] Phase 6: Bottlenecks identified, 25%+ speedup achieved
- [ ] Phase 7: Numba acceleration working (optional)
- [ ] Phase 8: Publication-quality plots generated
- [ ] Phase 9: README & PDF finalized, resume-ready
