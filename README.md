<p align="center">
  <h1 align="center">Crank–Nicolson FDM Option Pricing Engine</h1>
  <p align="center">
    A high-performance American &amp; European option pricing engine using finite difference methods on the Black–Scholes PDE.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/numpy-numerics-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scipy-sparse-8CAAE6?logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/tests-18%20passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## About

The Black–Scholes PDE governs the fair value of options under geometric Brownian motion. For European options, a closed-form solution exists. For **American options**, the possibility of early exercise transforms the PDE into a **free boundary problem**, requiring numerical methods.

This engine solves the resulting **Linear Complementarity Problem (LCP)** using the **Crank–Nicolson** finite difference scheme — second-order accurate and unconditionally stable — combined with the **Projected Successive Over-Relaxation (PSOR)** algorithm for early exercise enforcement.

### Key Equations

**Black–Scholes PDE** (with continuous dividend yield $q$):

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r - q)S\frac{\partial V}{\partial S} - rV = 0$$

**Log-space transform** ($x = \ln S$) yields constant PDE coefficients:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial x^2} + \left(r - q - \frac{\sigma^2}{2}\right)\frac{\partial V}{\partial x} - rV = 0$$

**American constraint** (LCP):

$$V \geq \Phi(S), \quad \mathcal{L}V \leq 0, \quad (V - \Phi) \cdot \mathcal{L}V = 0$$

---

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Log-Space Grid** | ✅ | Uniform grid in $x = \ln S$; strike alignment for $O(h^2)$ convergence |
| **Crank–Nicolson FDM** | ✅ | Second-order accurate, unconditionally stable PDE solver |
| **Rannacher Smoothing** | ✅ | Implicit Euler warm-up steps to damp payoff kink oscillations |
| **Thomas Algorithm** | ✅ | Hand-rolled $O(N)$ tridiagonal solver (Numba-ready) |
| **Analytical Black-Scholes** | ✅ | Closed-form European prices & Greeks for validation |
| **American Options (PSOR)** | ✅ | Early exercise via projected SOR on the LCP |
| **Dividend Yield** | ✅ | Continuous yield $q$ — critical for American call exercise |
| **Greeks** | ✅ | Delta, Gamma, Theta from numerical grid + analytical validation |
| **Benchmarking** | ✅ | CRR binomial tree benchmark & empirical $O(h^2)$ convergence test |
| **Performance** | ⬜ | Profiling pipeline with optional Numba JIT acceleration |

---

## Architecture

```
fdm_engine/
├── core/
│   ├── grid.py                    # Log-space grid (with strike alignment)
│   └── boundary_conditions.py     # Dirichlet & linearity boundary conditions
├── models/
│   └── black_scholes.py           # Analytical European pricing (validation)
├── solvers/
│   ├── crank_nicolson.py          # CN scheme: price_european & price_american
│   ├── tridiagonal.py             # Thomas algorithm — O(N) tridiagonal solve
│   └── psor.py                    # Projected SOR for American option LCP
├── greeks/
│   ├── analytical.py              # BSM analytical closed-form Greeks
│   └── numerical.py               # FDM grid spatial & time finite differences
├── benchmarks/
│   ├── binomial_tree.py           # CRR binomial tree reference pricer
│   └── convergence.py             # Empirical log-log convergence rate solver
├── instruments/                   # [Future] Option instrument classes
└── utils/                         # [Phase 6] Profiling instrumentation
```

---

## Quick Start

```bash
# Clone & setup
git clone https://github.com/<your-username>/crank-nicolson-fdm.git
cd crank-nicolson-fdm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the pricing engine
python main.py

# Run tests
python -m pytest tests/ -v
```

---

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Configure   │────▶│  Build Grid  │────▶│  Set Boundary   │
│  Parameters  │     │  (log-space) │     │  Conditions     │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────┐     ┌─────────▼────────┐
                    │  Compute     │◀────│  Solve PDE       │
                    │  Greeks      │     │  (CN + PSOR)     │
                    └──────┬───────┘     └──────────────────┘
                           │
                    ┌──────▼───────┐
                    │  Benchmark   │
                    │  & Validate  │
                    └──────────────┘
```

1. **Grid Construction** — Discretize the $(S, t)$ domain in log-space; enforce $\ln(K)$ is exactly a grid node (strike alignment)
2. **Rannacher Smoothing** — Run 4 implicit Euler half-steps from maturity to damp kink-induced oscillations
3. **Backward Time-Stepping** — March from maturity to present using Crank–Nicolson
4. **Early Exercise (American)** — At each timestep, solve the LCP via PSOR to enforce $V \geq \Phi(S)$
5. **Greeks Extraction** *(Phase 4)* — Compute Delta, Gamma, Theta from the solved grid via central differences
6. **Validation** *(Phase 5)* — Compare against analytical Black–Scholes and CRR binomial tree

---

## Current Results

| Option Type | FDM Price | Analytical Price | Status |
|-------------|-----------|-----------------|--------|
| European Call (ATM) | ~10.451 | ~10.451 | < 0.1% error ✅ |
| European Put (ATM) | ~5.574 | ~5.574 | < 0.1% error ✅ |
| American Put (ATM) | ~6.077 | — | > European Put ✅ |
| American Call (no div.) | ~10.451 | ~10.451 | = European Call ✅ |

> Parameters: $S_0 = K = 100$, $r = 5\%$, $\sigma = 20\%$, $T = 1$ year, $N_S = 200$, $N_t = 100$

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Core** | Python 3.9+, NumPy, SciPy |
| **Testing** | pytest (18 tests, all passing) |
| **Visualization** | Matplotlib *(Phase 8)* |
| **Acceleration** | Numba JIT *(Phase 7, optional)* |
| **Build** | `pyproject.toml` |

---

## Roadmap

- [x] Project scaffolding & engineering foundation
- [x] Log-space grid with strike alignment
- [x] Crank–Nicolson solver (European validation < 0.1% error)
- [x] American option pricing via PSOR (early exercise constraints)
- [x] Greeks computation & analytical validation
- [x] Benchmarking suite (CRR binomial tree, empirical $O(h^2)$ convergence rate)
- [ ] Performance profiling & optimization
- [ ] Numba JIT acceleration (stretch)
- [ ] Visualization notebook
- [ ] Final packaging & documentation

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
