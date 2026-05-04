<p align="center">
  <h1 align="center">Crank–Nicolson FDM Option Pricing Engine</h1>
  <p align="center">
    A high-performance American & European option pricing engine using finite difference methods on the Black–Scholes PDE.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/numpy-numerics-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scipy-sparse-8CAAE6?logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## About

The Black–Scholes PDE governs the fair value of options under geometric Brownian motion. For European options, a closed-form solution exists. For **American options**, the possibility of early exercise transforms the PDE into a **free boundary problem**, requiring numerical methods.

This engine solves the resulting **Linear Complementarity Problem (LCP)** using the **Crank–Nicolson** finite difference scheme — second-order accurate and unconditionally stable — combined with the **Projected Successive Over-Relaxation (PSOR)** algorithm for early exercise enforcement.

### Key Equations

**Black–Scholes PDE** (with continuous dividend yield $q$):

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r - q)S\frac{\partial V}{\partial S} - rV = 0$$

**Log-space transform** ($x = \ln S$) yields constant coefficients:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial x^2} + \left(r - q - \frac{\sigma^2}{2}\right)\frac{\partial V}{\partial x} - rV = 0$$

**American constraint** (LCP):

$$V \geq \Phi(S), \quad \mathcal{L}V \leq 0, \quad (V - \Phi) \cdot \mathcal{L}V = 0$$

---

## Features

| Feature | Description |
|---------|-------------|
| **Crank–Nicolson FDM** | Second-order accurate, unconditionally stable PDE solver |
| **Log-Space Transform** | Constant PDE coefficients, improved near-boundary accuracy |
| **American Options (PSOR)** | Early exercise via projected SOR on the LCP |
| **Dividend Yield** | Continuous yield $q$ — critical for American call exercise |
| **Greeks** | Delta, Gamma, Theta from the numerical grid + analytical validation |
| **Benchmarking** | Convergence analysis against CRR binomial tree & analytical BS |
| **Performance** | Profiling pipeline with optional Numba JIT acceleration |

---

## Architecture

```
fdm_engine/
├── core/
│   ├── grid.py                    # Spatial & temporal discretization
│   └── boundary_conditions.py     # Domain boundary handling
├── models/
│   └── black_scholes.py           # PDE coefficients & analytical formulas
├── solvers/
│   ├── crank_nicolson.py          # CN finite difference scheme
│   ├── tridiagonal.py             # Thomas algorithm (O(N) tridiagonal solve)
│   └── psor.py                    # Projected SOR for LCP
├── instruments/
│   ├── option.py                  # Base option class
│   ├── american_option.py
│   └── european_option.py
├── greeks/
│   ├── numerical.py               # FDM-based Greeks
│   └── analytical.py              # BS closed-form Greeks
├── benchmarks/
│   ├── binomial_tree.py            # CRR reference pricer
│   └── convergence.py             # Grid convergence analysis
└── utils/
    └── profiler.py                 # Performance instrumentation
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

1. **Grid Construction** — Discretize the $(S, t)$ domain in log-space for constant PDE coefficients
2. **Backward Time-Stepping** — March from maturity to present using Crank–Nicolson
3. **Early Exercise (American)** — At each timestep, solve the LCP via PSOR to enforce $V \geq \Phi(S)$
4. **Greeks Extraction** — Compute Delta, Gamma, Theta from the solved grid via central differences
5. **Validation** — Compare against CRR binomial tree and analytical Black–Scholes

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Core** | Python 3.10+, NumPy, SciPy (`sparse`, `linalg`) |
| **Visualization** | Matplotlib |
| **Acceleration** | Numba JIT (optional) |
| **Testing** | pytest |
| **Build** | `pyproject.toml` |

---

## Roadmap

- [x] Project scaffolding & engineering foundation
- [ ] Grid system with log-space transform
- [ ] Crank–Nicolson solver (European validation)
- [ ] American option pricing (PSOR / LCP)
- [ ] Greeks computation & analytical validation
- [ ] Benchmarking suite (CRR, convergence analysis)
- [ ] Performance profiling & optimization
- [ ] Numba JIT acceleration (stretch)
- [ ] Visualization notebook
- [ ] Final packaging & documentation

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
