# High-Performance American Option Pricing Engine

## Overview

This project implements a **high-performance American option pricing engine** using the **Crank–Nicolson Finite Difference Method (FDM)** applied to the Black–Scholes PDE. It is designed as a **modular, extensible quantitative library**, with a focus on:

- Numerical accuracy and stability
- Early exercise handling (American options)
- Scalable performance optimization
- Clean, production-ready engineering

---

## Objectives

1. Build a reusable PDE-based pricing engine
2. Accurately price American options with early exercise constraints
3. Compute option Greeks (Delta, Gamma, Theta) from the numerical grid
4. Validate against binomial tree and analytical solutions
5. Optimize performance through profiling and selective acceleration

---

## Tech Stack

| Component         | Choice                 |
| ----------------- | ---------------------- |
| **Language**      | Python (primary)       |
| **Numerics**      | NumPy                  |
| **Visualization** | Matplotlib             |
| **Performance**   | Numba / C++ (optional) |
| **Architecture**  | Modular package        |

---

## Project Structure

```
fdm_engine/
├── core/
│   ├── grid.py                    # Spatial & temporal discretization
│   └── boundary_conditions.py     # Domain boundary handling
├── models/
│   └── black_scholes.py           # PDE coefficients & parameters
├── solvers/
│   ├── crank_nicolson.py          # CN finite difference scheme
│   ├── thomas_solver.py           # Tridiagonal matrix solver
│   └── psor.py                    # Projected SOR for early exercise
├── instruments/
│   ├── american_option.py
│   └── european_option.py
├── benchmarks/
│   ├── binomial_tree.py           # Reference pricing model
│   └── convergence_tests.py
├── utils/
│   ├── greeks.py                  # Greeks computation
│   └── profiler.py                # Performance analysis
└── main.py                        # Entry point
```

---

## Quick Start

1. **Set up environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run pricing engine:**

   ```bash
   python main.py
   ```

3. **View implementation phases:**
   See [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed phase-by-phase breakdown.

---

## Key Features

- **Crank–Nicolson Finite Difference Method**: Second-order accurate, unconditionally stable scheme
- **American Option Support**: PSOR algorithm for early exercise constraint enforcement
- **Greeks Computation**: Delta, Gamma, Theta via finite differences
- **Comprehensive Benchmarking**: Validation against binomial tree and analytical Black–Scholes
- **Performance Profiling**: Identify and optimize bottlenecks

---

## Expected Outcomes

✓ Modular pricing engine (Python)  
✓ Benchmarking suite with convergence analysis  
✓ Visualization notebook with sensitivity plots  
✓ Performance report with optimization analysis

---

## Summary

Built a high-performance American option pricing engine using Crank–Nicolson FDM with PSOR for early exercise; validated against binomial tree and analytical models, achieving faster convergence and scalable performance.

---
