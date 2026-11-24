# Hadamard Search — Agentic Falsification, Orbit-Space Search & Lean 4 Verification

An agentic discovery loop targeting open orders of the Hadamard conjecture ($n \le 10{,}000$, live open list from `math74.com/Hadamard/` / `arXiv:2610.27175v1`).

## Architecture

1. **Algebraic Falsification (`agent_loop.py`, `deep_falsify.py`)**:
   - Eliminates algebraically obstructed $(n, m, U)$ parameter sets via exact row-sum subset-sum feasibility, 2-compression alternating-sum four-squares obstructions, full-mod-$q$ checks, and prime-divisor quadratic-norm PSD obstructions.
2. **Orbit-Space Combinatorial Search (`solver.cpp`, `solver_v2.cpp`)**:
   - C++20 multi-threaded orbit-space Iterated Local Search (ILS) + Tabu + exact row-sum-preserving same-size orbit swaps targeting bordered Goethals–Seidel Legendre quadruples (Theorem 5, $n = 4m + 4$).
3. **Independent Exact Verification (`verify.py`)**:
   - Reconstructs the full $n \times n$ matrix $H$ over exact integers and verifies $H_{ij} \in \{\pm 1\}$ and $H H^T = n I_n$.
4. **Formal Verification in Lean 4 (`lean/`)**:
   - Sorry-free, `native_decide`-free algebraic proof of Theorem 5 (bordered Goethals–Seidel) in Lean 4 + Mathlib, plus concrete witness verification.
