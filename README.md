# Hadamard Search — Agentic Falsification, Orbit-Space Search & Lean 4 Verification

An agentic discovery loop targeting open orders of the Hadamard conjecture ($n \le 10{,}000$, live open list from `math74.com/Hadamard/` / `arXiv:2610.27175v1`).

## Architecture

1. **Algebraic Falsification (`agent_loop.py`, `deep_falsify.py`)**:
   - Eliminates algebraically obstructed $(n, m, U)$ parameter sets via exact row-sum feasibility, compression, and power spectral density checks. The order-4412 analyzer uses prime-field moments and binary-field projections to complete a row from partial data.
2. **Orbit-Space Combinatorial Search (`solver.cpp`, `solver_v2.cpp`, `solver_v3.cpp`, `tmatrix_solver.cpp`)**:
   - C++20 multithreaded search for Goethals–Seidel sequence quadruples (Theorems 3 and 5) and circulant T-matrices for Cooper–Wallis constructions.
3. **Independent Exact Verification (`verify.py`)**:
   - Reconstructs the full $n \times n$ matrix $H$ over exact integers and verifies $H_{ij} \in \{\pm 1\}$ and $H H^T = n I_n$.
4. **Formal Verification in Lean 4 (`lean/`)**:
   - Lean 4 + Mathlib proofs of unbordered Goethals–Seidel (Theorem 3), even-length bordered Goethals–Seidel (Theorem 5), and the split order-three obstruction for Theorem 7. The published order-1948 witness is kernel-checked with `lake build HadamardWitness1948` from `lean/`.

See [HANDOFF.md](HANDOFF.md) for the search results and reproduction commands.
