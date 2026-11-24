# Handoff: Agentic Hadamard Matrix Discovery & Lean 4 Verification

## Goal
Find a previously unknown Hadamard matrix of open order $n \le 10{,}000$ (live list of 85 open orders from https://www.math74.com/Hadamard/, updating `arXiv:2610.27175v1`), verify it independently over exact integers, and formally prove its existence in Lean 4 + Mathlib (0 `sorry`, 0 `native_decide`).

## What Is Built & Verified

1. **Lean 4 Formalization (`lean/HadamardTheorem5.lean`)**
   - Complete, sorry-free, `native_decide`-free algebraic proof of **Theorem 5** (bordered Goethals–Seidel construction of order $4m + 4$ from a Legendre quadruple of even length $m$):
     `HadamardTheorem5.bordered_goethals_seidel_hadamard`
   - Verified with `lake build` (Lean v4.36.0 + Mathlib): 0 errors, 0 `sorry`, only standard axioms `[propext, Classical.choice, Quot.sound]`.

2. **Exact-Integer Algebraic Falsifier (`deep_falsify.py`)**
   - 100% exact integer arithmetic (zero floating point): checks row-sum feasibility, 2-compression, 4-compression PSD, prime $q \mid m$ and $p^2 \mid m$ compressions, and simultaneous `(2, q, 2q)` PSD obstructions.
   - **Scanned all 85 live unknown orders for cyclic multiplier subgroups with $r \le 36$ orbits:**
     - Theorem 3 ($b=0$): **478 / 478 FALSIFIED** (0 survivors).
     - Theorem 5 ($b=4$): **492 / 495 FALSIFIED** (including `n=3788 g=47`, `n=4892 g=3`, `n=5708 g=35`, which were all proved impossible via exact simultaneous `(2, q, 2q)` PSD or Parseval obstructions).
     - **Only 3 cyclic $r \le 36$ Theorem 5 targets survived every check:**
       1. `n = 7572` (`m = 1892 = 2^2 * 11 * 43`, `g = 15`, `|U| = 210`, `r = 27`, `K = 14`) — unique 2-compression `(-42, 44, 44, 44)`
       2. `n = 6668` (`m = 1666 = 2 * 7^2 * 17`, `g = 9`, `|U| = 168`, `r = 30`, `K = 17`)
       3. `n = 7572` (`m = 1892 = 2^2 * 11 * 43`, `g = 9`, `|U| = 105`, `r = 36`, `K = 18`)

3. **C++20 Orbit-Space Solvers & Exact Verifier (`solver.cpp`, `solver_v2.cpp`, `verify.py`)**
   - `solver.cpp`: C++20 multi-threaded orbit-space ILS + Tabu + exact row-sum-preserving same-size orbit swaps (~640k full tabu steps/sec).
   - `solver_v2.cpp`: Upgraded with 2-compression manifold energy term + Parallel Tempering / ILS.
   - `verify.py`: Independent exact integer verifier reconstructing the full $n \times n$ matrix $H$ and checking $H_{ij} \in \{\pm 1\}$ and $H H^T = n I_n$.

## Recommended Next Steps in the Connected Repo Chat
1. Run `deep_falsify.py` on `37 <= r <= 48` (Theorem 5) and on Theorem 7 (`b = 12, 20, 28`, non-cyclic groups) to expand the list of unobstructed targets.
2. Check higher composite compressions on the 3 surviving `r <= 36` targets (`7572 g=15`, `7572 g=9`, `6668 g=9`).
3. Run focused multi-minute C++ `solver_v2` attacks on the best unobstructed survivors (especially `n = 7572, m = 1892, g = 15` with `r = 27, K = 14` and unique 2-compression `(-42, 44, 44, 44)`).
4. Once `E = 0` is hit: verify with `verify.py`, plug the 4 orbit vectors into `lean/HadamardTheorem5.lean` to get a machine-checked Lean 4 existence theorem, and push to `ardadogan05/hadamard-search`.
