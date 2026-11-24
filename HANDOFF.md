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

## Session 2 Status (wrap-up)

**No Hadamard matrix of a previously unknown order has been found.** Everything below is internal progress or reproduction of known techniques.

### Established techniques (reproduced, not novel)
- Theorems 3/5/7 (Goethals–Seidel, bordered GS) and Cooper–Wallis `T`-matrices per arXiv:2610.27175v2.
- Exact-integer obstruction checks: row sums, d-compression PAF/PSD, shift diagonal-count / MaxCut bound. The diagonal-count bound is elementary; I have not checked whether it is already in the literature.
- Calibration: `verify_tmatrix.py tmat_163.json` passes (T-matrix t=163 -> H(652), an already-known order). `verify.py` passes on small tests (n=44, 76, 108).

### Falsification results (restricted search spaces only; each is a statement about a specific multiplier-group family, NOT about order existence)
- All cyclic r<=48 Thm 5 and non-cyclic r<=36 Thm 3/5 multiplier families scanned were ruled out by the scripts. Theorem 7 (b=12, sigma_3) r<=24 families were exhaustively ruled out by `mitm_thm7.cpp`.
- CAVEAT: these certificates come from agent-written scripts and were only partly cross-checked independently (e.g. brute force on one d=473 case). Re-verify before citing.

### Live targets (unobstructed per our checks)
1. `n=4412, m=1103` prime, b=0, g=749 (or 17), |U|=19, r=59, K=29, row sums (1,37,39,39). `solver_v3 --mode=ils --lock_row=1 --flat_init=1` reached best E=832 (not 0). Known-solvable calibration `n=1948 m=487` stalled at E=480 -> search algorithm needs improvement (SA/RRT, flatness regularization were proposed, untested).
2. T-matrix `t=463`, g=15, |U|=11, r=43, K=21, sums (-10,11,11,11) -> would give H(9260) via Cooper–Wallis with WT(5). `tmatrix_solver.cpp` is built but UNTESTED on t=463; `decompress_r54.py` (r=54 Thm 5 decompression for 9388/7180) is unfinished/unrun.
3. Thm 5 r=54 survivors: 7180 g=49, 9388 g=25, 6236 g=119.

### Next steps
- Tune solver on n=1948 until E=0, then retry n=4412; run tmatrix_solver on t=163/151 then t=463.
- Any E=0 must be checked with verify.py / verify_tmatrix.py before any claim; then Lean witness.
