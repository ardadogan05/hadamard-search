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

## Session 3 results

The bounded searches produced no witness for a previously unknown order. On
24 November 2026, the 85 orders in `UNKNOWN_85` matched Math74's list through
order 10,000.

- **Theorem 5.** `python3 verify_historical_exclusions.py` independently
  reproduces the exclusions of `7572 g=15`, `6668 g=9`, and `7572 g=9`.
  `python3 -u decompress_r54.py 9388 7180 6236` gives the following exact
  results for the specified cyclic multiplier groups.

  | Order | Generator | Compression | Lifts |
  |---:|---:|---:|---:|
  | 9388 | 25 | 391 → 782 | 0 of 320 |
  | 7180 | 49 | 299 → 598 | 0 of 2188 |
  | 6236 | 119 | 19 → 38 | 3 of 136 |

  The zero counts exclude the respective multiplier families. The surviving
  6236 family has a directly verified modulus-38 witness, exportable with
  `python3 decompress_r54.py 6236 --witness-out /tmp/compressed-6236.json`.
  The modulus-41 enumeration exceeded the exploratory memory budget.
- **Theorem 7.** The split symmetry cycling the three nonzero
  `Z_2^2` coordinates and fixing `S` pointwise forces order 28. The proof is
  in `scan_thm7.py`. The scan finds zero eligible open orders after applying
  this restriction. `python3 verify_thm7.py` verifies the known order-28
  example. The MITM solver returns `INCONCLUSIVE` when its pair cap skips a
  case and the remaining cases yield no solution.
- **Goethals–Seidel search.** `python3 verify.py fixtures/published_1948.json`
  checks all autocorrelations and the full integer Gram matrix for the
  published order-1948 control. The new record-to-record mode found and
  exactly verified order 52 in 376 proposals. Blind order-1948 trials
  reached `E=640`, above the previous `E=480`. The two-thread ILS trial at
  order 4412 reached `E=864` in 120 seconds, above the previous `E=832`.
  The source of the control and
  the original search method are in [Alpöge, Voinov and Reynolds-Haertle, §5.1 and Table A1](https://hexagonmath.org/2611.00637v1).
- **T-matrix search.** The updated hybrid solver rediscovered `t=163`.
  `verify_tmatrix.py` verified its witness exactly. The `t=151` run reached
  `E=4` in 45 seconds. At `t=463`, the baseline reached `E=34` in 30 seconds
  and the revised solver reached `E=36` in 60 seconds.
- **Lean and witness verification.** `cd lean && lake build` checks the
  unbordered and even-length bordered constructions and a concrete
  order-12 witness. `verify.py` expands both single-generator and
  multi-generator witness files using exact integers.

Build and reproduce the bounded solver trials from the repository root.

```sh
g++ -std=c++20 -O3 -march=native -pthread solver_v3.cpp -o solver_v3
g++ -std=c++20 -O3 -march=native -pthread tmatrix_solver.cpp -o tmatrix_solver

./solver_v3 487 362 0 80 /tmp/rrt-1948.json 1 --ts=17,1,17,37 \
  --mode=rrt --lock_row=1 --flat_init=1 --w_flat=0.8 --seed=1948 --rrt_stage_steps=32000
./solver_v3 1103 749 0 120 /tmp/ils-4412.json 2 --ts=1,37,39,39 \
  --mode=ils --lock_row=1 --flat_init=1 --seed=4420
./tmatrix_solver 163 38 0 1 1 0 30 /tmp/tmatrix-163.json 2 hybrid
./tmatrix_solver 151 8 0 2 1 1 45 /tmp/tmatrix-151.json 2 hybrid
./tmatrix_solver 463 15 -1 1 1 1 60 /tmp/tmatrix-463.json 2 hybrid
```

## Session 4 results

- **Order 6236, `g=119`.** All 24 normalized lifts through modulus 38
  are incompatible with the modulus-82 energy. The CRT orbit partition
  makes each modulus-82 entry `s + 18t`, where `s = ±1` and
  `t ∈ {-1, 0, 1}`. The required energy forces the weighted sum of
  `t²` into `[17, 20]`. The modulus-38 lifts force it to be
  `6 mod 10`. A separate direct enumeration reproduced the exclusion.
  Run `python3 -u decompress_r54.py 6236`.
- **Split Theorem 7.** `HadamardSigma3.card_eq_one_of_paf` derives the
  row and quotient norms from the split autocorrelation identities and
  proves that the subgroup has one element, forcing order 28.
  `cd lean && lake build` checks the proof.
- **Published order 1948.** `HadamardWitness1948.hadamard_order_1948`
  proves the exact matrix identity from the published sign sequences.
  Multiplication and reversal reduce the certificate to 27 shift
  representatives. Lean checks all the reductions and equations with
  standard axioms. Build the concrete witness with
  `cd lean && lake build HadamardWitness1948`.
- **Order 4412 row completion.** `python3 analyze_4412.py` gives the
  prime-field reduction. Three rows and the fourth row's odd moments
  determine its 29 even coefficients, with an exact signed-completion
  test. `python3 analyze_4412.py --binary` gives a separate necessary
  binary-field condition. Once three rows are fixed, a nonzero residual
  norm leaves at most `2^29 - 1` fourth-row projections before the row
  size and integer autocorrelation checks. Both models recovered the
  published order-1948 rows used for calibration.
- **Meet-in-the-middle.** The paired solver generates individual rows
  with exact row sums and matches complementary autocorrelation vectors
  up to joint rotation. The injected order-1948 control passed
  `verify.py`. With identical blind pools of 400 rows per sequence,
  canonical pair matching reduced full run time from 22.73 to 3.94
  seconds at 1948 and from 30.96 to 3.92 seconds at 4412. These bounded
  pools contained no witness.

The order-4412 pair benchmark can be reproduced after building `solver_v3`.

```sh
./solver_v3 1103 749 0 90 /tmp/mitm-4412.json 1 --mode=mitm \
  --ts=1,37,39,39 --seed=4412 --mitm_pool=400 --mitm_flat_max=4400
```
