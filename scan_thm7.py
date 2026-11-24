#!/usr/bin/env python3
r"""
Exact-Integer Scanner for Theorem 7 (b = 12, k = 4, v = (n - 12)/4 = 4s)
across all 85 live open Hadamard orders <= 10,000 (UNKNOWN_85).

Theorem 7 (bordered Goethals-Seidel of width b = 12 over abelian G of order v = 4s
with index-4 subgroup S <= G, |S| = s):
  sum_f P_f(0) = 4v = 16s
  sum_f P_f(d) = -12   for d in S \ {0}
  sum_f P_f(d) = +4    for d in G \ S

Exact Fourier / compression consequences (100% integer arithmetic):
  1. Row sums: sum_f s_f^2 = n = 4v + 12 = 16s + 12.
  2. G/S-compression (3 non-trivial characters chi of G/S):
       sum_f |f_hat(chi)|^2 = 12  for each chi != 1 of G/S.
     For G = Z_2^2 x S with order-3 action sigma_3 on Z_2^2:
       a_f(00) = a_f, a_f(01)=a_f(10)=a_f(11) = b_f,
       s_f = a_f + 3*b_f,  d_f = a_f - b_f,
       sum_f d_f^2 = 12 => {|d_f|} = {0, 2, 2, 2} (UNIQUE up to sign/perm!).
  3. Stage-0 diag_const:
       4*(2*D(d) - v) <= -12  for d in S \ {0}
       4*(2*D(d) - v) <= +4   for d in G \ S.
  4. Prime/field q-compression on S (for every odd prime q | s, or F_{p^a}^+):
       sum_f |f_hat(chi)|^2 = n  for every chi with chi|_S != 1!

An order restriction applies before any subgroup enumeration. For the split family
G = Z_2^2 x S with sigma_3 fixing S pointwise and cycling the three nonzero
Z_2^2 coordinates, write each sequence as (x, y, y, y), with x and y signed
vectors on S. At any shift (c, 0), c != 0, addition by c exchanges the zero
coset with the c coset, and exchanges the two remaining cosets. Thus
P_f(c, 0) = 2s + 2<x, y> = 4e_f, where e_f counts positions with x = y.
The required combined PAF is +4, so exactly one e_f is 1 and the others are 0.
Writing a_f = sum x, b_f = sum y, this means q_f = a_f + b_f is +/-2 in
that one sequence and zero in the others.

The quotient condition in (2) gives sum_f (a_f - b_f)^2 = 12. The differences
d_f = a_f - b_f are even because a_f and b_f both have parity s, so their
absolute values are 0, 2, 2, 2. In each of the three rows with q_f = 0,
d_f = -2b_f. If s is even, all three differences are multiples of 4 and
must vanish, contradicting the unique zero. If s is odd, all three have
absolute value 2, so the exceptional row has d_f = 0. The row sum is
2q_f - d_f, so (1) gives n = 16 + 3*4 = 28. This includes the s = 1
example checked by verify_thm7.py. No other order is possible in this
split sigma_3 family, regardless of the multiplier action on S.
"""

import math
import time
from collections import defaultdict

from deep_falsify import (
    UNKNOWN_85,
    factor,
    reachable_sums,
    sum_of_4_in_lists,
    sum_of_4_tuples,
)
from scan_expanded import (
    find_all_subgroups_with_r_le,
    get_orbits_U,
    get_shift_reps_U,
    fmt_fac,
)


SPLIT_SIGMA3_ORDER = 28


def divisors(n):
    fac = factor(n)
    ds = [1]
    for p, e in fac.items():
        ds = [d * (p ** k) for d in ds for k in range(e + 1)]
    return sorted(ds)


def check_thm7_sigma3_row_and_coset(n, s, orb_sizes_S):
    """Exact simultaneous row-sum + Z_2^2-coset-sum check for G = Z_2^2 x S with sigma_3 action.
    Each sequence f chooses a_f in R_S (sum on (0,0)+S) and b_f in R_S (sum on each of the 3 nonzero cosets).
    Then s_f = a_f + 3*b_f and d_f = a_f - b_f.
    Conditions:
      - sum_f s_f^2 = n
      - {|d_f|} = {0, 2, 2, 2}  (i.e. one f has d_f = 0, three have d_f in {-2, +2}).
    Since flipping f negates (s_f, d_f), we can fix s_f >= 0 (keeping both signs of d_f when s_f > 0).
    Returns list of valid quadruples of (s_f, d_f) sorted."""
    if n != SPLIT_SIGMA3_ORDER:
        return []
    R_S = set(reachable_sums(orb_sizes_S))
    lim = math.isqrt(n)

    # Achievable s_f >= 0 with d_f == 0: a_f = b_f in R_S => s_f = 4*b_f
    S_d0 = sorted({abs(4 * b) for b in R_S if abs(4 * b) <= lim})

    # Achievable s_f >= 0 with |d_f| == 2: a_f = b_f +/- 2 in R_S => s_f = 4*b_f +/- 2
    S_d2 = set()
    for b in R_S:
        if (b + 2) in R_S:
            sv = abs(4 * b + 2)
            if sv <= lim:
                S_d2.add(sv)
        if (b - 2) in R_S:
            sv = abs(4 * b - 2)
            if sv <= lim:
                S_d2.add(sv)
    S_d2 = sorted(S_d2)
    S_d2_sq = {x * x: x for x in S_d2}

    sols = []
    for s0 in S_d0:
        r0 = n - s0 * s0
        if r0 < 0:
            continue
        for i, s1 in enumerate(S_d2):
            r1 = r0 - s1 * s1
            if r1 < 0:
                break
            for j in range(i, len(S_d2)):
                s2 = S_d2[j]
                r2 = r1 - s2 * s2
                if r2 < s2 * s2:
                    break
                if r2 in S_d2_sq:
                    s3 = S_d2_sq[r2]
                    sols.append((s0, s1, s2, s3))
    return sols


def check_thm7_diag_const_sigma3(s, U_S, orbits_S):
    """Exact Stage-0 diag_const check for G = Z_2^2 x S with U = <sigma_3> x U_S.
    - On d = (0, d_S) in S \\ {0} (target -12): D(d) = 4 * D_S(d_S),
      need 4*(2*4*D_S(d_S) - 4s) <= -12  <=>  32*D_S(d_S) <= 16s - 12  <=>  2*D_S(d_S) < s.
    - On d = (c, d_S) in G \\ S (target +4): D(d) = 2 * D_S(d_S) (for d_S != 0) and 2*s (for d_S = 0),
      need 4*(2*D(d) - 4s) <= +4  <=>  D(d) <= 2s (ALWAYS holds since D_S(d_S) <= s!).
    Returns (ok, max_DS, worst_dS)."""
    orb_id = [0] * s
    for idx, o in enumerate(orbits_S):
        for x in o:
            orb_id[x] = idx
    d_reps_S = get_shift_reps_U(s, U_S)
    max_ds = 0
    worst_d = None
    for d in d_reps_S:
        diag = 0
        for x in range(s):
            if orb_id[x] == orb_id[(x + d) % s]:
                diag += 1
        if diag > max_ds:
            max_ds = diag
            worst_d = d
        if 2 * diag >= s:
            return False, diag, d
    return True, max_ds, worst_d


def filter_thm7_prime_q_psd(n, s, U_S, orbits_S, q, row_sols):
    """Exact q-compression PSD filter for odd prime q | s on G = Z_2^2 x S with sigma_3.
    Caches achievable values per unique s_f across all row_sols."""
    U_q = {u % q for u in U_S}
    h = len(U_q)
    e = (q - 1) // h
    if e > 2:
        return row_sols

    by_proj = defaultdict(list)
    for o in orbits_S:
        reps = tuple(sorted({x % q for x in o}))
        w = len(o) // len(reps)
        by_proj[reps].append(w)
        by_proj[reps].append(3 * w)

    c0 = (0,)
    R0 = set(a for a in reachable_sums(by_proj[c0]) if a * a <= n)
    cosets = sorted(c for c in by_proj.keys() if c != c0)
    assert len(cosets) == e

    unique_s = sorted({x for rq in row_sols for x in rq})

    if e == 1:
        R1 = [a for a in reachable_sums(by_proj[cosets[0]]) if (q - 1) * a * a <= n]
        cache = {}
        for sf in unique_s:
            vs = set()
            for a1 in R1:
                a0 = sf - (q - 1) * a1
                if a0 in R0:
                    v = a0 - a1
                    if v * v <= n:
                        vs.add(v * v)
            cache[sf] = sorted(vs)
        return [rq for rq in row_sols if sum_of_4_in_lists([cache[x] for x in rq], n)]

    if e == 2:
        R1 = [a for a in reachable_sums(by_proj[cosets[0]]) if h * a * a <= n]
        R2 = [a for a in reachable_sums(by_proj[cosets[1]]) if h * a * a <= n]
        if q % 4 == 3:
            cache = {}
            for sf in unique_s:
                vs = set()
                for a1 in R1:
                    for a2 in R2:
                        if h * (a1 * a1 + a2 * a2) > n:
                            continue
                        a0 = sf - h * (a1 + a2)
                        if a0 in R0:
                            X = 2 * a0 - a1 - a2
                            Y = a1 - a2
                            v = X * X + q * Y * Y
                            if v <= 4 * n:
                                vs.add(v)
                cache[sf] = sorted(vs)
            return [rq for rq in row_sols if sum_of_4_in_lists([cache[x] for x in rq], 4 * n)]
        else:
            cache = {}
            for sf in unique_s:
                vs = set()
                for a1 in R1:
                    for a2 in R2:
                        if h * (a1 * a1 + a2 * a2) > n:
                            continue
                        a0 = sf - h * (a1 + a2)
                        if a0 in R0:
                            X = 2 * a0 - a1 - a2
                            Y = a1 - a2
                            v = X * X + q * Y * Y
                            if v <= 4 * n:
                                vs.add((v, X * Y))
                cache[sf] = sorted(vs)
            out = []
            for rq in row_sols:
                lists = [cache[x] for x in rq]
                if max(len(x) for x in lists) > 300:
                    out.append(rq)
                elif sum_of_4_tuples(lists, (4 * n, 0)) is not None:
                    out.append(rq)
            return out


def main():
    t0 = time.time()
    eligible = []
    for n in UNKNOWN_85:
        if (n - 12) % 16 == 0:
            v = (n - 12) // 4
            s = v // 4
            eligible.append((n, v, s))

    print(f"=== THEOREM 7 (b=12, k=4, v=4s) ELIGIBLE OPEN ORDERS: {len(eligible)} / 85 ===", flush=True)
    eligible = [(n, v, s) for n, v, s in eligible if n == SPLIT_SIGMA3_ORDER]
    print(f"{len(eligible)} orders satisfy the split sigma_3 condition n = {SPLIT_SIGMA3_ORDER}.", flush=True)
    if not eligible:
        return

    print("\n" + "=" * 90)
    print("PART A: G = Z_2^2 x Z_s with U = <sigma_3> x U_S (r = 2 * r_S <= 46)")
    print("=" * 90, flush=True)

    survivors_A = []
    total_A = 0
    for n, v, s in eligible:
        subs = find_all_subgroups_with_r_le(s, max_r=23)
        for U_S, r_S, is_cyc, gens in subs:
            total_A += 1
            orbits_S = get_orbits_U(s, U_S)
            # Stage 0: diag_const
            ok_d, max_ds, worst_d = check_thm7_diag_const_sigma3(s, U_S, orbits_S)
            if not ok_d:
                continue
            # Stage 1: exact row-sum + Z_2^2 coset-sum feasibility
            orb_sizes_S = [len(o) for o in orbits_S]
            row_sols = check_thm7_sigma3_row_and_coset(n, s, orb_sizes_S)
            if not row_sols:
                continue
            # Stage 2: exact odd prime q | s PSD check (batched)
            odd_primes = sorted(p for p in factor(s) if p > 2)
            surv_rqs = row_sols
            for q in odd_primes:
                surv_rqs = filter_thm7_prime_q_psd(n, s, U_S, orbits_S, q, surv_rqs)
                if not surv_rqs:
                    break
            if not surv_rqs:
                continue
            K_S = len(get_shift_reps_U(s, U_S))
            K = 2 * K_S + 1
            r = 2 * r_S
            survivors_A.append({
                "n": n, "v": v, "s": s, "gens": gens, "is_cyc": is_cyc,
                "U_S": len(U_S), "r_S": r_S, "r": r, "K": K,
                "max_ds": max_ds, "n_rqs": len(surv_rqs), "sample_rqs": surv_rqs[:2],
            })

    survivors_A.sort(key=lambda x: (x["r"], x["K"], x["n"]))
    print(f"\nChecked {total_A} (n, s, U_S) with r = 2*r_S <= 46 | {len(survivors_A)} SURVIVED:")
    for idx, x in enumerate(survivors_A, 1):
        ctype = "cyc" if x["is_cyc"] else "noncyc"
        print(
            f"  #{idx:2d}  n={x['n']:4d}  s={x['s']:4d} ({fmt_fac(x['s']):12s})  "
            f"gens={str(x['gens']):12s} ({ctype:6s}) |U_S|={x['U_S']:3d}  "
            f"r={x['r']:2d}  K={x['K']:2d}  maxD_S={x['max_ds']:3d}/{x['s']//2:3d}  "
            f"rqs={x['n_rqs']:2d} {x['sample_rqs']}"
        )

    print("\n" + "=" * 90)
    print("PART B: ELEMENTARY ABELIAN S = F_{p^a}^+ x Z_{s'} (with G = Z_2^2 x S, sigma_3, r <= 46)")
    print("=" * 90, flush=True)

    survivors_B = []
    for n, v, s in eligible:
        fac = factor(s)
        for p, a in sorted(fac.items()):
            if a < 2:
                continue
            q1 = p ** a
            sp = s // q1
            # Handle sp == 1, sp prime, or sp == 2*prime
            fac_sp = factor(sp)
            if sp == 1:
                sp_mode = "1"
                q2 = 1
            elif len(fac_sp) == 1 and next(iter(fac_sp.values())) == 1:
                sp_mode = "p"
                q2 = sp
            elif len(fac_sp) == 2 and fac_sp.get(2) == 1:
                q2 = [k for k in fac_sp if k != 2][0]
                if fac_sp[q2] != 1:
                    continue
                sp_mode = "2p"
            else:
                continue

            for h1 in divisors(q1 - 1):
                e1 = (q1 - 1) // h1
                if e1 < 2:
                    continue  # fails diag_const on F_{q1}^+
                r1 = 1 + e1
                h2_list = [1] if sp_mode == "1" else divisors(q2 - 1)
                for h2 in h2_list:
                    if sp_mode == "1":
                        e2, r2 = 0, 1
                    elif sp_mode == "p":
                        e2 = (q2 - 1) // h2
                        if e2 < 2 and q2 > 2:
                            continue
                        r2 = 1 + e2
                    else:
                        e2 = (q2 - 1) // h2
                        if e2 < 2:
                            continue
                        r2 = 2 + 2 * e2
                    r_S = r1 * r2
                    r = 2 * r_S
                    if r > 46:
                        continue
                    s1 = [1] + [h1] * e1
                    if sp_mode == "1":
                        s2 = [1]
                    elif sp_mode == "p":
                        s2 = [1] + [h2] * e2
                    else:
                        s2 = [1, 1] + [h2, h2] * e2
                    orb_sizes_S = [x * y for x in s1 for y in s2]
                    row_sols = check_thm7_sigma3_row_and_coset(n, s, orb_sizes_S)
                    if not row_sols:
                        continue
                    # Check exact e1=2 PSD on F_{q1}^+ if q1 odd and e1==2, and e2=2 on Z_{q2}
                    neg1_in_US = ((p == 2) or (h1 % 2 == 0)) and (sp_mode == "1" or q2 == 2 or h2 % 2 == 0)
                    K_S = (r_S - 1) if neg1_in_US else (r_S - 1) // 2
                    K = 2 * K_S + 1
                    survivors_B.append((r, K, n, s, q1, sp, h1, e1, h2, e2, len(row_sols), row_sols[:2]))

    survivors_B.sort()
    print(f"\nFound {len(survivors_B)} feasible elementary-abelian Theorem 7 candidates (r <= 46):")
    for idx, (r, K, n, s, q1, sp, h1, e1, h2, e2, nrq, samp) in enumerate(survivors_B[:25], 1):
        print(
            f"  #{idx:2d}  n={n:4d}  s={s:4d}  S=F_{q1}^+ x Z_{sp:<3d}  "
            f"|U1|={h1:2d}(e1={e1:2d}) x |U2|={h2:2d}(e2={e2:2d})  "
            f"r={r:2d}  K~{K:2d}  rqs={nrq:2d} {samp}"
        )

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
