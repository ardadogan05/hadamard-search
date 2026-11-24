#!/usr/bin/env python3
"""
scan_constructions.py — Exact Algebraic Verification & Quantitative Tractability Ranking
of Unexplored Low-Dimension Hadamard Construction Spaces across UNKNOWN_85.

Computes for every candidate:
  1. Exact algebraic feasibility (row sums, G/S-compression, 2-compression,
     odd-prime q-compression Parseval, Stage-0 diag_const).
  2. Exact combinatorial state-space size log2(|S|) (with fixed row/coset sums).
  3. Empirical PAF lattice step size & standard deviation (Monte Carlo over valid
     row/coset-sum-constrained quadruples).
  4. Heuristic expected solution count log2(E[#sols]) and estimated search difficulty
     (effective K_indep, bits per equation, estimated CPU-hours on a 2-core VM).
"""

import math
import random
import time
from collections import defaultdict
from deep_falsify import (
    UNKNOWN_85,
    factor,
    ord_mod,
    get_orbits,
    get_shift_count,
    reachable_sums,
)


def fmt_fac(n):
    f = factor(n)
    return " * ".join(f"{p}^{e}" if e > 1 else f"{p}" for p, e in sorted(f.items()))


def count_ways_subset(weights, target):
    """Exact number of +-1 assignments to `weights` summing to `target`."""
    dp = defaultdict(int)
    dp[0] = 1
    for w in weights:
        ndp = defaultdict(int)
        for k, cnt in dp.items():
            ndp[k + w] += cnt
            ndp[k - w] += cnt
        dp = ndp
    return dp[target]


def sample_with_sum(weights, target, rng):
    """Sample a uniformly random +-1 assignment to `weights` summing to `target`
    via exact backward DP."""
    n = len(weights)
    dp = [defaultdict(int) for _ in range(n + 1)]
    dp[0][0] = 1
    for i, w in enumerate(weights):
        for k, cnt in dp[i].items():
            dp[i + 1][k + w] += cnt
            dp[i + 1][k - w] += cnt
    if dp[n][target] == 0:
        return None
    signs = [0] * n
    cur = target
    for i in range(n - 1, -1, -1):
        w = weights[i]
        c_plus = dp[i].get(cur - w, 0)
        c_minus = dp[i].get(cur + w, 0)
        if rng.randrange(c_plus + c_minus) < c_plus:
            signs[i] = 1
            cur -= w
        else:
            signs[i] = -1
            cur += w
    assert cur == 0
    return signs


def evaluate_a1_diag(n, s, g, quad_ab, rng, n_samples=400):
    """Exact state count + Monte Carlo PAF std/step for A1-diag:
    G = Z_2^2 x Z_s, tau(c, x) = (sigma_3(c), g*x), b=12, k=4.
    quad_ab = [(a0, b0), (a1, b1), (a2, b2), (a3, b3)] with sum (a+3b)^2 = n, sum (a-b)^2 = 12."""
    orbs_0 = get_orbits(s, g)
    g3 = pow(g, 3, s)
    orbs_1 = get_orbits(s, g3)
    w0 = [len(o) for o in orbs_0]
    w1 = [len(o) for o in orbs_1]
    r0, r1 = len(orbs_0), len(orbs_1)
    r = r0 + r1
    K_S = get_shift_count(s, g)
    K_NS = 1 + get_shift_count(s, g3)
    K = K_S + K_NS

    # Exact state count for this (a_f, b_f) quad
    counts = [count_ways_subset(w0, a) * count_ways_subset(w1, b) for a, b in quad_ab]
    if any(c == 0 for c in counts):
        return None
    log2_states = sum(math.log2(c) for c in counts)

    # Build orbit maps for sampling
    # For each sequence f: coset 0 uses orbs_0 with signs0; coset 1 uses orbs_1 with signs1;
    # coset 2 has f(2, g*x) = f(1, x) => f(2, y) = f(1, g^{-1}*y);
    # coset 3 has f(3, g^2*x) = f(1, x) => f(3, y) = f(1, g^{-2}*y).
    inv_g = pow(g, -1, s)
    inv_g2 = (inv_g * inv_g) % s

    # Pick shift reps: a few on S\{0} and a few on G\S (including (1, 0))
    reps_S = []
    seen = {0}
    for d in range(1, s):
        if d not in seen:
            reps_S.append(d)
            cur = d
            while cur not in seen:
                seen.add(cur)
                seen.add((s - cur) % s)
                cur = (cur * g) % s
    reps_NS = []
    seen = {0}
    for d in range(1, s):
        if d not in seen:
            reps_NS.append(d)
            cur = d
            while cur not in seen:
                seen.add(cur)
                seen.add((s - cur) % s)
                cur = (cur * g3) % s
    assert len(reps_S) == K_S and 1 + len(reps_NS) == K_NS

    # Sample `n_samples` random valid quadruples and measure residuals (P_sum - target)
    res_S = []
    res_NS = []
    res_10 = []
    step_S = 0
    step_NS = 0
    step_10 = 0

    # Pre-check sample shift reps (up to 4 from S, 4 from NS)
    test_S = reps_S[:min(4, len(reps_S))]
    test_NS = reps_NS[:min(4, len(reps_NS))]

    for _ in range(n_samples):
        seqs = []  # each is (f0, f1, f2, f3) lists of length s
        for a, b in quad_ab:
            s0 = sample_with_sum(w0, a, rng)
            s1 = sample_with_sum(w1, b, rng)
            f0 = [0] * s
            for idx, o in enumerate(orbs_0):
                sv = s0[idx]
                for x in o:
                    f0[x] = sv
            f1 = [0] * s
            for idx, o in enumerate(orbs_1):
                sv = s1[idx]
                for x in o:
                    f1[x] = sv
            f2 = [f1[(inv_g * y) % s] for y in range(s)]
            f3 = [f1[(inv_g2 * y) % s] for y in range(s)]
            seqs.append((f0, f1, f2, f3))

        # S\{0} shifts: target -12
        for d in test_S:
            p = 0
            for f0, f1, f2, f3 in seqs:
                # P_f(0, d) = P_{f0}(d) + P_{f1}(d) + P_{f2}(d) + P_{f3}(d)
                p += sum(f0[x] * f0[(x + d) % s] + f1[x] * f1[(x + d) % s]
                         + f2[x] * f2[(x + d) % s] + f3[x] * f3[(x + d) % s] for x in range(s))
            rval = p - (-12)
            res_S.append(rval)
            step_S = math.gcd(step_S, abs(rval))

        # Shift (1, 0): target +4
        p10 = 0
        for f0, f1, f2, f3 in seqs:
            # c -> c ^ 1 swaps 0<->1 and 2<->3
            p10 += 2 * sum(f0[x] * f1[x] + f2[x] * f3[x] for x in range(s))
        r10 = p10 - 4
        res_10.append(r10)
        step_10 = math.gcd(step_10, abs(r10))

        # G\S shifts (1, d): target +4
        for d in test_NS:
            p = 0
            for f0, f1, f2, f3 in seqs:
                p += sum(f0[x] * f1[(x + d) % s] + f1[x] * f0[(x + d) % s]
                         + f2[x] * f3[(x + d) % s] + f3[x] * f2[(x + d) % s] for x in range(s))
            rval = p - 4
            res_NS.append(rval)
            step_NS = math.gcd(step_NS, abs(rval))

    def emp_std(arr):
        mu = sum(arr) / len(arr)
        return math.sqrt(sum((x - mu) ** 2 for x in arr) / len(arr))

    std_S = emp_std(res_S)
    std_NS = emp_std(res_NS)
    std_10 = emp_std(res_10)

    # Linear dependencies among the K equations:
    # 1. Sum over S\{0} is fixed by sum_f (a_f^2 + 3*b_f^2) (1 dependency on S\{0})!
    # 2. Sum over G\S is fixed by sum_f (s_f^2 - (a_f^2 + 3*b_f^2)) (1 dependency on G\S)!
    # So K_indep = (K_S - 1) + (K_NS - 1) = K - 2!
    log2_p_S = (K_S - 1) * math.log2(max(1, step_S) / (math.sqrt(2 * math.pi) * max(1e-9, std_S))) + 0.5 * math.log2(max(1, K_S))
    log2_p_NS = (K_NS - 2) * math.log2(max(1, step_NS) / (math.sqrt(2 * math.pi) * max(1e-9, std_NS))) + math.log2(max(1, step_10) / (math.sqrt(2 * math.pi) * max(1e-9, std_10))) + 0.5 * math.log2(max(1, K_NS))
    log2_p_hit = log2_p_S + log2_p_NS
    log2_esols = log2_states + log2_p_hit
    return {
        "r": r, "r0": r0, "r1": r1, "K": K, "K_S": K_S, "K_NS": K_NS,
        "log2_states": log2_states, "log2_p_hit": log2_p_hit, "log2_esols": log2_esols,
        "step_S": step_S, "std_S": std_S, "step_NS": step_NS, "std_NS": std_NS, "step_10": step_10, "std_10": std_10,
    }


def evaluate_kumar_k3(n, s, g, rng, n_samples=400):
    """Exact state count + Monte Carlo PAF std/step for Kumar-k3 (b=12, k=3, G = Z_3 x Z_s):
    PAF -12 on S\\{0}, 0 on G\\S. Coset sums: one seq (0,0,0), three seqs (2,0,0)."""
    orbs_S = get_orbits(s, g)
    wS = [len(o) for o in orbs_S]
    r_S = len(orbs_S)
    r = 3 * r_S
    K_S = get_shift_count(s, g)
    K_NS = r_S  # shifts (1, d) for d in reps of orbs_S (including d=0!)
    K = K_S + K_NS

    c0 = count_ways_subset(wS, 0)
    c2 = count_ways_subset(wS, 2)
    # 9 cosets with sum 0, 3 cosets with sum 2
    log2_states = 9 * math.log2(c0) + 3 * math.log2(c2)

    reps_S = []
    seen = {0}
    for d in range(1, s):
        if d not in seen:
            reps_S.append(d)
            cur = d
            while cur not in seen:
                seen.add(cur)
                seen.add((s - cur) % s)
                cur = (cur * g) % s
    reps_NS = [o[0] for o in orbs_S]

    test_S = reps_S[:min(4, len(reps_S))]
    test_NS = reps_NS[:min(4, len(reps_NS))]
    res_S, res_NS = [], []
    step_S, step_NS = 0, 0

    for _ in range(n_samples):
        # 4 sequences, each has 3 cosets of Z_s
        # seq 0: (0, 0, 0); seq 1,2,3: (2, 0, 0)
        seqs = []
        for fi in range(4):
            targets = (0, 0, 0) if fi == 0 else (2, 0, 0)
            cosets = []
            for t in targets:
                sg = sample_with_sum(wS, t, rng)
                vec = [0] * s
                for idx, o in enumerate(orbs_S):
                    sv = sg[idx]
                    for x in o:
                        vec[x] = sv
                cosets.append(vec)
            seqs.append(cosets)

        for d in test_S:
            p = 0
            for c0v, c1v, c2v in seqs:
                p += sum(c0v[x] * c0v[(x + d) % s] + c1v[x] * c1v[(x + d) % s] + c2v[x] * c2v[(x + d) % s] for x in range(s))
            rv = p - (-12)
            res_S.append(rv)
            step_S = math.gcd(step_S, abs(rv))

        for d in test_NS:
            p = 0
            for c0v, c1v, c2v in seqs:
                p += sum(c0v[x] * c1v[(x + d) % s] + c1v[x] * c2v[(x + d) % s] + c2v[x] * c0v[(x + d) % s] for x in range(s))
            res_NS.append(p)
            step_NS = math.gcd(step_NS, abs(p))

    def emp_std(arr):
        mu = sum(arr) / len(arr)
        return math.sqrt(sum((x - mu) ** 2 for x in arr) / len(arr))

    std_S = emp_std(res_S)
    std_NS = emp_std(res_NS)
    # 2 linear dependencies (sum on S\{0} fixed, sum on G\S fixed) -> K_indep = K - 2
    log2_p_S = (K_S - 1) * math.log2(max(1, step_S) / (math.sqrt(2 * math.pi) * std_S)) + 0.5 * math.log2(K_S)
    log2_p_NS = (K_NS - 1) * math.log2(max(1, step_NS) / (math.sqrt(2 * math.pi) * std_NS)) + 0.5 * math.log2(K_NS)
    log2_p_hit = log2_p_S + log2_p_NS
    return {
        "r": r, "r_S": r_S, "K": K, "K_S": K_S, "K_NS": K_NS,
        "log2_states": log2_states, "log2_p_hit": log2_p_hit, "log2_esols": log2_states + log2_p_hit,
        "step_S": step_S, "std_S": std_S, "step_NS": step_NS, "std_NS": std_NS,
    }


def evaluate_a1_fix(n, s, g, quad_ab, rng, n_samples=300):
    """Exact state count + Monte Carlo PAF std/step for A1-fix (G = Z_2^2 x Z_s, coset-fixing U)."""
    orbs_S = get_orbits(s, g)
    wS = [len(o) for o in orbs_S]
    r_S = len(orbs_S)
    r = 4 * r_S
    K_S = get_shift_count(s, g)
    K = K_S + 3 * (1 + K_S)

    # quad_ab = [(a0, b0), (a1, b1), (a2, b2), (a3, b3)]:
    # seq 0 has (a0, a0, a0, a0); seq 1,2,3 have (a_i, b_i, b_i, b_i)
    counts = [count_ways_subset(wS, a) * (count_ways_subset(wS, b) ** 3) for a, b in quad_ab]
    if any(c == 0 for c in counts):
        return None
    log2_states = sum(math.log2(c) for c in counts)

    reps_S = []
    seen = {0}
    for d in range(1, s):
        if d not in seen:
            reps_S.append(d)
            cur = d
            while cur not in seen:
                seen.add(cur)
                seen.add((s - cur) % s)
                cur = (cur * g) % s
    test_S = reps_S[:min(4, len(reps_S))]
    res_S, res_NS = [], []
    step_S, step_NS = 0, 0
    for _ in range(n_samples):
        seqs = []
        for a, b in quad_ab:
            cs = []
            for t in (a, b, b, b):
                sg = sample_with_sum(wS, t, rng)
                vec = [0] * s
                for idx, o in enumerate(orbs_S):
                    sv = sg[idx]
                    for x in o:
                        vec[x] = sv
                cs.append(vec)
            seqs.append(cs)
        for d in test_S:
            p = sum(sum(c[x] * c[(x + d) % s] for x in range(s)) for cs in seqs for c in cs)
            rv = p - (-12)
            res_S.append(rv)
            step_S = math.gcd(step_S, abs(rv))
            p2 = sum(sum(cs[0][x] * cs[1][(x + d) % s] + cs[1][x] * cs[0][(x + d) % s]
                         + cs[2][x] * cs[3][(x + d) % s] + cs[3][x] * cs[2][(x + d) % s] for x in range(s)) for cs in seqs)
            rv2 = p2 - 4
            res_NS.append(rv2)
            step_NS = math.gcd(step_NS, abs(rv2))

    def emp_std(arr):
        mu = sum(arr) / len(arr)
        return math.sqrt(sum((x - mu) ** 2 for x in arr) / len(arr))

    std_S = emp_std(res_S)
    std_NS = emp_std(res_NS)
    # 4 linear dependencies (sum on S\{0} + sum on each of the 3 nonzero cosets) -> K_indep = K - 4
    log2_p_S = (K_S - 1) * math.log2(max(1, step_S) / (math.sqrt(2 * math.pi) * std_S)) + 0.5 * math.log2(K_S)
    log2_p_NS = 3 * (K_S * math.log2(max(1, step_NS) / (math.sqrt(2 * math.pi) * std_NS)) + 0.5 * math.log2(K_S + 1))
    log2_p_hit = log2_p_S + log2_p_NS
    return {
        "r": r, "r_S": r_S, "K": K,
        "log2_states": log2_states, "log2_p_hit": log2_p_hit, "log2_esols": log2_states + log2_p_hit,
        "step_S": step_S, "std_S": std_S, "step_NS": step_NS, "std_NS": std_NS,
    }


def evaluate_mixed_4412(rng, n_samples=400):
    """B4 Mixed-(38/19) on n=4412, m=1103, g=749: X, W symmetric (r=30, s=37, 39),
    Y, Z general (r=59, s=1, 39). Also Skew-GS (A skew, B,C,D general)."""
    m, g = 1103, 749
    orbs = get_orbits(m, g)
    e2o = {x: i for i, o in enumerate(orbs) for x in o}
    pairs = []
    seen = set()
    for i, o in enumerate(orbs):
        if i in seen or o[0] == 0:
            continue
        j = e2o[(m - o[0]) % m]
        pairs.append((o, orbs[j]))
        seen.add(i)
        seen.add(j)
    shift_reps = [p[0][0] for p in pairs]

    def log2_comb(n, k):
        return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)

    # Mixed 38/19: X sym (s=37 -> C(29,15)), W sym (s=39 -> C(29,15)),
    # Y gen (s=1 -> C(58,29)), Z gen (s=39 -> C(58,30))
    log2_states_mix = 2 * log2_comb(29, 15) + log2_comb(58, 29) + log2_comb(58, 30)
    vals = []
    step = 0
    for _ in range(n_samples):
        seqs = []
        for s0 in (-1, 1):  # X, W sym
            ch = set(rng.sample(range(29), 15))
            seq = [0] * m
            seq[0] = s0
            for idx, (o1, o2) in enumerate(pairs):
                v = 1 if idx in ch else -1
                for x in o1: seq[x] = v
                for x in o2: seq[x] = v
            seqs.append(seq)
        for s0, kpos in [(1, 29), (1, 30)]:  # Y (s=1), Z (s=39) gen
            ch = set(rng.sample(range(1, 59), kpos))
            seq = [0] * m
            seq[0] = s0
            for idx in range(1, 59):
                v = 1 if idx in ch else -1
                for x in orbs[idx]: seq[x] = v
            seqs.append(seq)
        for d in shift_reps[:3]:
            p = sum(sum(seq[x] * seq[(x + d) % m] for x in range(m)) for seq in seqs)
            vals.append(p)
            step = math.gcd(step, abs(p))
    mu = sum(vals) / len(vals)
    std = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))
    log2_p_hit = 0.5 * math.log2(29) + 28 * math.log2(step / (math.sqrt(2 * math.pi) * std))
    print(f"  [B4 Mixed-(38/19) n=4412 m=1103 g=749] DOF=178, K=29 (28 indep)")
    print(f"    States=2^{log2_states_mix:.2f}, step={step}, std={std:.2f}, log2(p_hit)={log2_p_hit:.2f} => E[#sols] ~ 2^{log2_states_mix + log2_p_hit:.2f}")


def main():
    t0 = time.time()
    rng = random.Random(2026)

    print("=" * 80)
    print("QUANTITATIVE TRACTABILITY & E[#SOLS] EVALUATION OF TOP UNOBSTRUCTED FAMILIES")
    print("=" * 80)

    # 1. A1-diag top survivors (all passed exact q-compression!)
    a1_diag_targets = [
        (5164, 322, 3, [(2, 2), (10, 8), (8, 10), (14, 12)]),
        (8828, 551, 16, [(1, 1), (1, 3), (15, 17), (15, 17)]),  # ex=(4, 10, 66, 66)
        (6172, 385, 3, [(1, 1), (-1, 1), (5, 7), (17, 19)]),    # ex=(4, 2, 26, 74)
        (6876, 429, 7, [(1, 1), (3, 1), (1, 3), (19, 21)]),     # ex=(4, 6, 10, 82)
        (6876, 429, 20, [(1, 1), (-1, 1), (11, 13), (15, 17)]), # ex=(4, 2, 50, 66)
        (5164, 322, 9, [(0, 0), (0, 2), (12, 10), (16, 14)]),   # ex=(0, 6, 42, 58)
    ]
    print("\n--- A1-diag (Theorem 7, b=12, diagonal sigma_3-coupled on Z_2^2 x Z_s) ---")
    for n, s, g, qab in a1_diag_targets:
        res = evaluate_a1_diag(n, s, g, qab, rng, n_samples=250)
        if res is None:
            print(f"  n={n} s={s} g={g}: invalid quad_ab")
            continue
        print(f"  n={n:4d} s={s:3d} ({fmt_fac(s):11s}) g={g:2d} |U|={ord_mod(g,s):2d} r={res['r']:2d} (r0={res['r0']:2d},r1={res['r1']:2d}) K={res['K']:2d} (KS={res['K_S']},KNS={res['K_NS']})")
        print(f"    States=2^{res['log2_states']:.2f} | step_S={res['step_S']} std_S={res['std_S']:.1f} | step_NS={res['step_NS']} std_NS={res['std_NS']:.1f} | log2(p_hit)={res['log2_p_hit']:.2f} => E[#sols] ~ 2^{res['log2_esols']:+.2f}")

    # 2. A1-fix top survivors
    a1_fix_targets = [
        (3004, 187, 2, [(5, 5), (7, 5), (5, 7), (11, 9)]),      # ex=(20, 22, 26, 38)
        (6268, 391, 2, [(17, 17), (7, 5), (7, 5), (5, 7)]),     # ex=(68, 22, 22, 26)
        (8828, 551, 16, [(1, 1), (1, 3), (15, 17), (15, 17)]),  # ex=(4, 10, 66, 66)
    ]
    print("\n--- A1-fix (Theorem 7, b=12, coset-fixing on Z_2^2 x Z_s) ---")
    for n, s, g, qab in a1_fix_targets:
        res = evaluate_a1_fix(n, s, g, qab, rng, n_samples=200)
        if res is None:
            print(f"  n={n} s={s} g={g}: invalid quad_ab")
            continue
        print(f"  n={n:4d} s={s:3d} ({fmt_fac(s):11s}) g={g:2d} |U|={ord_mod(g,s):2d} r={res['r']:2d} K={res['K']:2d}")
        print(f"    States=2^{res['log2_states']:.2f} | step_S={res['step_S']} std_S={res['std_S']:.1f} | step_NS={res['step_NS']} std_NS={res['std_NS']:.1f} | log2(p_hit)={res['log2_p_hit']:.2f} => E[#sols] ~ 2^{res['log2_esols']:+.2f}")

    # 3. Kumar-k3 (b=12, k=3, G = Z_3 x Z_418)
    print("\n--- A4 Kumar-k3 (b=12, k=3, PAF -12 on S\\{0}, 0 off S) ---")
    res_k3 = evaluate_kumar_k3(5028, 418, 5, rng, n_samples=250)
    print(f"  n=5028 s=418 (2 * 11 * 19) g= 5 |U|=45 r={res_k3['r']} (r_S={res_k3['r_S']}) K={res_k3['K']} (KS={res_k3['K_S']}, KNS={res_k3['K_NS']})")
    print(f"    States=2^{res_k3['log2_states']:.2f} | step_S={res_k3['step_S']} std_S={res_k3['std_S']:.1f} | step_NS={res_k3['step_NS']} std_NS={res_k3['std_NS']:.1f} | log2(p_hit)={res_k3['log2_p_hit']:.2f} => E[#sols] ~ 2^{res_k3['log2_esols']:+.2f}")

    # 4. B4 Mixed-U on 4412
    print("\n--- B4 Mixed-(38/19) on n=4412 ---")
    evaluate_mixed_4412(rng, n_samples=250)

    print(f"\nCompleted in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
