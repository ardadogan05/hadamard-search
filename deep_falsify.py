#!/usr/bin/env python3
"""
Deep Algebraic Falsification & Compression Analysis for Open Hadamard Orders.

Checks exact integer obstructions (zero floating-point arithmetic):
  1. Row-sum feasibility (b=4: sum s_f^2 = 4; b=0: sum s_f^2 = n = 4m).
  2. Exact 2-compression (for even m): achievable quadruples (e_f - o_f) with sum d_f^2 = n.
  3. Exact q-compression for EVERY odd prime q | m:
     - Exact L^2-norm (Parseval) feasibility: sum_{f,r} (f^(q)(r))^2 = (sum s_f^2 + (q-1)*n) / q
       computed via exact 2D DP over coset weights.
     - Exact full q-compression PAF / PSD feasibility:
       * e = 1: exact 4-squares sum (a0 - a1)^2 = n.
       * e = 2, q = 3 mod 4: exact sum (X_f^2 + q * Y_f^2) = 4n.
       * e = 2, q = 1 mod 4: exact sum (X_f^2 + q * Y_f^2) = 4n AND sum (X_f * Y_f) = 0.
       * e >= 3: exact full PAF vector meet-in-the-middle check when single-sequence
         achievable tuple count is tractable (plus exact 2q-compression L^2 check).
  4. Full scan of all 85 live UNKNOWN orders (from math74.com/Hadamard/) for both
     Theorem 5 (b=4) and Theorem 3 (b=0, including composite m = n/4) with cyclic
     subgroups U = <g> <= Z_m^* having r <= 36 orbits.
"""

import math
from collections import defaultdict

# Verified live UNKNOWN orders <= 10000 from https://www.math74.com/Hadamard/
UNKNOWN_85 = [
    2396, 2676, 2876, 2956, 3004, 3452, 3628, 3788, 3884, 4076,
    4124, 4156, 4348, 4412, 4460, 4676, 4748, 4792, 4892, 5028,
    5036, 5132, 5164, 5276, 5308, 5468, 5692, 5708, 5748, 5752,
    5756, 5804, 5884, 5892, 5948, 5996, 6172, 6236, 6268, 6284,
    6316, 6332, 6476, 6652, 6668, 6852, 6876, 6988, 7132, 7148,
    7180, 7324, 7388, 7484, 7516, 7572, 7628, 7716, 7876, 7916,
    7924, 7948, 8108, 8156, 8252, 8332, 8348, 8380, 8396, 8684,
    8812, 8828, 8860, 9004, 9148, 9260, 9308, 9356, 9388, 9484,
    9532, 9596, 9692, 9716, 9836,
]


def factor(n):
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def carmichael_lambda(m):
    fac = factor(m)
    lam = 1
    for p, e in fac.items():
        if p == 2 and e >= 3:
            lp = 1 << (e - 2)
        else:
            lp = (p - 1) * (p ** (e - 1))
        lam = math.lcm(lam, lp)
    return lam


def ord_mod(g, q):
    g %= q
    if math.gcd(g, q) != 1:
        return 0
    c, k = g, 1
    while c != 1:
        c = (c * g) % q
        k += 1
    return k


def get_orbits(m, g):
    seen = [False] * m
    orbits = []
    for x in range(m):
        if not seen[x]:
            orb = []
            cur = x
            while not seen[cur]:
                seen[cur] = True
                orb.append(cur)
                cur = (cur * g) % m
            orbits.append(orb)
    return orbits


def get_shift_count(m, g):
    seen = [False] * m
    seen[0] = True
    K = 0
    for d in range(1, m):
        if not seen[d]:
            K += 1
            cur = d
            while not seen[cur]:
                seen[cur] = True
                neg = (m - cur) % m
                seen[neg] = True
                cur = (cur * g) % m
    return K


def reachable_sums(weights):
    """Exact set of values of sum(eps_i * w_i) for eps_i in {-1, +1}."""
    dp = 1
    tot = 0
    for w in weights:
        dp |= (dp << w)
        tot += w
    return sorted(tot - 2 * sub for sub in range(tot + 1) if (dp >> sub) & 1)


def sum_of_4_in_lists(lists, target):
    """Check if target = a + b + c + d with a in lists[0], ..., d in lists[3].
    Uses exact meet-in-the-middle."""
    s01 = {a + b for a in lists[0] for b in lists[1]}
    if not s01:
        return False
    for c in lists[2]:
        for d in lists[3]:
            if (target - c - d) in s01:
                return True
    return False


def sum_of_4_tuples(lists, target):
    """Exact meet-in-the-middle for 4 tuple-valued vectors summing to target."""
    dim = len(target)
    s01 = set()
    for a in lists[0]:
        for b in lists[1]:
            s01.add(tuple(a[i] + b[i] for i in range(dim)))
    if not s01:
        return None
    for c in lists[2]:
        for d in lists[3]:
            need = tuple(target[i] - c[i] - d[i] for i in range(dim))
            if need in s01:
                # Reconstruct one witness
                for a in lists[0]:
                    for b in lists[1]:
                        if tuple(a[i] + b[i] for i in range(dim)) == need:
                            return (a, b, c, d)
    return None


def get_target_row_sum_quadruples(n, m, b, orbits):
    """Return all valid sorted-up-to-symmetry row-sum quadruples (s0, s1, s2, s3)
    with s_f >= 0 achievable from orbit sizes."""
    orb_sizes = [len(o) for o in orbits]
    R_all = set(reachable_sums(orb_sizes))
    if b == 4:
        if m % 2 == 0:
            cand = (2, 0, 0, 0)
            if 2 in R_all and 0 in R_all:
                return [cand]
            return []
        else:
            cand = (1, 1, 1, 1)
            if 1 in R_all:
                return [cand]
            return []
    elif b == 0:
        # sum s_f^2 = n = 4m, s_f >= 0, s_f in R_all
        pos = sorted(x for x in R_all if x >= 0 and x * x <= n)
        pos_set = set(pos)
        quads = []
        for i, a in enumerate(pos):
            a2 = a * a
            if 4 * a2 > n:
                break
            for j in range(i, len(pos)):
                b_val = pos[j]
                ab2 = a2 + b_val * b_val
                if ab2 + 2 * b_val * b_val > n:
                    break
                for k in range(j, len(pos)):
                    c = pos[k]
                    rem = n - ab2 - c * c
                    if rem < c * c:
                        break
                    d = math.isqrt(rem)
                    if d * d == rem and d in pos_set:
                        quads.append((a, b_val, c, d))
        return quads
    else:
        raise ValueError(f"Unsupported b={b}")


def check_2_compression(n, m, orbits, row_sum_quad):
    """Exact 2-compression check when m is even.
    Returns list of achievable quadruples (dA, dB, dC, dD) with sum d_f^2 = n,
    where d_f = e_f - o_f and e_f + o_f = s_f."""
    if m % 2 != 0:
        return None  # Not applicable for odd m
    even_w = [len(o) for o in orbits if o[0] % 2 == 0]
    odd_w = [len(o) for o in orbits if o[0] % 2 == 1]
    R0 = set(reachable_sums(even_w))
    R1 = set(reachable_sums(odd_w))

    # For each target s_f, find all achievable d_f = e_f - o_f with d_f^2 <= n
    # Note: flipping sign of f flips (s_f, d_f) -> (-s_f, -d_f).
    # Fixing s_f >= 0 fixes the sign of f when s_f > 0, so we must keep BOTH signs of d_f;
    # when s_f == 0, d_f and -d_f are symmetric, so we can keep d_f >= 0 (or track |d_f|).
    def achievable_d(s_f):
        vals = []
        lim = math.isqrt(n)
        for d in range(-lim, lim + 1):
            if (s_f + d) % 2 != 0:
                continue
            e_f = (s_f + d) // 2
            o_f = (s_f - d) // 2
            if e_f in R0 and o_f in R1:
                vals.append(d)
        return vals

    s0, s1, s2, s3 = row_sum_quad
    D_lists = [achievable_d(s) for s in row_sum_quad]
    # Collect unique quadruples (d0, d1, d2, d3) up to permutation among equal s_f
    # For clarity, if s_f == 0 we only take d_f >= 0, and sort d_f among equal s_f.
    D_pos_lists = []
    for s, dl in zip(row_sum_quad, D_lists):
        if s == 0:
            D_pos_lists.append(sorted(x for x in dl if x >= 0))
        else:
            D_pos_lists.append(sorted(dl))

    # Map d^2 -> list of d for the 4th coordinate
    d3_by_sq = defaultdict(list)
    for d in D_pos_lists[3]:
        d3_by_sq[d * d].append(d)

    sols = set()
    for d0 in D_pos_lists[0]:
        r0 = n - d0 * d0
        if r0 < 0:
            continue
        for d1 in D_pos_lists[1]:
            if s0 == s1 and d1 < d0:
                continue
            r1 = r0 - d1 * d1
            if r1 < 0:
                continue
            for d2 in D_pos_lists[2]:
                if s1 == s2 and d2 < d1:
                    continue
                r2 = r1 - d2 * d2
                if r2 < 0:
                    continue
                if r2 in d3_by_sq:
                    for d3 in d3_by_sq[r2]:
                        if s2 == s3 and d3 < d2:
                            continue
                        sols.add((d0, d1, d2, d3))
    return sorted(sols)


def check_D_compression_L2(n, m, b, orbits, D, row_sum_quad):
    """Exact D-compression L^2-norm (Parseval) feasibility for any divisor D | m.
    Checks sum_{f, r in Z_D} (f^(D)(r))^2 = T0 where:
      T0 = (sum s_f^2 + (D-1)*n) // D  (= n - 4m/D for b=4, = n for b=0).
    Uses exact 2D DP over the orbits of <g mod D> on Z_D.
    Returns (ok, details_dict)."""
    sum_s2 = sum(s * s for s in row_sum_quad)
    num = sum_s2 + (D - 1) * n
    assert num % D == 0
    T0 = num // D

    # Group orbits of <g> on Z_m by their projection mod D
    by_proj = defaultdict(list)
    for o in orbits:
        reps = tuple(sorted({x % D for x in o}))
        assert len(o) % len(reps) == 0
        by_proj[reps].append(len(o) // len(reps))

    # Separate C_0 = (0,) if present, and all other cosets
    cosets = sorted(by_proj.keys())
    R_map = {}
    for c in cosets:
        c_size = len(c)
        R_full = reachable_sums(by_proj[c])
        # Prune by c_size * a^2 <= T0
        R_pruned = [a for a in R_full if c_size * a * a <= T0]
        R_map[c] = R_pruned

    # Split cosets into c0 = (0,) and rest
    c0 = (0,)
    rest = [c for c in cosets if c != c0]
    R0_set = set(R_map[c0])

    # DP over rest: track achievable (weighted_sum, weighted_sq_sum) with weighted_sq_sum <= T0
    # Wait: weighted_sum = sum_{c in rest} |c| * a_c, weighted_sq_sum = sum_{c in rest} |c| * a_c^2
    dp = {(0, 0)}
    for c in rest:
        csz = len(c)
        vals = R_map[c]
        new_dp = set()
        for ws, wsq in dp:
            for a in vals:
                nwsq = wsq + csz * a * a
                if nwsq <= T0:
                    new_dp.add((ws + csz * a, nwsq))
        dp = new_dp
        if not dp:
            break

    # For each s_f in row_sum_quad, collect achievable P_f^(D)(0) = a0^2 + wsq
    # where a0 = s_f - ws in R0_set and a0^2 + wsq <= T0
    ach_P0 = {}
    for s_f in set(row_sum_quad):
        p0s = set()
        for ws, wsq in dp:
            a0 = s_f - ws
            if a0 in R0_set:
                val = a0 * a0 + wsq
                if val <= T0:
                    p0s.add(val)
        ach_P0[s_f] = sorted(p0s)

    lists = [ach_P0[s] for s in row_sum_quad]
    ok = sum_of_4_in_lists(lists, T0)
    max_single = {s: (max(ach_P0[s]) if ach_P0[s] else None) for s in set(row_sum_quad)}
    return ok, {
        "D": D,
        "T0": T0,
        "num_cosets": len(cosets),
        "max_single_P0": max_single,
        "num_P0_vals": {s: len(ach_P0[s]) for s in set(row_sum_quad)},
    }


def check_prime_q_compression_exact(n, m, g, b, orbits, q, row_sum_quad, max_tuples=40000):
    """Exact q-compression PSD/PAF check for an odd prime q | m.
    1. Always runs exact L^2-norm (Parseval) DP check.
    2. For e = 1: exact 4-squares sum (a0 - a1)^2 = n.
    3. For e = 2:
       - q = 3 mod 4: exact sum (X_f^2 + q*Y_f^2) = 4n.
       - q = 1 mod 4: exact sum (X_f^2 + q*Y_f^2) = 4n AND sum (X_f * Y_f) = 0.
    4. For e >= 3: enumerates all valid (a_0, a_1, ..., a_e) with P_f^(q)(0) <= T0
       (if count <= max_tuples) and runs exact PAF vector meet-in-the-middle!
    Returns (ok, status_str, details_dict)."""
    h = ord_mod(g, q)
    e = (q - 1) // h
    sum_s2 = sum(s * s for s in row_sum_quad)
    T0 = (sum_s2 + (q - 1) * n) // q
    Td = (sum_s2 - n) // q  # -4m/q for b=4, 0 for b=0

    # Step 1: Exact L^2-norm check
    ok_l2, l2_info = check_D_compression_L2(n, m, b, orbits, q, row_sum_quad)
    if not ok_l2:
        return False, f"FALSIFIED by q={q} (e={e}, h={h}) exact L2-norm obstruction (T0={T0}, max_single={l2_info['max_single_P0']})", l2_info

    # Partition orbits by projection mod q
    by_proj = defaultdict(list)
    for o in orbits:
        reps = tuple(sorted({x % q for x in o}))
        by_proj[reps].append(len(o) // len(reps))

    c0 = (0,)
    R0 = set(a for a in reachable_sums(by_proj[c0]) if a * a <= T0)
    cosets = sorted(c for c in by_proj.keys() if c != c0)
    assert len(cosets) == e
    R_cosets = [
        sorted(a for a in reachable_sums(by_proj[c]) if h * a * a <= T0)
        for c in cosets
    ]

    # Step 2: e = 1 (primitive root mod q)
    if e == 1:
        R1 = R_cosets[0]
        def sq_vals(s_f):
            vals = set()
            for a1 in R1:
                a0 = s_f - (q - 1) * a1
                if a0 in R0:
                    v = a0 - a1
                    if v * v <= n:
                        vals.add(v * v)
            return sorted(vals)
        lists = [sq_vals(s) for s in row_sum_quad]
        if not sum_of_4_in_lists(lists, n):
            return False, f"FALSIFIED by q={q} (e=1, h={h}) exact 4-squares PSD obstruction", l2_info
        return True, f"passed q={q} (e=1, h={h}, exact PSD)", l2_info

    # Step 3: e = 2 (quadratic residues / non-residues)
    if e == 2:
        R1, R2 = R_cosets[0], R_cosets[1]
        if q % 4 == 3:
            def norm_vals(s_f):
                vals = set()
                for a1 in R1:
                    for a2 in R2:
                        a0 = s_f - h * (a1 + a2)
                        if a0 in R0:
                            X = 2 * a0 - a1 - a2
                            Y = a1 - a2
                            v = X * X + q * Y * Y
                            if v <= 4 * n:
                                vals.add(v)
                return sorted(vals)
            lists = [norm_vals(s) for s in row_sum_quad]
            if not sum_of_4_in_lists(lists, 4 * n):
                return False, f"FALSIFIED by q={q} (e=2, q=3 mod 4) exact quadratic norm obstruction", l2_info
            return True, f"passed q={q} (e=2, q=3 mod 4, exact norm)", l2_info
        else:
            # q = 1 mod 4: BOTH sum(X^2 + q*Y^2) = 4n AND sum(X*Y) = 0!
            def pair_vals(s_f):
                vals = set()
                for a1 in R1:
                    for a2 in R2:
                        a0 = s_f - h * (a1 + a2)
                        if a0 in R0:
                            X = 2 * a0 - a1 - a2
                            Y = a1 - a2
                            v1 = X * X + q * Y * Y
                            if v1 <= 4 * n:
                                vals.add((v1, X * Y))
                return sorted(vals)
            lists = [pair_vals(s) for s in row_sum_quad]
            if sum_of_4_tuples(lists, (4 * n, 0)) is None:
                return False, f"FALSIFIED by q={q} (e=2, q=1 mod 4) exact real-quadratic PSD obstruction", l2_info
            return True, f"passed q={q} (e=2, q=1 mod 4, exact PSD)", l2_info

    # Step 4: e >= 3 — check if full exact PAF vector enumeration is tractable
    # Estimate / enumerate valid tuples (a_1, ..., a_e) with h * sum a_i^2 <= T0
    # First build shift representatives of F_q^* / (+- U_bar)
    seen_d = [False] * q
    seen_d[0] = True
    d_reps = []
    for d in range(1, q):
        if not seen_d[d]:
            d_reps.append(d)
            cur = d
            while not seen_d[cur]:
                seen_d[cur] = True
                seen_d[(q - cur) % q] = True
                cur = (cur * g) % q

    # Precompute cyclotomic incidence matrices for each d_rep
    # Map each r in Z_q to coset index 0..e (0 is {0}, 1..e are cosets[0..e-1])
    res_to_cid = [0] * q
    for idx, c in enumerate(cosets, start=1):
        for r in c:
            res_to_cid[r] = idx

    # For each shift d in d_reps, count pairs (i, j) with x in C_i, (x+d)%q in C_j
    num_c = e + 1
    paf_mats = []
    for d in d_reps:
        mat = [[0] * num_c for _ in range(num_c)]
        for x in range(q):
            ci = res_to_cid[x]
            cj = res_to_cid[(x + d) % q]
            mat[ci][cj] += 1
        paf_mats.append(mat)

    # Enumerate valid (a_1, ..., a_e) with h * sum a_i^2 <= T0 up to max_tuples
    # Use DFS with suffix min bound
    tuples_by_s = {s: set() for s in set(row_sum_quad)}
    target_s_set = set(row_sum_quad)
    aborted = False
    count_checked = 0

    def dfs(idx, cur_sum, cur_sq, cur_a):
        nonlocal aborted, count_checked
        if aborted:
            return
        if idx == e:
            for s_f in target_s_set:
                a0 = s_f - h * cur_sum
                if a0 in R0:
                    p0 = a0 * a0 + h * cur_sq
                    if p0 <= T0:
                        count_checked += 1
                        if count_checked > max_tuples:
                            aborted = True
                            return
                        avec = (a0,) + tuple(cur_a)
                        # Compute exact PAF vector (p0, P(d_1), ..., P(d_{k-1}))
                        # Note last d_rep is redundant given s_f^2 and p0, so drop last if len(d_reps) >= 1
                        pvec = [p0]
                        for mat in paf_mats[:-1]:
                            val = 0
                            for i in range(num_c):
                                ai = avec[i]
                                if ai == 0:
                                    continue
                                row = mat[i]
                                for j in range(num_c):
                                    if row[j]:
                                        val += row[j] * ai * avec[j]
                            pvec.append(val)
                        tuples_by_s[s_f].add(tuple(pvec))
            return
        for a in R_cosets[idx]:
            nsq = cur_sq + a * a
            if h * nsq <= T0:
                cur_a.append(a)
                dfs(idx + 1, cur_sum + a, nsq, cur_a)
                cur_a.pop()
                if aborted:
                    return

    dfs(0, 0, 0, [])
    if not aborted:
        target_vec = (T0,) + (Td,) * (len(d_reps) - 1)
        lists = [sorted(tuples_by_s[s]) for s in row_sum_quad]
        # Only run meet-in-the-middle if pair set is reasonable
        max_len = max(len(x) for x in lists)
        if max_len <= 1500:
            wit = sum_of_4_tuples(lists, target_vec)
            if wit is None:
                return False, f"FALSIFIED by q={q} (e={e}, h={h}) exact full PAF vector obstruction", l2_info
            return True, f"passed q={q} (e={e}, h={h}, exact full PAF, |V|={[len(x) for x in lists]})", l2_info
        else:
            return True, f"passed q={q} (e={e}, h={h}, exact L2; full PAF skipped |V|={max_len})", l2_info
    return True, f"passed q={q} (e={e}, h={h}, exact L2; tuple space > {max_tuples})", l2_info


def check_4_compression_exact(n, m, g, b, orbits, row_sum_quad):
    """Exact 4-compression simultaneous (k=m/2, k=m/4) PSD check when 4 | m."""
    if m % 4 != 0:
        return True, "N/A"
    sum_s2 = sum(s * s for s in row_sum_quad)
    T0 = (sum_s2 + 3 * n) // 4
    by_m4 = defaultdict(list)
    for o in orbits:
        reps = tuple(sorted({x % 4 for x in o}))
        by_m4[reps].append(len(o) // len(reps))
    if g % 4 == 3:
        R0 = set(a for a in reachable_sums(by_m4[(0,)]) if a * a <= T0)
        R2 = set(a for a in reachable_sums(by_m4[(2,)]) if a * a <= T0)
        R13 = set(a for a in reachable_sums(by_m4[(1, 3)]) if 2 * a * a <= T0)
        def get_pairs(s_f):
            ps = set()
            for a1 in R13:
                e_f = s_f - 2 * a1
                d2 = e_f - 2 * a1
                if d2 * d2 > n:
                    continue
                for a0 in R0:
                    a2 = e_f - a0
                    if a2 in R2 and a0 * a0 + a2 * a2 + 2 * a1 * a1 <= T0:
                        d4 = a0 - a2
                        if d4 * d4 <= n:
                            ps.add((d2 * d2, d4 * d4))
            return sorted(ps)
        lists = [get_pairs(s) for s in row_sum_quad]
        wit = sum_of_4_tuples(lists, (n, n))
        if wit is None:
            return False, "FALSIFIED by 4-compression exact (m/2, m/4) PSD obstruction (g=3 mod 4)"
        return True, "passed 4-comp (exact (m/2, m/4) PSD)"
    else:
        R = {k: [a for a in reachable_sums(v) if a * a <= T0] for k, v in by_m4.items()}
        even_map = defaultdict(set)
        for a0 in R[(0,)]:
            for a2 in R[(2,)]:
                if a0 * a0 + a2 * a2 <= T0:
                    even_map[a0 + a2].add((a0 - a2) ** 2)
        odd_map = defaultdict(set)
        for a1 in R[(1,)]:
            for a3 in R[(3,)]:
                if a1 * a1 + a3 * a3 <= T0:
                    odd_map[a1 + a3].add((a1 - a3) ** 2)
        def get_pairs(s_f):
            ps = set()
            for ef, e_sq in even_map.items():
                of = s_f - ef
                if of not in odd_map:
                    continue
                d2 = ef - of
                if d2 * d2 > n:
                    continue
                for sq1 in e_sq:
                    for sq2 in odd_map[of]:
                        if sq1 + sq2 <= n:
                            ps.add((d2 * d2, sq1 + sq2))
            return sorted(ps)
        lists = [get_pairs(s) for s in row_sum_quad]
        wit = sum_of_4_tuples(lists, (n, n))
        if wit is None:
            return False, "FALSIFIED by 4-compression exact (m/2, m/4) PSD obstruction (g=1 mod 4)"
        return True, "passed 4-comp (exact (m/2, m/4) PSD)"


def check_2q_simultaneous_e2(n, m, g, b, orbits, q, row_sum_quad):
    """Exact simultaneous (k=m/2, k=m/q, k=(q+2)m/(2q)) PSD check when 2q | m and e=2."""
    if m % 2 != 0:
        return True, "N/A"
    h = ord_mod(g, q)
    if (q - 1) // h != 2:
        return True, "N/A"
    sum_s2 = sum(s * s for s in row_sum_quad)
    T0 = (sum_s2 + (2 * q - 1) * n) // (2 * q)
    qrs = {(x * x) % q for x in range(1, q)}
    buckets = defaultdict(list)
    for o in orbits:
        par = o[0] % 2
        rem = o[0] % q
        cls = 0 if rem == 0 else (1 if rem in qrs else 2)
        csz = 1 if cls == 0 else h
        buckets[(par, cls)].append(len(o) // csz)
    Rf = {}
    for (par, cls), wts in buckets.items():
        csz = 1 if cls == 0 else h
        Rf[(par, cls)] = sorted(v for v in reachable_sums(wts) if csz * v * v <= T0)

    even_by_s = defaultdict(list)
    for e0 in Rf[(0, 0)]:
        for e1 in Rf[(0, 1)]:
            if e0 * e0 + h * e1 * e1 > T0:
                continue
            for e2 in Rf[(0, 2)]:
                if e0 * e0 + h * (e1 * e1 + e2 * e2) <= T0:
                    even_by_s[e0 + h * (e1 + e2)].append((e0, e1, e2))
    odd_by_s = defaultdict(list)
    for o0 in Rf[(1, 0)]:
        for o1 in Rf[(1, 1)]:
            if o0 * o0 + h * o1 * o1 > T0:
                continue
            for o2 in Rf[(1, 2)]:
                if o0 * o0 + h * (o1 * o1 + o2 * o2) <= T0:
                    odd_by_s[o0 + h * (o1 + o2)].append((o0, o1, o2))

    q3 = (q % 4 == 3)
    V = {}
    for sf in set(row_sum_quad):
        vs = set()
        for es, elist in even_by_s.items():
            os_ = sf - es
            if os_ not in odd_by_s:
                continue
            d2 = es - os_
            if d2 * d2 > n:
                continue
            olist = odd_by_s[os_]
            for e0, e1, e2 in elist:
                esq = e0 * e0 + h * (e1 * e1 + e2 * e2)
                for o0, o1, o2 in olist:
                    if esq + o0 * o0 + h * (o1 * o1 + o2 * o2) > T0:
                        continue
                    X1 = 2 * (e0 + o0) - (e1 + o1 + e2 + o2)
                    Y1 = (e1 + o1) - (e2 + o2)
                    pq = X1 * X1 + q * Y1 * Y1
                    if pq > 4 * n:
                        continue
                    X2 = 2 * (e0 - o0) - (e1 - o1 + e2 - o2)
                    Y2 = (e1 - o1) - (e2 - o2)
                    p2q = X2 * X2 + q * Y2 * Y2
                    if p2q > 4 * n:
                        continue
                    if q3:
                        vs.add((d2 * d2, pq, p2q))
                    else:
                        vs.add((d2 * d2, pq, X1 * Y1, p2q, X2 * Y2))
        V[sf] = sorted(vs)

    lists = [V[s] for s in row_sum_quad]
    target = (n, 4 * n, 4 * n) if q3 else (n, 4 * n, 0, 4 * n, 0)
    if max(len(x) for x in lists) > 2500:
        return True, f"passed 2q={2*q} (simultaneous check skipped |V|={[len(x) for x in lists]})"
    wit = sum_of_4_tuples(lists, target)
    if wit is None:
        return False, f"FALSIFIED by 2q={2*q} exact simultaneous (2, {q}, {2*q}) PSD obstruction (|V|={[len(x) for x in lists]})"
    return True, f"passed 2q={2*q} (exact simultaneous (2, {q}, {2*q}) PSD, |V|={[len(x) for x in lists]})"


def full_falsify_candidate(n, m, g, b, verbose=False):
    """Run all exact algebraic checks on (n, m, g, b).
    Returns (passed, summary_dict)."""
    orbits = get_orbits(m, g)
    r = len(orbits)
    K = get_shift_count(m, g)
    U_size = ord_mod(g, m)

    row_quads = get_target_row_sum_quadruples(n, m, b, orbits)
    if not row_quads:
        return False, {
            "n": n, "m": m, "g": g, "b": b, "U": U_size, "r": r, "K": K,
            "reason": "FALSIFIED: no achievable row-sum quadruple",
        }

    fac = factor(m)
    odd_primes = sorted(p for p in fac if p > 2)

    surviving_quads = []
    quad_reports = []
    for rq in row_quads:
        rep = {"row_sums": rq, "checks": []}
        ok_rq = True

        # 1. Exact 2-compression
        if m % 2 == 0:
            sols_2 = check_2_compression(n, m, orbits, rq)
            rep["2_comp_sols"] = sols_2
            if not sols_2:
                ok_rq = False
                rep["fail_reason"] = "FALSIFIED by 2-compression (no achievable (e-o) quadruple)"
            else:
                rep["checks"].append(f"2-comp: {len(sols_2)} achievable quadruples")

        # 1b. Exact 4-compression when 4 | m
        if ok_rq and m % 4 == 0:
            ok_4, msg_4 = check_4_compression_exact(n, m, g, b, orbits, rq)
            rep["checks"].append(msg_4)
            if not ok_4:
                ok_rq = False
                rep["fail_reason"] = msg_4

        # 2. Exact q-compression for every odd prime q | m
        if ok_rq:
            for q in odd_primes:
                ok_q, msg_q, info_q = check_prime_q_compression_exact(n, m, g, b, orbits, q, rq)
                rep["checks"].append(msg_q)
                if not ok_q:
                    ok_rq = False
                    rep["fail_reason"] = msg_q
                    break

        # 3. Exact 2q-compression (L^2-norm + simultaneous (2, q, 2q) PSD when e=2)
        if ok_rq and m % 2 == 0:
            for q in odd_primes:
                ok_2q, info_2q = check_D_compression_L2(n, m, b, orbits, 2 * q, rq)
                if not ok_2q:
                    ok_rq = False
                    msg_2q = f"FALSIFIED by 2q={2*q} exact L2-norm obstruction (T0={info_2q['T0']}, max_single={info_2q['max_single_P0']})"
                    rep["checks"].append(msg_2q)
                    rep["fail_reason"] = msg_2q
                    break
                ok_sim, msg_sim = check_2q_simultaneous_e2(n, m, g, b, orbits, q, rq)
                if msg_sim != "N/A":
                    rep["checks"].append(msg_sim)
                    if not ok_sim:
                        ok_rq = False
                        rep["fail_reason"] = msg_sim
                        break
                else:
                    rep["checks"].append(f"passed 2q={2*q} (exact L2)")

        quad_reports.append(rep)
        if ok_rq:
            surviving_quads.append(rep)

    if surviving_quads:
        return True, {
            "n": n, "m": m, "g": g, "b": b, "U": U_size, "r": r, "K": K,
            "surviving_quads": surviving_quads,
            "all_reports": quad_reports,
        }
    else:
        return False, {
            "n": n, "m": m, "g": g, "b": b, "U": U_size, "r": r, "K": K,
            "reason": quad_reports[0].get("fail_reason", "FALSIFIED"),
            "all_reports": quad_reports,
        }


def find_cyclic_subgroups_with_r_le(m, max_r=36):
    """Find all distinct cyclic subgroups U = <g> <= Z_m^* with r <= max_r orbits."""
    lam = carmichael_lambda(m)
    # Necessary condition: r >= 1 + (m - 1) / |U| => |U| >= (m - 1) / (max_r - 1)
    min_u = (m - 1 + max_r - 2) // (max_r - 1)
    if lam < min_u:
        return []

    seen_subgroups = {}
    for g in range(1, m):
        if math.gcd(g, m) != 1:
            continue
        # Generate subgroup
        elems = []
        cur = 1
        while True:
            elems.append(cur)
            cur = (cur * g) % m
            if cur == 1:
                break
        if len(elems) < min_u:
            continue
        key = frozenset(elems)
        if key in seen_subgroups:
            continue
        # Burnside's lemma for exact orbit count r:
        # r = (1/|U|) * sum_{u in U} gcd(u - 1, m)
        r = sum(math.gcd(u - 1, m) for u in elems) // len(elems)
        if r <= max_r:
            seen_subgroups[key] = (g, len(elems), r)
    return sorted(seen_subgroups.values(), key=lambda x: (x[2], -x[1], x[0]))


def main():
    print("=" * 80)
    print("PART 1: DEEP ALGEBRAIC FALSIFICATION OF TOP 3 OPEN HADAMARD ORDERS (b=4)")
    print("=" * 80)
    top3 = [
        (3788, 946, 47, 4),
        (4892, 1222, 3, 4),
        (5708, 1426, 35, 4),
    ]
    for n, m, g, b in top3:
        fac_str = " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factor(m).items()))
        passed, info = full_falsify_candidate(n, m, g, b, verbose=True)
        status = "SURVIVES ALL CHECKS" if passed else "FALSIFIED"
        print(f"\n>>> n={n}, m={m} ({fac_str}), g={g}, |U|={info['U']}, r={info['r']}, K={info['K']} -> {status}")
        for rep in info.get("all_reports", []):
            print(f"    Row sums: {rep['row_sums']}")
            if "2_comp_sols" in rep:
                sols = rep["2_comp_sols"]
                print(f"    Exact 2-compression achievable (eA-oA, eB-oB, eC-oC, eD-oD) [{len(sols)}]:")
                for s in sols:
                    print(f"      {s}  (sum sq = {sum(x*x for x in s)})")
            for chk in rep["checks"]:
                print(f"    - {chk}")
            if "fail_reason" in rep:
                print(f"    *** {rep['fail_reason']} ***")

    print("\n" + "=" * 80)
    print("PART 2: FULL SCAN OF ALL 85 LIVE UNKNOWN ORDERS (b=4 AND b=0, r <= 36)")
    print("=" * 80)
    survivors_b4 = []
    survivors_b0 = []
    falsified_count = {4: 0, 0: 0}
    total_checked = {4: 0, 0: 0}

    for b, label in [(4, "Theorem 5 (b=4)"), (0, "Theorem 3 (b=0)")]:
        print(f"\n--- Scanning {label} across 85 live UNKNOWN orders (r <= 36) ---")
        for n in UNKNOWN_85:
            m = (n - b) // 4
            subgroups = find_cyclic_subgroups_with_r_le(m, max_r=36)
            for g, u_sz, r in subgroups:
                total_checked[b] += 1
                passed, info = full_falsify_candidate(n, m, g, b)
                if passed:
                    if b == 4:
                        survivors_b4.append(info)
                    else:
                        survivors_b0.append(info)
                else:
                    falsified_count[b] += 1

    for b, surv, label in [
        (4, survivors_b4, "Theorem 5 (b=4)"),
        (0, survivors_b0, "Theorem 3 (b=0)"),
    ]:
        print(f"\n=== {label}: {total_checked[b]} cyclic (n,m,<g>) with r<=36 checked | "
              f"{falsified_count[b]} FALSIFIED | {len(surv)} SURVIVED ===")
        surv.sort(key=lambda x: (x["K"], x["r"], x["n"]))
        for info in surv:
            fac_str = "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factor(info["m"]).items()))
            n_rq = len(info["surviving_quads"])
            n_2c = sum(len(q.get("2_comp_sols", [])) for q in info["surviving_quads"]) if info["m"] % 2 == 0 else "N/A"
            print(f"  n={info['n']:4d}  m={info['m']:4d} ({fac_str:14s})  g={info['g']:4d}  "
                  f"|U|={info['U']:3d}  r={info['r']:2d}  K={info['K']:2d}  "
                  f"row_quads={n_rq}  2-comp={n_2c}")


if __name__ == "__main__":
    main()
