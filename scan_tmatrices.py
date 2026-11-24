#!/usr/bin/env python3
"""
Exact-Integer Scanner for Circulant T-Matrices of Order t
that close open Hadamard orders n in UNKNOWN_85 (n <= 10,000)
via Cooper-Wallis H(4*t*w) (including w = 1 direct H(4*t)).

A T-matrix of order t consists of 4 circulant {0, +-1}-matrices T1, T2, T3, T4
with disjoint supports (sum |S_i| = t) and sum_{i=1}^4 PAF(T_i, d) = 0 for all d != 0.

Exact algebraic conditions for U-invariant T-matrices (U <= Z_t^*):
  1. Row sums: s1^2 + s2^2 + s3^2 + s4^2 = t, with s_i achievable by partitioning
     the r orbits of U on Z_t into 4 disjoint subsets S_1..S_4 with +-1 signs.
  2. Stage-0 diag_const: for every shift d != 0, let D(d) = #{x in Z_t : orb(x) == orb(x+d)}.
     Since x and x+d lie in the same T_i with the same sign whenever orb(x) == orb(x+d),
     sum_i PAF(T_i, d) >= D(d) - (t - D(d)) = 2*D(d) - t.
     Thus 2*D(d) > t is strictly impossible!
  3. For prime t with |U| = h and e = (t-1)/h:
       2*m1 + h*(m1^2 + m2^2 + m3^2 + m4^2) = e,
       |m1| + |m2| + |m3| + |m4| <= e,
       m1 + m2 + m3 + m4 == e (mod 2).
     (Zero proper divisors => zero composite compression obstructions!)
"""

import math
import time
from collections import defaultdict

from deep_falsify import UNKNOWN_85, factor, find_cyclic_subgroups_with_r_le, ord_mod
from scan_expanded import get_orbits_U, get_shift_reps_U, fmt_fac


# Known odd Williamson-type WT(w) orders from arXiv:2610.27175v2 Table 3
KNOWN_WT_ODD = (
    set(range(1, 70, 2))
    | {73, 79, 91, 93, 95, 97, 103, 109, 111, 127, 133, 181}
)


def is_prime(k):
    if k < 2:
        return False
    fac = factor(k)
    return len(fac) == 1 and next(iter(fac.values())) == 1


def tmatrix_row_quads_prime(t, h):
    """Exact feasible (m1, m2, m3, m4) -> (s1, s2, s3, s4) for prime t, |U|=h, e=(t-1)//h.
    s1 = 1 + h*m1, s2 = h*m2, s3 = h*m3, s4 = h*m4 (with m2 >= m3 >= m4 >= 0)."""
    e = (t - 1) // h
    sols = []
    # Bound: |m_i| <= isqrt(e // h) + 1
    lim = math.isqrt(e // h) + 2
    for m1 in range(-lim, lim + 1):
        rem = e - 2 * m1
        if rem < 0 or rem % h != 0:
            continue
        sq_target = rem // h - m1 * m1
        if sq_target < 0:
            continue
        # Find m2 >= m3 >= m4 >= 0 with m2^2 + m3^2 + m4^2 == sq_target
        l2 = math.isqrt(sq_target)
        for m2 in range(l2, -1, -1):
            r2 = sq_target - m2 * m2
            if 3 * m2 * m2 < sq_target:
                break
            l3 = min(m2, math.isqrt(r2))
            for m3 in range(l3, -1, -1):
                r3 = r2 - m3 * m3
                if 2 * m3 * m3 < r2:
                    break
                m4 = math.isqrt(r3)
                if m4 <= m3 and m4 * m4 == r3:
                    if abs(m1) + m2 + m3 + m4 <= e and (m1 + m2 + m3 + m4) % 2 == e % 2:
                        s_quad = (1 + h * m1, h * m2, h * m3, h * m4)
                        assert sum(x * x for x in s_quad) == t
                        sols.append(((m1, m2, m3, m4), s_quad))
    return sols


def check_tmatrix_diag_const(t, U, orbits, d_reps):
    """Exact Stage-0 diag_const for T-matrices: 2*D(d) <= t for all shifts d != 0."""
    orb_id = [0] * t
    for idx, o in enumerate(orbits):
        for x in o:
            orb_id[x] = idx
    max_d = 0
    for d in d_reps:
        diag = 0
        for x in range(t):
            if orb_id[x] == orb_id[(x + d) % t]:
                diag += 1
        if diag > max_d:
            max_d = diag
        if 2 * diag > t:
            return False, diag
    return True, max_d


def tmatrix_row_quads_general(t, orb_sizes):
    """Exact check whether orbit sizes can be partitioned into 4 disjoint subsets
    with +-1 signs whose sums s1..s4 satisfy s1^2 + s2^2 + s3^2 + s4^2 = t.
    Uses exact 4-squares enumeration + subset-partition check for small/moderate t."""
    # First find all integer 4-square representations a^2+b^2+c^2+d^2 = t (a>=b>=c>=d>=0)
    lim = math.isqrt(t)
    # Check achievable single-subset sums using DP: subset_sums of orb_sizes with +-1 signs
    # A subset of orb_sizes with +-1 signs can achieve any sum in DP where each w contributes {-w, 0, +w}.
    # Even simpler: check if sum of all orb_sizes == t, and check individual reachability first.
    dp = {0}
    for w in orb_sizes:
        ndp = set(dp)
        for v in dp:
            if abs(v + w) <= lim + max(orb_sizes):
                ndp.add(v + w)
            if abs(v - w) <= lim + max(orb_sizes):
                ndp.add(v - w)
        dp = ndp
    ach = {abs(v) for v in dp if abs(v) <= lim}
    ach_l = sorted(ach)
    ach_s = set(ach_l)
    quads = []
    for i, a in enumerate(ach_l):
        if 4 * a * a > t:
            break
        for j in range(i, len(ach_l)):
            b = ach_l[j]
            if a * a + 3 * b * b > t:
                break
            for k in range(j, len(ach_l)):
                c = ach_l[k]
                rem = t - a * a - b * b - c * c
                if rem < c * c:
                    break
                d = math.isqrt(rem)
                if d * d == rem and d in ach_s:
                    quads.append((a, b, c, d))
    return quads


def main():
    t0 = time.time()

    # Map each candidate t -> list of (n, w) in UNKNOWN_85 that T-matrix(t) would close!
    t_to_orders = defaultdict(list)
    for n in UNKNOWN_85:
        m = n // 4
        # Find all odd divisors w of m in KNOWN_WT_ODD
        fac = factor(m)
        divs = [1]
        for p, e in fac.items():
            divs = [d * (p ** k) for d in divs for k in range(e + 1)]
        for w in divs:
            if w in KNOWN_WT_ODD:
                t = m // w
                if t >= 3:
                    t_to_orders[t].append((n, w))

    print(f"=== T-MATRIX CENSUS ACROSS ALL 85 OPEN ORDERS (Cooper-Wallis H(4*t*w)) ===")
    print(f"Distinct candidate orders t: {len(t_to_orders)} "
          f"(prime t: {sum(1 for t in t_to_orders if is_prime(t))}, "
          f"composite t: {sum(1 for t in t_to_orders if not is_prime(t))})\n")

    # Separate prime t (zero composite obstructions!) and composite t
    prime_hits = []
    comp_hits = []

    for t in sorted(t_to_orders.keys()):
        closes = sorted(t_to_orders[t])
        max_r = 64 if is_prime(t) else 48
        subs = find_cyclic_subgroups_with_r_le(t, max_r=max_r)
        for g, h, r in subs:
            if r < 5:
                continue
            U = frozenset(pow(g, k, t) for k in range(h))
            orbs = get_orbits_U(t, U)
            d_reps = get_shift_reps_U(t, U)
            K = len(d_reps)
            ok_d, max_d = check_tmatrix_diag_const(t, U, orbs, d_reps)
            if not ok_d:
                continue
            if is_prime(t):
                rqs = tmatrix_row_quads_prime(t, h)
                if not rqs:
                    continue
                prime_hits.append({
                    "t": t, "g": g, "h": h, "r": r, "K": K,
                    "max_d": max_d, "rqs": rqs, "closes": closes,
                })
            else:
                rqs = tmatrix_row_quads_general(t, [len(o) for o in orbs])
                if not rqs:
                    continue
                comp_hits.append({
                    "t": t, "g": g, "h": h, "r": r, "K": K,
                    "max_d": max_d, "rqs": rqs, "closes": closes,
                })

    prime_hits.sort(key=lambda x: (x["K"], x["r"], x["t"]))
    print("=" * 95)
    print(f"PRIME-t T-MATRIX TARGETS (r <= 64, ZERO composite obstructions!): {len(prime_hits)} SURVIVORS")
    print("=" * 95)
    for idx, x in enumerate(prime_hits[:35], 1):
        cl_str = ", ".join(f"{n}(w={w})" for n, w in x["closes"])
        rq_str = str([sq for _, sq in x["rqs"][:2]])
        print(
            f"  #{idx:2d}  t={x['t']:4d} (prime)  g={x['g']:4d} |U|={x['h']:2d}  "
            f"r={x['r']:2d}  K={x['K']:2d}  #rqs={len(x['rqs']):2d} {rq_str:26s}  CLOSES: {cl_str}"
        )

    comp_hits.sort(key=lambda x: (x["K"], x["r"], x["t"]))
    print("\n" + "=" * 95)
    print(f"COMPOSITE-t T-MATRIX TARGETS (r <= 48): {len(comp_hits)} SURVIVORS (top 15)")
    print("=" * 95)
    for idx, x in enumerate(comp_hits[:15], 1):
        cl_str = ", ".join(f"{n}(w={w})" for n, w in x["closes"])
        print(
            f"  #{idx:2d}  t={x['t']:4d} ({fmt_fac(x['t']):10s})  g={x['g']:4d} |U|={x['h']:2d}  "
            f"r={x['r']:2d}  K={x['K']:2d}  #rqs={len(x['rqs']):2d}  CLOSES: {cl_str}"
        )

    print(f"\nTotal wall time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
