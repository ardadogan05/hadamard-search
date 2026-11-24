#!/usr/bin/env python3
"""
Exact necessary d and 2d compression checks for cyclic Theorem 5 families.

The full sequences have length m and their summed PAF is n-4 at shift
zero and -4 at every other shift. At modulus d, the summed PAF is
n-4*m/d at shift zero and -4*m/d at every other shift. A failed lift rules out the selected
multiplier family. A successful lift gives a compressed witness, which is
a full sequence witness only if 2d = m.

Mathematical identity for odd d with 2d | m:
  Writing f^(2d) as (e, o) on Z_d where e(j) = f^(2d)(even rep of j mod d),
  o(j) = f^(2d)(odd rep of j mod d), we have:
    a = e + o = f^(d)
    b = e - o  (the alternating / signed d-compression)
  Then for every shift s in Z_d:
    PAF(f^(2d), even shift = s mod d) + PAF(f^(2d), odd shift = s mod d) = PAF(a, s)
    PAF(f^(2d), even shift = s mod d) - PAF(f^(2d), odd shift = s mod d) = PAF(b, s)
  Hence a quadruple of lifts (e_k, o_k) satisfies the EXACT 2d-compression PAF equations
  IF AND ONLY IF:
    (1) (a_1, a_2, a_3, a_4) satisfies the exact d-compression PAF equations, AND
    (2) sum_k PAF(b_k, 0) = n and sum_k PAF(b_k, s) = 0 for all s != 0 mod d.
  The PSD of b_k is the PSD of f_k^(2d) at the corresponding odd frequency.
  Thus 0 <= PSD(b_k, u) <= n for every u in Z_d.
"""

import argparse
import json
import time
from collections import defaultdict
from fractions import Fraction
from math import gcd

from check_composites_v2 import (
    compute_paf_vector,
    enumerate_subset_by_sum,
    generate_subgroup,
    get_d_orbit_structure_U,
    get_orbits_U,
)
from deep_falsify import reachable_sums, check_2_compression


TARGETS = {
    9388: (2346, 25, 391),
    7180: (1794, 49, 299),
    6236: (1558, 119, 19),
}
ROW_SUMS = (2, 0, 0, 0)


def build_exact_psd_filter(n, d, U, shift_reps):
    """Use rational coefficient intervals to prune PSD values greater than n."""
    scale = 10**20

    def atan_bounds(denominator):
        # An even number of terms bounds atan from below. The next term
        # bounds the remainder by the alternating series theorem.
        terms = 20
        lower = sum(
            Fraction((-1)**k, (2*k + 1) * denominator**(2*k + 1))
            for k in range(terms)
        )
        return lower, lower + Fraction(1, (2*terms + 1) * denominator**(2*terms + 1))

    a_low, a_high = atan_bounds(5)
    b_low, b_high = atan_bounds(239)
    pi_low, pi_high = 16*a_low - 4*b_high, 16*a_high - 4*b_low
    # Machin's formula certifies these shorter rational bounds on pi.
    pi = Fraction(31415926535897932384626, 10**22)
    pi_error = Fraction(1, 10**22)
    assert pi < pi_low <= pi_high < pi + pi_error

    cos_bounds = {}
    for j in range(d // 2 + 1):
        x = pi * Fraction(2*j, d)
        term = value = Fraction(1)
        for k in range(1, 24):
            term *= -x*x / ((2*k - 1) * (2*k))
            value += term
        # Taylor's theorem bounds the error by x**48 / 48!. Replacing
        # the true angle by x changes cosine by at most pi_error.
        error = abs(term * x*x) / (47 * 48) + pi_error
        low, high = (value - error) * scale, (value + error) * scale
        cos_bounds[j] = (low.numerator // low.denominator, -(-high.numerator // high.denominator))

    orbit_sets = [{sign*s*u % d for sign in (-1, 1) for u in U} for s in shift_reps]
    rows = []
    for u in shift_reps:
        row = []
        for shifts in orbit_sets:
            indices = (u*s % d for s in shifts)
            bounds = [cos_bounds[min(j, d-j)] for j in indices]
            row.append((sum(low for low, _ in bounds), sum(high for _, high in bounds)))
        rows.append(row)

    def psd_ok(paf):
        for row in rows:
            lower = paf[0] * scale
            for value, (low, high) in zip(paf[1:], row):
                lower += value * (low if value >= 0 else high)
            if lower > n * scale:
                return False
        return True

    return psd_ok


def enumerate_exact(csizes, value_lists, row_sum, energy_bound):
    if any(not values for values in value_lists):
        return []
    aborted, candidates = enumerate_subset_by_sum(csizes, value_lists, row_sum, energy_bound)
    if aborted:
        raise RuntimeError(f"Enumeration exceeded the candidate limit for row sum {row_sum}")
    return candidates


def self_test_2d_identity():
    """Verify the exact (a = e+o, b = e-o) identity for d -> 2d on m=6 (d=3, 2d=6)."""
    # Take arbitrary integer arrays e, o of length d=3, build f of length 6
    d = 3
    e = [3, -1, 1]
    o = [-1, 3, -3]
    f = [0] * (2 * d)
    for x in range(2 * d):
        f[x] = e[x % d] if x % 2 == 0 else o[x % d]
    a = [e[i] + o[i] for i in range(d)]
    b = [e[i] - o[i] for i in range(d)]
    for s in range(d):
        # Find even and odd shifts mod 2d congruent to s mod d
        ev = s if s % 2 == 0 else s + d
        od = s if s % 2 == 1 else s + d
        paf_ev = sum(f[x] * f[(x + ev) % (2 * d)] for x in range(2 * d))
        paf_od = sum(f[x] * f[(x + od) % (2 * d)] for x in range(2 * d))
        paf_a = sum(a[i] * a[(i + s) % d] for i in range(d))
        paf_b = sum(b[i] * b[(i + s) % d] for i in range(d))
        assert paf_ev + paf_od == paf_a
        assert paf_ev - paf_od == paf_b


def get_all_d_quads(n, m, g, d):
    """Enumerate ALL exact d-compression sequence quadruples (a1, a2, a3, a4)
    where a1 has row sum 2 and a2, a3, a4 have row sum 0 (canonicalized up to sign)."""
    U = generate_subgroup(m, [g])
    orbits = get_orbits_U(m, U)
    T0 = n - 4 * (m // d)
    Td = -4 * (m // d)
    cosets, weights, shift_reps, shift_sizes, paf_mats = get_d_orbit_structure_U(m, U, orbits, d)
    csizes = [len(c) for c in cosets]
    R_lists = [sorted(a for a in reachable_sums(w) if csizes[i] * a * a <= T0) for i, w in enumerate(weights)]

    seqs2 = enumerate_exact(csizes, R_lists, 2, T0)
    seqs0 = [
        a for a in enumerate_exact(csizes, R_lists, 0, T0)
        if a <= tuple(-x for x in a)
    ]

    # The sum of the PAF over all shifts is the squared row sum, so one
    # nonzero-shift orbit entry follows from the others.
    paf_used = paf_mats[:-1]
    tvec = (T0,) + (Td,) * (len(shift_reps) - 1)
    dim = len(tvec)

    psd_ok = build_exact_psd_filter(n, d, U, shift_reps)
    last_sz = shift_sizes[-1]

    def keep_paf(v, sf):
        rem = sf * sf - v[0] - sum(shift_sizes[t] * v[t + 1] for t in range(len(shift_reps) - 1))
        assert rem % last_sz == 0
        return psd_ok(v + (rem // last_sz,))

    m2 = defaultdict(list)
    for a in seqs2:
        pv = compute_paf_vector(a, csizes, paf_used)
        if keep_paf(pv, 2):
            m2[pv].append(a)
    m0 = defaultdict(list)
    for a in seqs0:
        pv = compute_paf_vector(a, csizes, paf_used)
        if keep_paf(pv, 0):
            m0[pv].append(a)

    V2 = sorted(m2.keys())
    V0 = sorted(m0.keys())
    if not V2 or not V0:
        return U, orbits, cosets, shift_reps, paf_mats, []
    min2, min0 = V2[0][0], V0[0][0]
    max_p0_pair = T0 - min2 - min0

    pair00 = defaultdict(list)
    for i, vi in enumerate(V0):
        if vi[0] + vi[0] > max_p0_pair:
            break
        for j in range(i, len(V0)):
            vj = V0[j]
            if vi[0] + vj[0] > max_p0_pair:
                break
            pair00[tuple(vi[k] + vj[k] for k in range(dim))].append((i, j))

    # Collect all distinct unordered seq-quads (a1, {a2, a3, a4})
    quads = []
    for v1 in V2:
        for k, vk in enumerate(V0):
            if v1[0] + vk[0] > T0 - 2 * min0:
                break
            need = tuple(tvec[idx] - v1[idx] - vk[idx] for idx in range(dim))
            if need in pair00:
                for i, j in pair00[need]:
                    if j <= k:
                        L1 = m2[v1]
                        Li, Lj, Lk = m0[V0[i]], m0[V0[j]], m0[V0[k]]
                        for a1 in L1:
                            for p_i, ai in enumerate(Li):
                                start_j = p_i if i == j else 0
                                for p_j in range(start_j, len(Lj)):
                                    aj = Lj[p_j]
                                    start_k = p_j if j == k else 0
                                    for p_k in range(start_k, len(Lk)):
                                        ak = Lk[p_k]
                                        quads.append((a1, ai, aj, ak))
    return U, orbits, cosets, shift_reps, paf_mats, quads


def lift_d_to_2d(n, m, g, d):
    if n != 4*m + 4 or m % 4 != 2 or d <= 1 or d % 2 != 1 or m % (2*d) or gcd(m, g) != 1:
        raise ValueError("Expected n=4*m+4, odd m/2, odd divisor d>1 of m/2, and a unit g")

    t0 = time.time()
    print("=" * 95, flush=True)
    print(f"n={n}, m={m}, g={g}: d={d} -> 2d={2*d}", flush=True)
    print("=" * 95, flush=True)
    U, orbits, cosets, shift_reps, paf_mats, quads = get_all_d_quads(n, m, g, d)
    K_d = len(shift_reps)
    csizes = [len(c) for c in cosets]
    print(f"  Found {len(quads)} exact d={d} seq-quads ({time.time()-t0:.2f}s)", flush=True)

    # Collect weights of even and odd halves for each orbit on Z_d.
    # Odd d gives one even and one odd representative of each residue mod d.
    even_R = []
    odd_R = []
    by_2d = defaultdict(list)
    for o in orbits:
        reps = tuple(sorted({x % (2 * d) for x in o}))
        assert len(o) % len(reps) == 0
        by_2d[reps].append(len(o) // len(reps))
    # For each coset c of Z_d, find its even and odd lifts in Z_{2d}
    for c in cosets:
        c_ev = tuple(sorted(x if x % 2 == 0 else x + d for x in c))
        c_od = tuple(sorted(x if x % 2 == 1 else x + d for x in c))
        even_R.append(set(reachable_sums(by_2d[c_ev])))
        odd_R.append(set(reachable_sums(by_2d[c_od])))
    # Translation by m/2 commutes with multiplication by odd units and,
    # for odd m/2 with d | m/2, swaps these even and odd fibers.
    if even_R != odd_R:
        raise ValueError("Even and odd fibers must have identical reachable sums")

    # Exact 2-compression restricts the attainable row sums of b.
    sols2 = check_2_compression(n, m, orbits, ROW_SUMS)
    allowed_d2 = sorted({abs(d0) for (d0, _, _, _) in sols2})
    allowed_d0 = sorted({dv for (_, d1, d2, d3) in sols2 for dv in (d1, d2, d3)})
    print(f"  2-comp allowed d_f: |d2|={len(allowed_d2)}, |d0|={len(allowed_d0)} (from {len(sols2)} 2-comp sols)", flush=True)

    # The PSDs of the four b sequences sum to n at each frequency.
    psd_ok = build_exact_psd_filter(n, d, U, shift_reps)

    # Collect unique single sequences appearing in quads
    uniq_a2 = sorted({q[0] for q in quads})
    uniq_a0 = sorted({q[k] for q in quads for k in (1, 2, 3)})
    print(f"  Unique single d={d} seqs in quads: |uniq_a2|={len(uniq_a2)}, |uniq_a0|={len(uniq_a0)}", flush=True)

    # Enumerate every compatible signed lift b = e - o with:
    #   b_i in allowed splits of a_i,
    #   sum_i csizes[i] * b_i in allowed_d_list,
    #   PAF(b, 0) <= n,
    #   the necessary PSD bound at each representative frequency.
    def compute_b_paf_map(a, allowed_d_list):
        # For each coordinate i, valid b_i values
        B_lists = []
        for i in range(len(cosets)):
            ai = a[i]
            bvals = sorted(
                e_val - o_val
                for e_val in even_R[i]
                for o_val in odd_R[i]
                if e_val + o_val == ai and csizes[i] * (e_val - o_val) ** 2 <= n
            )
            if not bvals:
                return {}
            B_lists.append(bvals)

        b_by_paf = defaultdict(list)
        for df in allowed_d_list:
            for b in enumerate_exact(csizes, B_lists, df, n):
                if df == 0 and b > tuple(-x for x in b):
                    continue
                fp = compute_paf_vector(b, csizes, paf_mats)
                if psd_ok(fp):
                    b_by_paf[fp].append(b)
        return b_by_paf

    t1 = time.time()
    # Swapping the even and odd fibers keeps a fixed and flips b.
    # Use one representative of each b/-b pair for both row-sum types.
    cache_a2 = {a: compute_b_paf_map(a, allowed_d2) for a in uniq_a2}
    cache_a0 = {a: compute_b_paf_map(a, allowed_d0) for a in uniq_a0}

    tot_b2 = sum(len(v) for v in cache_a2.values())
    tot_b0 = sum(len(v) for v in cache_a0.values())
    print(f"  Built b-lift caches in {time.time()-t1:.2f}s: avg |V(b)| per a2={tot_b2/max(1,len(uniq_a2)):.1f}, per a0={tot_b0/max(1,len(uniq_a0)):.1f}", flush=True)

    # Encode PAF vectors additively for meet-in-the-middle. Each entry of an
    # individual PAF has magnitude at most PAF(0) <= n. Each entry of a
    # four-way sum minus the target therefore has magnitude at most 5*n.
    # With this radix, a zero encoded difference forces every entry to be
    # zero by reducing modulo the radix and repeating.
    radix = 5*n + 1
    multipliers = [radix**i for i in range(K_d + 1)]

    def encode_cache(cache):
        return {
            a: [
                (paf[0], sum(x*y for x, y in zip(paf, multipliers)), len(lifts), lifts[0])
                for paf, lifts in sorted(paf_map.items())
            ]
            for a, paf_map in cache.items()
        }

    cache_a2 = encode_cache(cache_a2)
    cache_a0 = encode_cache(cache_a0)
    surviving_d_quads = 0
    total_2d_paf_quads = 0
    total_2d_seq_quads = 0
    sample_2d_witness = None

    t2 = time.time()
    for a1, a2, a3, a4 in quads:
        V1, V2, V3, V4 = cache_a2[a1], cache_a0[a2], cache_a0[a3], cache_a0[a4]
        if not (V1 and V2 and V3 and V4):
            continue

        min1, min2, min3, min4 = V1[0][0], V2[0][0], V3[0][0], V4[0][0]
        if min1 + min2 + min3 + min4 > n:
            continue

        # Store pairs from the smaller Cartesian product.
        left, right = (V1, V2), (V3, V4)
        swapped = len(V1)*len(V2) > len(V3)*len(V4)
        if swapped:
            left, right = right, left
        max_left = n - right[0][0][0] - right[1][0][0]
        max_right = n - left[0][0][0] - left[1][0][0]

        pair_left = defaultdict(list)
        for energy1, key1, count1, b1 in left[0]:
            left_budget = max_left - energy1
            if left_budget < left[1][0][0]:
                break
            for energy2, key2, count2, b2 in left[1]:
                if energy2 > left_budget:
                    break
                pair_left[key1 + key2].append((count1 * count2, b1, b2))
        if not pair_left:
            continue

        q_paf_hits = 0
        q_seq_hits = 0
        for energy3, key3, count3, b3 in right[0]:
            right_budget = max_right - energy3
            if right_budget < right[1][0][0]:
                break
            need = n - key3
            for energy4, key4, count4, b4 in right[1]:
                if energy4 > right_budget:
                    break
                matches = pair_left.get(need - key4)
                if matches:
                    q_paf_hits += len(matches)
                    q_seq_hits += count3 * count4 * sum(count for count, _, _ in matches)
                    if sample_2d_witness is None:
                        _, b1, b2 = matches[0]
                        bs = (b3, b4, b1, b2) if swapped else (b1, b2, b3, b4)
                        sample_2d_witness = tuple(zip((a1, a2, a3, a4), bs))
        if q_paf_hits > 0:
            surviving_d_quads += 1
            total_2d_paf_quads += q_paf_hits
            total_2d_seq_quads += q_seq_hits

    print(
        f"  --> RESULT mod 2d={2*d} ({time.time()-t2:.2f}s): "
        f"{surviving_d_quads} / {len(quads)} d-quads lift to 2d={2*d}! "
        f"({total_2d_paf_quads} PAF tuples, {total_2d_seq_quads} normalized lifts)",
        flush=True,
    )

    if sample_2d_witness is not None:
        # Verify the sample directly on Z_{2d}, without the signed transform.
        T0_2d = n - 4 * (m // (2 * d))
        Td_2d = -4 * (m // (2 * d))
        full_2d = []
        for a, b in sample_2d_witness:
            arr = [0] * (2 * d)
            for cid, c in enumerate(cosets):
                ev_val = (a[cid] + b[cid]) // 2
                od_val = (a[cid] - b[cid]) // 2
                assert ev_val + od_val == a[cid] and ev_val - od_val == b[cid]
                for r in c:
                    ev_pos = r if r % 2 == 0 else r + d
                    od_pos = r if r % 2 == 1 else r + d
                    arr[ev_pos] = ev_val
                    arr[od_pos] = od_val
            full_2d.append(arr)
        assert tuple(sum(arr) for arr in full_2d) == ROW_SUMS
        for s in range(2 * d):
            tot = sum(sum(arr[j] * arr[(j + s) % (2 * d)] for j in range(2 * d)) for arr in full_2d)
            expected = T0_2d if s == 0 else Td_2d
            assert tot == expected, f"2d verify failed at s={s}: {tot} != {expected}"
        print(f"  [VERIFIED] Sample 2d={2*d} witness passed brute-force PAF check on ALL {2*d} shifts!", flush=True)

        return {"n": n, "m": m, "g": g, "modulus": 2*d, "sequences": full_2d}
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0].strip())
    parser.add_argument("orders", nargs="*", type=int, help="orders to test (default: 9388 and 7180)")
    parser.add_argument("--witness-out", help="write the verified compressed witness for a single order")
    args = parser.parse_args()
    orders = args.orders or [9388, 7180]
    if any(n not in TARGETS for n in orders):
        parser.error(f"orders must be one of {', '.join(map(str, TARGETS))}")
    if args.witness_out and len(orders) != 1:
        parser.error("--witness-out requires one order")

    self_test_2d_identity()
    for n in orders:
        witness = lift_d_to_2d(n, *TARGETS[n])
        if args.witness_out and witness:
            with open(args.witness_out, "w") as output:
                json.dump(witness, output)
                output.write("\n")


if __name__ == "__main__":
    main()
