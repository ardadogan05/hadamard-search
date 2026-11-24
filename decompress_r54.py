#!/usr/bin/env python3
"""
Exact Compression-Decompression Tree for the top two r=54 Theorem 5 survivors:
  1. n = 9388, m = 2346 = 6 * 391, g = 25 (|U| = 88, r = 54, K = 29):
     d = 391 (320 seq-quads) -> d = 782 = 2 * 391 -> full m = 2346
  2. n = 7180, m = 1794 = 6 * 299, g = 49 (|U| = 66, r = 54, K = 29):
     d = 299 (2188 seq-quads) -> d = 598 = 2 * 299 -> full m = 1794

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
    (2) sum_{k=1}^4 PAF(b_k, 0) = n,  and  sum_{k=1}^4 PAF(b_k, s) = 0 for all s != 0 mod d!
  Moreover, PSD(b_k, u) = PSD(f_k^(2d), odd freq) >= 0 and sum_k PSD(b_k, u) = n for all u in Z_d!
"""

import math
import time
from collections import defaultdict
from check_composites_v2 import (
    generate_subgroup,
    get_orbits_U,
    get_d_orbit_structure_U,
    enumerate_subset_by_sum,
    compute_paf_vector,
    build_rigorous_psd_filter,
    verify_witness_on_Zd,
)
from deep_falsify import reachable_sums, check_2_compression


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

    ab2, seqs2 = enumerate_subset_by_sum(csizes, R_lists, 2, T0)
    ab0, seqs0 = enumerate_subset_by_sum(csizes, R_lists, 0, T0)
    assert not ab2 and not ab0

    canon0 = {}
    for a in seqs0:
        neg = tuple(-x for x in a)
        canon0[a if a <= neg else neg] = True
    seqs0 = list(canon0.keys())

    paf_used = paf_mats[:-1]
    tvec = (T0,) + (Td,) * (len(shift_reps) - 1)
    dim = len(tvec)

    psd_ok = build_rigorous_psd_filter(n, d, U, shift_reps)
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
    seen_paf = set()
    for v1 in V2:
        for k, vk in enumerate(V0):
            if v1[0] + vk[0] > T0 - 2 * min0:
                break
            need = tuple(tvec[idx] - v1[idx] - vk[idx] for idx in range(dim))
            if need in pair00:
                for i, j in pair00[need]:
                    if j <= k and (v1, i, j, k) not in seen_paf:
                        seen_paf.add((v1, i, j, k))
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
    return U, orbits, cosets, weights, shift_reps, shift_sizes, paf_mats, quads


def lift_d_to_2d(n, m, g, d, label):
    t0 = time.time()
    print("=" * 95, flush=True)
    print(f"DECOMPRESSING {label}: d={d} -> 2d={2*d}", flush=True)
    print("=" * 95, flush=True)
    U, orbits, cosets, weights, shift_reps, shift_sizes, paf_mats, quads = get_all_d_quads(n, m, g, d)
    r_d = len(cosets)
    K_d = len(shift_reps)
    csizes = [len(c) for c in cosets]
    print(f"  Found {len(quads)} exact d={d} seq-quads ({time.time()-t0:.2f}s)", flush=True)

    # Check weights of even and odd halves for each coset of Z_d
    # Since m / d = 6 and gcd(2, d) = 1, each coset of Z_d splits into 3 even orbits and 3 odd orbits!
    even_R = []
    odd_R = []
    by_2d = defaultdict(list)
    for o in orbits:
        reps = tuple(sorted({x % (2 * d) for x in o}))
        by_2d[reps].append(len(o) // len(reps))
    # For each coset c of Z_d, find its even and odd lifts in Z_{2d}
    for c in cosets:
        c_ev = tuple(sorted(x if x % 2 == 0 else x + d for x in c))
        c_od = tuple(sorted(x if x % 2 == 1 else x + d for x in c))
        even_R.append(set(reachable_sums(by_2d[c_ev])))
        odd_R.append(set(reachable_sums(by_2d[c_od])))
    print(f"  Even/odd coset value sets sample: even_R[0]={sorted(even_R[0])}, odd_R[0]={sorted(odd_R[0])}", flush=True)

    # Exact 2-compression allowed d_f = e_f - o_f values
    sols2 = check_2_compression(n, m, orbits, (2, 0, 0, 0))
    allowed_d2 = sorted({d0 for (d0, _, _, _) in sols2})
    allowed_d0 = sorted({dv for (_, d1, d2, d3) in sols2 for dv in (d1, d2, d3)})
    print(f"  2-comp allowed d_f: |d2|={len(allowed_d2)}, |d0|={len(allowed_d0)} (from {len(sols2)} 2-comp sols)", flush=True)

    # Rigorous PSD <= n filter for b on Z_d (since sum_k PSD(b_k, u) = n for ALL u in Z_d!)
    psd_ok = build_rigorous_psd_filter(n, d, U, shift_reps)

    # Collect unique single sequences appearing in quads
    uniq_a2 = sorted({q[0] for q in quads})
    uniq_a0 = sorted({q[k] for q in quads for k in (1, 2, 3)})
    print(f"  Unique single d={d} seqs in quads: |uniq_a2|={len(uniq_a2)}, |uniq_a0|={len(uniq_a0)}", flush=True)

    # For each unique a, find ALL valid lifts b = e - o with:
    #   b_i in allowed splits of a_i,
    #   sum_i csizes[i] * b_i in allowed_d_list,
    #   PAF(b, 0) <= n,
    #   PSD(b, u) <= n for all u in shift_reps!
    def compute_b_paf_map(a, allowed_d_list, is_s0):
        # For each coordinate i, valid b_i values
        B_lists = []
        for i in range(r_d):
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
            ab, b_tups = enumerate_subset_by_sum(csizes, B_lists, df, n, max_tuples=200000)
            assert not ab, f"Overflow on a={a}, df={df}"
            for b in b_tups:
                fp = compute_paf_vector(b, csizes, paf_mats)
                if psd_ok(fp):
                    # Store full (K_d + 1)-dim vector? Wait: sum_{s in Z_d} PAF(b, s) = df^2,
                    # so (P0, PAF(s_1), ..., PAF(s_{K_d})) has K_d + 1 coords (including df^2 implicitly).
                    # Actually keeping all K_d + 1 coords is tiny (K_d = 5 -> 6 coords) and exact!
                    b_by_paf[fp].append(b)
        return b_by_paf

    t1 = time.time()
    cache_a2 = {}
    for a in uniq_a2:
        cache_a2[a] = compute_b_paf_map(a, allowed_d2, False)
    cache_a0 = {}
    for a in uniq_a0:
        # Note: for s=0, a was canonicalized up to sign, and flipping a -> -a flips b -> -b,
        # which leaves PAF(b) unchanged! And df >= 0 is sufficient because flipping f_k -> -f_k
        # flips BOTH a -> -a and b -> -b (Wait: if we fix a, flipping f_k -> -f_k flips a -> -a!
        # Since a is FIXED, we MUST allow BOTH +df and -df for b!!
        # Wait: CHECK THIS! If a is fixed and nonzero, e -> -e, o -> -o flips a -> -a AND b -> -b.
        # Swapping e <-> o keeps a = e+o FIXED and flips b = e-o -> -b!
        # Does swapping e <-> o keep PAF(b) unchanged? YES, PAF(-b) = PAF(b)!
        # So for a fixed a, if b is achievable, then -b is ALSO achievable (just swap even and odd cosets,
        # which is valid because even_R[i] == odd_R[i] for all i!).
        # Let's assert even_R[i] == odd_R[i], then df >= 0 suffices for s=0!)
        assert all(even_R[i] == odd_R[i] for i in range(r_d))
        cache_a0[a] = compute_b_paf_map(a, allowed_d0, True)

    tot_b2 = sum(len(v) for v in cache_a2.values())
    tot_b0 = sum(len(v) for v in cache_a0.values())
    print(f"  Built b-lift caches in {time.time()-t1:.2f}s: avg |V(b)| per a2={tot_b2/max(1,len(uniq_a2)):.1f}, per a0={tot_b0/max(1,len(uniq_a0)):.1f}", flush=True)

    # Now check EVERY d-quad (a1, a2, a3, a4) for exact b-lift quadruple summing to (n, 0, 0, ..., 0)!
    target_b = (n,) + (0,) * K_d
    dim = K_d + 1
    surviving_d_quads = 0
    total_2d_paf_quads = 0
    total_2d_seq_quads = 0
    sample_2d_witness = None

    t2 = time.time()
    for idx_q, (a1, a2, a3, a4) in enumerate(quads):
        M1 = cache_a2[a1]
        M2 = cache_a0[a2]
        M3 = cache_a0[a3]
        M4 = cache_a0[a4]
        if not (M1 and M2 and M3 and M4):
            continue
        V1 = sorted(M1.keys())
        V2 = sorted(M2.keys())
        V3 = sorted(M3.keys())
        V4 = sorted(M4.keys())

        min1, min2, min3, min4 = V1[0][0], V2[0][0], V3[0][0], V4[0][0]
        if min1 + min2 + min3 + min4 > n:
            continue

        # MITM: pair (V1, V2) and (V3, V4)
        max12 = n - min3 - min4
        pair12 = defaultdict(list)
        for v1 in V1:
            if v1[0] + min2 > max12:
                break
            for v2 in V2:
                if v1[0] + v2[0] > max12:
                    break
                pair12[tuple(v1[c] + v2[c] for c in range(dim))].append((v1, v2))
        if not pair12:
            continue

        max34 = n - min1 - min2
        q_paf_hits = 0
        q_seq_hits = 0
        for v3 in V3:
            if v3[0] + min4 > max34:
                break
            for v4 in V4:
                if v3[0] + v4[0] > max34:
                    break
                need = tuple(target_b[c] - v3[c] - v4[c] for c in range(dim))
                if need in pair12:
                    for v1, v2 in pair12[need]:
                        q_paf_hits += 1
                        q_seq_hits += len(M1[v1]) * len(M2[v2]) * len(M3[v3]) * len(M4[v4])
                        if sample_2d_witness is None:
                            sample_2d_witness = (
                                (a1, M1[v1][0]),
                                (a2, M2[v2][0]),
                                (a3, M3[v3][0]),
                                (a4, M4[v4][0]),
                            )
        if q_paf_hits > 0:
            surviving_d_quads += 1
            total_2d_paf_quads += q_paf_hits
            total_2d_seq_quads += q_seq_hits

    print(
        f"  --> RESULT mod 2d={2*d} ({time.time()-t2:.2f}s): "
        f"{surviving_d_quads} / {len(quads)} d-quads lift to 2d={2*d}! "
        f"({total_2d_paf_quads} 2d-PAF-quads, {total_2d_seq_quads} 2d-seq-quads)",
        flush=True,
    )

    if sample_2d_witness is not None:
        # Independently verify the lifted 2d-compressed quadruple on Z_{2d}!
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
        assert [sum(arr) for arr in full_2d] == [2, 0, 0, 0]
        for s in range(2 * d):
            tot = sum(sum(arr[j] * arr[(j + s) % (2 * d)] for j in range(2 * d)) for arr in full_2d)
            expected = T0_2d if s == 0 else Td_2d
            assert tot == expected, f"2d verify failed at s={s}: {tot} != {expected}"
        print(f"  [VERIFIED] Sample 2d={2*d} witness passed brute-force PAF check on ALL {2*d} shifts!", flush=True)


if __name__ == "__main__":
    self_test_2d_identity()
    print("Self-test of exact d -> 2d (a=e+o, b=e-o) identity PASSED.\n", flush=True)
    lift_d_to_2d(9388, 2346, 25, 391, "n=9388, m=2346, g=25 (r=54, K=29)")
    lift_d_to_2d(7180, 1794, 49, 299, "n=7180, m=1794, g=49 (r=54, K=29)")
