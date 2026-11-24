#!/usr/bin/env python3
"""
Exact-Integer Composite-Compression & Shift-MaxCut Algebraic Falsifier
for the 3 surviving r <= 36 Theorem 5 (b=4, n=4m+4) Hadamard targets:
  1. n = 7572, m = 1892 = 2^2 * 11 * 43, g = 15 (|U| = 210, r = 27, K = 14)
  2. n = 6668, m = 1666 = 2 * 7^2 * 17,  g = 9  (|U| = 168, r = 30, K = 17)
  3. n = 7572, m = 1892 = 2^2 * 11 * 43, g = 9  (|U| = 105, r = 36, K = 18)

100% exact integer arithmetic (zero floating point).
"""

import math
import time
from collections import defaultdict
from deep_falsify import (
    get_orbits,
    get_shift_count,
    ord_mod,
    reachable_sums,
    check_2_compression,
    check_D_compression_L2,
)


def self_test_paf_identities():
    """Verify exact d-compression PAF identities on a known Theorem 5 quadruple (m=4, n=20)."""
    m, n = 4, 20
    seqs = [
        [1, 1, 1, -1],
        [1, 1, -1, -1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
    ]
    for s in range(m):
        tot = sum(sum(f[x] * f[(x + s) % m] for x in range(m)) for f in seqs)
        expected = 4 * m if s == 0 else -4
        assert tot == expected
    for d in [1, 2, 4]:
        comps = [[sum(f[x] for x in range(m) if x % d == j) for j in range(d)] for f in seqs]
        T0 = n - 4 * (m // d)
        Td = -4 * (m // d)
        for s in range(d):
            tot = sum(sum(c[j] * c[(j + s) % d] for j in range(d)) for c in comps)
            expected = T0 if s == 0 else Td
            assert tot == expected


def check_shift_bounds(n, m, g, orbits, max_comp_size=18):
    """Exact diagonal-count (diag_const) and exact MaxCut lower bound on PAF(f, t)
    for every shift orbit t of <g, -1> on Z_m \\ {0}.

    For any shift t:
      PAF(f, t) = D(t) + sum_{i<j} C^(t)_{i,j} x_i x_j = m - 2 * Cut(G^(t), x),
      so min_f PAF(f, t) = m - 2 * MaxCut(G^(t)) >= 2 * D(t) - m.
    For Theorem 5 (b=4), sum_{k=1}^4 PAF(f_k, t) = -4, so 4 * PAF_min(t) <= -4
    (i.e. PAF_min(t) <= -1, or <= -2 for even m) is NECESSARY.
    """
    elem_to_orb = [0] * m
    for idx, o in enumerate(orbits):
        for x in o:
            elem_to_orb[x] = idx

    seen = [False] * m
    seen[0] = True
    reps = []
    for t in range(1, m):
        if not seen[t]:
            reps.append(t)
            cur = t
            while not seen[cur]:
                seen[cur] = True
                seen[(m - cur) % m] = True
                cur = (cur * g) % m

    results = []
    falsified = []
    for t in reps:
        D = 0
        adj = defaultdict(lambda: defaultdict(int))
        for x in range(m):
            i = elem_to_orb[x]
            j = elem_to_orb[(x + t) % m]
            if i == j:
                D += 1
            else:
                u, v = (i, j) if i < j else (j, i)
                adj[u][v] += 1

        # Build undirected adjacency for components
        undir = defaultdict(set)
        for u in adj:
            for v in adj[u]:
                undir[u].add(v)
                undir[v].add(u)

        vis = set()
        comps = []
        for u in sorted(undir.keys()):
            if u not in vis:
                comp = []
                stack = [u]
                vis.add(u)
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    for nb in undir[cur]:
                        if nb not in vis:
                            vis.add(nb)
                            stack.append(nb)
                comps.append(sorted(comp))

        triv_min = 2 * D - m
        exact_min = None
        if all(len(c) <= max_comp_size for c in comps):
            total_mc = 0
            for comp in comps:
                k = len(comp)
                imap = {u: idx for idx, u in enumerate(comp)}
                edges = []
                for u in comp:
                    for v, w in adj[u].items():
                        if v in imap:
                            edges.append((imap[u], imap[v], w))
                best = 0
                for mask in range(1 << (k - 1)):
                    cut = 0
                    for a, b, w in edges:
                        sa = (mask >> (a - 1)) & 1 if a > 0 else 0
                        sb = (mask >> (b - 1)) & 1 if b > 0 else 0
                        if sa != sb:
                            cut += w
                    if cut > best:
                        best = cut
                total_mc += best
            exact_min = m - 2 * total_mc

        eff_min = exact_min if exact_min is not None else triv_min
        is_bad = (4 * eff_min > -4)
        rec = {
            "t": t, "gcd": math.gcd(t, m), "D": D,
            "triv_min": triv_min, "exact_min": exact_min,
            "comp_sizes": [len(c) for c in comps],
            "falsified": is_bad,
        }
        results.append(rec)
        if is_bad:
            falsified.append(rec)
    return falsified, results


def get_d_orbit_structure(m, g, orbits, d):
    by_proj = defaultdict(list)
    for o in orbits:
        reps = tuple(sorted({x % d for x in o}))
        assert len(o) % len(reps) == 0
        by_proj[reps].append(len(o) // len(reps))

    cosets = sorted(by_proj.keys())
    weights = [by_proj[c] for c in cosets]
    r_d = len(cosets)

    res_to_cid = [0] * d
    for idx, c in enumerate(cosets):
        for r in c:
            res_to_cid[r] = idx

    seen = [False] * d
    seen[0] = True
    shift_reps = []
    shift_sizes = []
    g_d = g % d
    for s in range(1, d):
        if not seen[s]:
            orb = set()
            cur = s
            while cur not in orb:
                orb.add(cur)
                orb.add((d - cur) % d)
                cur = (cur * g_d) % d
            for x in orb:
                seen[x] = True
            shift_reps.append(s)
            shift_sizes.append(len(orb))

    paf_mats = []
    for s in shift_reps:
        mat = [[0] * r_d for _ in range(r_d)]
        for x in range(d):
            ci = res_to_cid[x]
            cj = res_to_cid[(x + s) % d]
            mat[ci][cj] += 1
        paf_mats.append(mat)

    return cosets, weights, shift_reps, shift_sizes, paf_mats


def enumerate_subset_by_sum(csizes, R_lists, target_sum, max_sq, max_tuples=500000):
    k = len(csizes)
    if k == 0:
        return (False, [()] if target_sum == 0 and max_sq >= 0 else [])

    min_sq = [min(csizes[i] * a * a for a in R_lists[i]) for i in range(k)]
    if sum(min_sq) > max_sq:
        return (False, [])

    best_split = 0
    best_cost = float("inf")
    for sp in range(k + 1):
        pL = 1
        for i in range(sp):
            pL *= len(R_lists[i])
        pR = 1
        for i in range(sp, k):
            pR *= len(R_lists[i])
        if pL + pR < best_cost:
            best_cost = pL + pR
            best_split = sp

    rem_min_R = sum(min_sq[best_split:])
    left_by_sum = defaultdict(list)

    def dfs_L(idx, csum, csq, cur):
        if idx == best_split:
            left_by_sum[csum].append((csq, tuple(cur)))
            return
        csz = csizes[idx]
        rem = sum(min_sq[idx + 1 : best_split]) + rem_min_R
        for a in R_lists[idx]:
            nsq = csq + csz * a * a
            if nsq + rem <= max_sq:
                cur.append(a)
                dfs_L(idx + 1, csum + csz * a, nsq, cur)
                cur.pop()

    dfs_L(0, 0, 0, [])
    if not left_by_sum:
        return (False, [])

    for s_val in left_by_sum:
        left_by_sum[s_val].sort(key=lambda x: x[0])

    results = []
    aborted = False

    def dfs_R(idx, csum, csq, cur):
        nonlocal aborted
        if aborted:
            return
        if idx == k:
            need = target_sum - csum
            if need in left_by_sum:
                rem_sq = max_sq - csq
                for lsq, ltup in left_by_sum[need]:
                    if lsq > rem_sq:
                        break
                    results.append(ltup + tuple(cur))
                    if len(results) > max_tuples:
                        aborted = True
                        return
            return
        csz = csizes[idx]
        rem = sum(min_sq[idx + 1 :])
        for a in R_lists[idx]:
            nsq = csq + csz * a * a
            if nsq + rem <= max_sq:
                cur.append(a)
                dfs_R(idx + 1, csum + csz * a, nsq, cur)
                cur.pop()
                if aborted:
                    return

    dfs_R(best_split, 0, 0, [])
    return aborted, results


def compute_paf_vector(a, csizes, paf_mats_used):
    p0 = sum(csizes[i] * a[i] * a[i] for i in range(len(a)))
    vec = [p0]
    r_d = len(a)
    for mat in paf_mats_used:
        val = 0
        for i in range(r_d):
            ai = a[i]
            if ai == 0:
                continue
            row = mat[i]
            for j in range(r_d):
                mij = row[j]
                if mij:
                    val += mij * ai * a[j]
        vec.append(val)
    return tuple(vec)


def verify_witness_on_Zd(n, m, d, cosets, witness):
    T0 = n - 4 * (m // d)
    Td = -4 * (m // d)
    full_seqs = []
    for a in witness:
        arr = [0] * d
        for cid, c in enumerate(cosets):
            for r in c:
                arr[r] = a[cid]
        full_seqs.append(arr)
    assert [sum(arr) for arr in full_seqs] == [2, 0, 0, 0]
    for s in range(d):
        tot = sum(sum(arr[j] * arr[(j + s) % d] for j in range(d)) for arr in full_seqs)
        expected = T0 if s == 0 else Td
        assert tot == expected, f"Verification failed at d={d}, s={s}: got {tot}, expected {expected}"


def analyze_divisor_exact(n, m, g, orbits, d, sols2, max_single_tuples=400000, max_paf_vecs=8000):
    t0 = time.time()
    row_sum_quad = (2, 0, 0, 0)
    T0 = n - 4 * (m // d)
    Td = -4 * (m // d)

    cosets, weights, shift_reps, shift_sizes, paf_mats = get_d_orbit_structure(m, g, orbits, d)
    r_d = len(cosets)
    K_d = len(shift_reps)
    csizes = [len(c) for c in cosets]
    R_lists = [
        sorted(a for a in reachable_sums(weights[i]) if csizes[i] * a * a <= T0)
        for i in range(r_d)
    ]

    # Step 1: Exact L2-norm DP check (only when r_d <= 12 to avoid huge DP state space)
    if r_d <= 12:
        ok_l2, l2_info = check_D_compression_L2(n, m, 4, orbits, d, row_sum_quad)
        if not ok_l2:
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "FALSIFIED_L2",
                "detail": f"Exact L2-norm obstruction: T0={T0}, max_single={l2_info['max_single_P0']}",
                "elapsed": time.time() - t0,
            }
    else:
        return {
            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
            "status": "SKIPPED_LARGE_RD",
            "detail": f"r_d={r_d} > 12 (skipped generic enumeration)",
            "elapsed": time.time() - t0,
        }

    # For even d, use exact 2-compression (e_f, o_f) to split even/odd cosets
    allowed_eo_2 = None
    allowed_eo_0 = None
    if d % 2 == 0 and sols2 is not None:
        allowed_eo_2 = sorted({((2 + d0) // 2, (2 - d0) // 2) for (d0, _, _, _) in sols2})
        d0_vals = set()
        for (_, d1, d2, d3) in sols2:
            d0_vals.add(d1)
            d0_vals.add(d2)
            d0_vals.add(d3)
        allowed_eo_0 = sorted({(dv // 2, -dv // 2) for dv in d0_vals})

    def enumerate_for_s(s_f, allowed_eo):
        if d % 2 == 0 and allowed_eo is not None:
            even_idx = [i for i, c in enumerate(cosets) if c[0] % 2 == 0]
            odd_idx = [i for i, c in enumerate(cosets) if c[0] % 2 == 1]
            e_csizes = [csizes[i] for i in even_idx]
            o_csizes = [csizes[i] for i in odd_idx]
            e_R = [R_lists[i] for i in even_idx]
            o_R = [R_lists[i] for i in odd_idx]

            all_seqs = []
            for ef, of in allowed_eo:
                abE, e_tups = enumerate_subset_by_sum(e_csizes, e_R, ef, T0, max_single_tuples)
                if abE:
                    return True, []
                if not e_tups:
                    continue
                abO, o_tups = enumerate_subset_by_sum(o_csizes, o_R, of, T0, max_single_tuples)
                if abO:
                    return True, []
                if not o_tups:
                    continue
                e_with_sq = sorted(
                    (sum(e_csizes[j] * t[j] * t[j] for j in range(len(e_csizes))), t)
                    for t in e_tups
                )
                o_with_sq = sorted(
                    (sum(o_csizes[j] * t[j] * t[j] for j in range(len(o_csizes))), t)
                    for t in o_tups
                )
                for esq, et in e_with_sq:
                    rem = T0 - esq
                    if rem < 0:
                        break
                    for osq, ot in o_with_sq:
                        if osq > rem:
                            break
                        full = [0] * r_d
                        for idx_pos, orig_i in enumerate(even_idx):
                            full[orig_i] = et[idx_pos]
                        for idx_pos, orig_i in enumerate(odd_idx):
                            full[orig_i] = ot[idx_pos]
                        all_seqs.append(tuple(full))
                        if len(all_seqs) > max_single_tuples:
                            return True, []
            return False, all_seqs
        else:
            return enumerate_subset_by_sum(csizes, R_lists, s_f, T0, max_single_tuples)

    ab2, seqs2 = enumerate_for_s(2, allowed_eo_2)
    ab0, seqs0 = enumerate_for_s(0, allowed_eo_0)

    if ab2 or ab0:
        return {
            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
            "status": "PASSED_L2_ONLY",
            "detail": f"Exact L2 passed; single-seq space > {max_single_tuples} (ab2={ab2}, ab0={ab0})",
            "elapsed": time.time() - t0,
        }

    # For s=0, canonicalize under a -> -a (also handles d_f == 0 when d is even)
    canon0 = {}
    for a in seqs0:
        neg = tuple(-x for x in a)
        key = a if a <= neg else neg
        canon0[key] = True
    seqs0 = list(canon0.keys())

    paf_mats_used = paf_mats[:-1] if K_d >= 1 else []
    target_vec = (T0,) + (Td,) * (K_d - 1) if K_d >= 1 else (T0,)

    paf2_map = defaultdict(list)
    for a in seqs2:
        pv = compute_paf_vector(a, csizes, paf_mats_used)
        paf2_map[pv].append(a)

    paf0_map = defaultdict(list)
    for a in seqs0:
        pv = compute_paf_vector(a, csizes, paf_mats_used)
        paf0_map[pv].append(a)

    V2 = sorted(paf2_map.keys())
    V0 = sorted(paf0_map.keys())

    if not V2 or not V0:
        return {
            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
            "status": "FALSIFIED_SINGLE",
            "detail": f"Empty achievable set: |V2|={len(V2)}, |V0|={len(V0)}",
            "elapsed": time.time() - t0,
        }

    min_p0_2 = min(v[0] for v in V2)
    min_p0_0 = min(v[0] for v in V0)
    V2 = [v for v in V2 if v[0] <= T0 - 3 * min_p0_0]
    V0 = [v for v in V0 if v[0] <= T0 - min_p0_2 - 2 * min_p0_0]

    if not V2 or not V0:
        return {
            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
            "status": "FALSIFIED_P0_BOUND",
            "detail": f"No sequences survive tightened P0 bound (min_p0_2={min_p0_2}, min_p0_0={min_p0_0}, T0={T0})",
            "elapsed": time.time() - t0,
        }

    if len(V0) > max_paf_vecs:
        return {
            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
            "status": "PASSED_L2_SKIPPED_MITM",
            "detail": f"|seqs2|={len(seqs2)}, |seqs0|={len(seqs0)}, |V2|={len(V2)}, |V0|={len(V0)} (|V0| > {max_paf_vecs})",
            "elapsed": time.time() - t0,
        }

    dim = len(target_vec)
    max_p0_pair00 = T0 - min_p0_2 - min_p0_0
    pair00 = defaultdict(list)
    n0 = len(V0)
    for i in range(n0):
        vi = V0[i]
        vi0 = vi[0]
        if vi0 + vi0 > max_p0_pair00:
            break
        for j in range(i, n0):
            vj = V0[j]
            if vi0 + vj[0] > max_p0_pair00:
                break
            svec = tuple(vi[k] + vj[k] for k in range(dim))
            pair00[svec].append((i, j))

    paf_quads = set()
    seq_quad_count = 0
    sample_witness = None

    for v1 in V2:
        v1_0 = v1[0]
        c1 = len(paf2_map[v1])
        for k in range(n0):
            vk = V0[k]
            if v1_0 + vk[0] > T0 - 2 * min_p0_0:
                break
            need = tuple(target_vec[idx] - v1[idx] - vk[idx] for idx in range(dim))
            if need in pair00:
                for i, j in pair00[need]:
                    if j <= k:
                        qkey = (v1, i, j, k)
                        if qkey not in paf_quads:
                            paf_quads.add(qkey)
                            ci = len(paf0_map[V0[i]])
                            cj = len(paf0_map[V0[j]])
                            ck = len(paf0_map[V0[k]])
                            if i == j == k:
                                mult0 = ci * (ci + 1) * (ci + 2) // 6
                            elif i == j:
                                mult0 = (ci * (ci + 1) // 2) * ck
                            elif j == k:
                                mult0 = ci * (cj * (cj + 1) // 2)
                            else:
                                mult0 = ci * cj * ck
                            seq_quad_count += c1 * mult0
                            if sample_witness is None:
                                sample_witness = (
                                    paf2_map[v1][0],
                                    paf0_map[V0[i]][0],
                                    paf0_map[V0[j]][0],
                                    paf0_map[V0[k]][0],
                                )

    elapsed = time.time() - t0
    if not paf_quads:
        return {
            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
            "status": "FALSIFIED_EXACT_PAF",
            "detail": (
                f"EXACT PAF OBSTRUCTION mod d={d} (r_d={r_d}, K_d={K_d}, T0={T0}, Td={Td}): "
                f"|seqs2|={len(seqs2)}, |seqs0|={len(seqs0)}, |V2|={len(V2)}, |V0|={len(V0)}, 0 feasible quadruples!"
            ),
            "elapsed": elapsed,
        }
    else:
        verify_witness_on_Zd(n, m, d, cosets, sample_witness)
        return {
            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
            "status": "SURVIVED_EXACT_PAF",
            "paf_quads": len(paf_quads),
            "seq_quads": seq_quad_count,
            "V2": len(V2), "V0": len(V0),
            "seqs2": len(seqs2), "seqs0": len(seqs0),
            "sample_witness": sample_witness,
            "cosets": cosets,
            "detail": (
                f"|seqs2|={len(seqs2)}, |seqs0|={len(seqs0)}, |V2|={len(V2)}, |V0|={len(V0)} -> "
                f"{len(paf_quads)} PAF-quads, {seq_quad_count} seq-quads"
            ),
            "elapsed": elapsed,
        }


def main():
    self_test_paf_identities()
    print("Self-test of exact d-compression PAF identities PASSED.\n", flush=True)

    targets = [
        (7572, 1892, 15, "Target 1: n=7572, m=1892=2^2*11*43, g=15"),
        (6668, 1666, 9,  "Target 2: n=6668, m=1666=2*7^2*17, g=9"),
        (7572, 1892, 9,  "Target 3: n=7572, m=1892=2^2*11*43, g=9"),
    ]

    for n, m, g, label in targets:
        orbits = get_orbits(m, g)
        r = len(orbits)
        K = get_shift_count(m, g)
        U = ord_mod(g, m)
        sols2 = check_2_compression(n, m, orbits, (2, 0, 0, 0))
        print("=" * 95, flush=True)
        print(f"{label}  (|U|={U}, r={r}, K={K}, 2-comp sols={len(sols2)})", flush=True)
        print("=" * 95, flush=True)

        # Phase A: Exact diag_const + MaxCut shift bounds
        bad_shifts, all_shifts = check_shift_bounds(n, m, g, orbits)
        max_D_rec = max(all_shifts, key=lambda x: x["D"])
        print(f"  [Shift Bounds] max D(t) = {max_D_rec['D']} at t={max_D_rec['t']} (m/2={m//2})", flush=True)
        for rec in bad_shifts:
            print(
                f"    *** FALSIFIED BY SHIFT t={rec['t']} (gcd={rec['gcd']}): "
                f"D={rec['D']}, triv_min={rec['triv_min']}, exact_MaxCut_min={rec['exact_min']}, "
                f"4*min={4*(rec['exact_min'] if rec['exact_min'] is not None else rec['triv_min'])} > -4 ***",
                flush=True,
            )

        if bad_shifts:
            print(f"--> VERDICT for {label}: *** FALSIFIED BY EXACT SHIFT LOWER BOUND ***\n", flush=True)
            continue

        # Phase B: Exact Composite Compressions for ALL divisors d | m
        divs = [d for d in range(2, m) if m % d == 0]
        falsified = False
        for d in divs:
            res = analyze_divisor_exact(n, m, g, orbits, d, sols2)
            print(
                f"  d={d:4d} (r_d={res['r_d']:2d}, K_d={res['K_d']:2d}, T0={res['T0']:4d}, Td={res['Td']:5d}) "
                f"[{res['status']:22s}] ({res['elapsed']:.2f}s) {res['detail']}",
                flush=True,
            )
            if res["status"].startswith("FALSIFIED"):
                falsified = True
        print(
            f"--> VERDICT for {label}: {'*** FALSIFIED ***' if falsified else 'SURVIVED ALL CHECKED DIVISORS'}\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
