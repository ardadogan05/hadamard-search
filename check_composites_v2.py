#!/usr/bin/env python3
"""
Exact-Integer Shift-MaxCut & Composite-Compression Algebraic Falsifier (v2)
Supports arbitrary multi-generator subgroups U = <g1, g2, ...> <= Z_m^*
and both even m (row sums (2,0,0,0)) and odd m (row sums (1,1,1,1)) for Theorem 5 (b=4, n=4m+4).

100% exact integer arithmetic (zero floating point).
"""

import math
import time
from decimal import Decimal, getcontext
from collections import defaultdict
from deep_falsify import reachable_sums, check_2_compression, check_D_compression_L2

getcontext().prec = 50


def _dec_pi():
    def atan(x):
        x = Decimal(x)
        x2 = x * x
        term = x
        s = x
        k = 1
        while True:
            term = -term * x2
            add = term / (2 * k + 1)
            if abs(add) < Decimal("1e-45"):
                break
            s += add
            k += 1
        return s
    return 4 * (4 * atan(Decimal(1) / Decimal(5)) - atan(Decimal(1) / Decimal(239)))


def _dec_cos(x):
    x2 = x * x
    term = Decimal(1)
    s = term
    k = 1
    while True:
        term = -term * x2 / Decimal((2 * k - 1) * (2 * k))
        if abs(term) < Decimal("1e-45"):
            break
        s += term
        k += 1
    return s


_DEC_PI = None


def build_rigorous_psd_filter(n, d, U, shift_reps):
    """Build a 100% rigorous integer-enclosure filter for PSD(f^(d), u) <= n
    for all u in shift_reps using 50-digit Decimal Taylor series (B = 10^20)."""
    global _DEC_PI
    if _DEC_PI is None:
        _DEC_PI = _dec_pi()
    B = 10**20
    U_d = sorted({u % d for u in U})
    shift_orbs = []
    for s in shift_reps:
        orb = set()
        for u in U_d:
            v = (s * u) % d
            orb.add(v)
            orb.add((d - v) % d)
        shift_orbs.append(sorted(orb))

    freq_bounds = []
    for u in shift_reps:
        row_L, row_U = [], []
        for orb in shift_orbs:
            val = sum(_dec_cos(Decimal(2) * _DEC_PI * Decimal((u * s) % d) / Decimal(d)) for s in orb)
            scaled = val * Decimal(B)
            row_L.append(int(scaled.to_integral_value(rounding="ROUND_FLOOR")) - 10)
            row_U.append(int(scaled.to_integral_value(rounding="ROUND_CEILING")) + 10)
        freq_bounds.append((row_L, row_U))

    K_d = len(shift_reps)
    limit = n * B

    def psd_ok(full_paf):
        p0 = full_paf[0]
        for row_L, row_U in freq_bounds:
            lb = p0 * B
            for t in range(K_d):
                pt = full_paf[t + 1]
                lb += pt * (row_L[t] if pt >= 0 else row_U[t])
            if lb > limit:
                return False
        return True

    return psd_ok


def generate_subgroup(m, gens):
    """Generate exact subgroup U = <gens> <= Z_m^*."""
    U = {1}
    frontier = [1]
    while frontier:
        cur = frontier.pop()
        for g in gens:
            nxt = (cur * g) % m
            if nxt not in U:
                U.add(nxt)
                frontier.append(nxt)
    return sorted(U)


def get_orbits_U(m, U):
    """Orbits of U on Z_m."""
    seen = [False] * m
    orbits = []
    for x in range(m):
        if not seen[x]:
            orb = sorted({(x * u) % m for u in U})
            for y in orb:
                seen[y] = True
            orbits.append(orb)
    return orbits


def get_shift_orbits_U(m, U):
    """Nonzero shift orbits of <U, -1> on Z_m."""
    seen = [False] * m
    seen[0] = True
    reps = []
    sizes = []
    for t in range(1, m):
        if not seen[t]:
            orb = set()
            for u in U:
                v = (t * u) % m
                orb.add(v)
                orb.add((m - v) % m)
            for y in orb:
                seen[y] = True
            reps.append(t)
            sizes.append(len(orb))
    return reps, sizes


def check_shift_bounds_U(n, m, U, orbits, max_comp_size=16):
    """Exact diag_const and exact MaxCut lower bound on PAF(f, t) for all shift orbits."""
    elem_to_orb = [0] * m
    for idx, o in enumerate(orbits):
        for x in o:
            elem_to_orb[x] = idx

    reps, _ = get_shift_orbits_U(m, U)
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


def get_d_orbit_structure_U(m, U, orbits, d):
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

    U_d = sorted({u % d for u in U})
    shift_reps, shift_sizes = get_shift_orbits_U(d, U_d)

    paf_mats = []
    for s in shift_reps:
        mat = [[0] * r_d for _ in range(r_d)]
        for x in range(d):
            ci = res_to_cid[x]
            cj = res_to_cid[(x + s) % d]
            mat[ci][cj] += 1
        paf_mats.append(mat)

    return cosets, weights, shift_reps, shift_sizes, paf_mats


def enumerate_subset_by_sum(csizes, R_lists, target_sum, max_sq, max_tuples=400000):
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


def verify_witness_on_Zd(n, m, d, cosets, witness, expected_rsums):
    T0 = n - 4 * (m // d)
    Td = -4 * (m // d)
    full_seqs = []
    for a in witness:
        arr = [0] * d
        for cid, c in enumerate(cosets):
            for r in c:
                arr[r] = a[cid]
        full_seqs.append(arr)
    assert [sum(arr) for arr in full_seqs] == list(expected_rsums)
    for s in range(d):
        tot = sum(sum(arr[j] * arr[(j + s) % d] for j in range(d)) for arr in full_seqs)
        expected = T0 if s == 0 else Td
        assert tot == expected, f"Verification failed at d={d}, s={s}: got {tot}, expected {expected}"


def mitm_even_m(V2, V0, paf2_map, paf0_map, target_vec):
    """Exact MITM for row_sums = (2, 0, 0, 0). Returns (paf_quad_count, seq_quad_count, sample_witness)."""
    T0 = target_vec[0]
    dim = len(target_vec)
    min_p0_2 = V2[0][0]
    min_p0_0 = V0[0][0]
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
    return len(paf_quads), seq_quad_count, sample_witness


def mitm_odd_m(V1, paf1_map, target_vec):
    """Exact MITM for row_sums = (1, 1, 1, 1). Returns (paf_quad_count, seq_quad_count, sample_witness)."""
    T0 = target_vec[0]
    dim = len(target_vec)
    min_p0 = V1[0][0]
    max_p0_pair = T0 - 2 * min_p0

    pair11 = defaultdict(list)
    n1 = len(V1)
    for i in range(n1):
        vi = V1[i]
        vi0 = vi[0]
        if vi0 + vi0 > max_p0_pair:
            break
        for j in range(i, n1):
            vj = V1[j]
            if vi0 + vj[0] > max_p0_pair:
                break
            svec = tuple(vi[k] + vj[k] for k in range(dim))
            pair11[svec].append((i, j))

    paf_quads = set()
    seq_quad_count = 0
    sample_witness = None

    for k in range(n1):
        vk = V1[k]
        vk0 = vk[0]
        if vk0 + vk0 > max_p0_pair:
            break
        for l in range(k, n1):
            vl = V1[l]
            if vk0 + vl[0] > max_p0_pair:
                break
            need = tuple(target_vec[idx] - vk[idx] - vl[idx] for idx in range(dim))
            if need in pair11:
                for i, j in pair11[need]:
                    if j <= k:
                        qkey = (i, j, k, l)
                        if qkey not in paf_quads:
                            paf_quads.add(qkey)
                            # Multiset count for unordered 4-tuple of sequences
                            counts = defaultdict(int)
                            for idx_v in qkey:
                                counts[idx_v] += 1
                            mult = 1
                            for idx_v, rep in counts.items():
                                c = len(paf1_map[V1[idx_v]])
                                mult *= math.comb(c + rep - 1, rep)
                            seq_quad_count += mult
                            if sample_witness is None:
                                sample_witness = tuple(paf1_map[V1[idx_v]][0] for idx_v in qkey)
    return len(paf_quads), seq_quad_count, sample_witness


def analyze_divisor_exact_U(n, m, U, orbits, d, sols2, max_single_tuples=300000, max_paf_vecs=5000):
    t0 = time.time()
    is_even = (m % 2 == 0)
    row_sum_quad = (2, 0, 0, 0) if is_even else (1, 1, 1, 1)
    T0 = n - 4 * (m // d)
    Td = -4 * (m // d)

    cosets, weights, shift_reps, shift_sizes, paf_mats = get_d_orbit_structure_U(m, U, orbits, d)
    r_d = len(cosets)
    K_d = len(shift_reps)
    csizes = [len(c) for c in cosets]
    R_lists = [
        sorted(a for a in reachable_sums(weights[i]) if csizes[i] * a * a <= T0)
        for i in range(r_d)
    ]

    if r_d <= 12:
        ok_l2, l2_info = check_D_compression_L2(n, m, 4, orbits, d, row_sum_quad)
        if not ok_l2:
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "FALSIFIED_L2",
                "detail": f"Exact L2-norm obstruction: T0={T0}, max_single={l2_info['max_single_P0']}",
                "elapsed": time.time() - t0,
            }
    elif r_d > 14:
        return {
            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
            "status": "SKIPPED_LARGE_RD",
            "detail": f"r_d={r_d} > 14",
            "elapsed": time.time() - t0,
        }

    # Enumerate single sequences
    def enumerate_for_s(s_f, allowed_eo=None):
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

    paf_mats_used = paf_mats[:-1] if K_d >= 1 else []
    target_vec = (T0,) + (Td,) * (K_d - 1) if K_d >= 1 else (T0,)

    if is_even:
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

        ab2, seqs2 = enumerate_for_s(2, allowed_eo_2)
        ab0, seqs0 = enumerate_for_s(0, allowed_eo_0)
        if ab2 or ab0:
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "PASSED_L2_ONLY",
                "detail": f"Single-seq space > {max_single_tuples} (ab2={ab2}, ab0={ab0})",
                "elapsed": time.time() - t0,
            }
        canon0 = {}
        for a in seqs0:
            neg = tuple(-x for x in a)
            canon0[a if a <= neg else neg] = True
        seqs0 = list(canon0.keys())

        paf2_map = defaultdict(list)
        for a in seqs2:
            paf2_map[compute_paf_vector(a, csizes, paf_mats_used)].append(a)
        paf0_map = defaultdict(list)
        for a in seqs0:
            paf0_map[compute_paf_vector(a, csizes, paf_mats_used)].append(a)

        V2 = sorted(paf2_map.keys())
        V0 = sorted(paf0_map.keys())
        if not V2 or not V0:
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "FALSIFIED_SINGLE",
                "detail": f"Empty achievable set: |V2|={len(V2)}, |V0|={len(V0)}",
                "elapsed": time.time() - t0,
            }
        min_p0_2, min_p0_0 = V2[0][0], V0[0][0]
        V2 = [v for v in V2 if v[0] <= T0 - 3 * min_p0_0]
        V0 = [v for v in V0 if v[0] <= T0 - min_p0_2 - 2 * min_p0_0]
        if not V2 or not V0:
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "FALSIFIED_P0_BOUND",
                "detail": f"Tightened P0 bound killed all candidates",
                "elapsed": time.time() - t0,
            }
        if len(V0) > 1500 and K_d >= 2:
            psd_ok = build_rigorous_psd_filter(n, d, U, shift_reps)
            last_sz = shift_sizes[-1]
            def keep_vec(v, sf):
                rem = sf * sf - v[0] - sum(shift_sizes[t] * v[t + 1] for t in range(K_d - 1))
                assert rem % last_sz == 0
                return psd_ok(v + (rem // last_sz,))
            V2 = [v for v in V2 if keep_vec(v, 2)]
            V0 = [v for v in V0 if keep_vec(v, 0)]
            if not V2 or not V0:
                return {
                    "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                    "status": "FALSIFIED_PSD_BOUND",
                    "detail": f"Rigorous PSD <= {n} bound killed all candidates",
                    "elapsed": time.time() - t0,
                }
        if len(V0) > max_paf_vecs:
            # Try 3D projection on non-unit shifts (shifts with gcd(s, d) > 1) if available!
            div_shifts = [idx + 1 for idx, s in enumerate(shift_reps[:-1]) if math.gcd(s, d) > 1]
            if len(div_shifts) >= 2:
                idxs = (0, div_shifts[0], div_shifts[1])
                P2 = sorted({tuple(v[i] for i in idxs) for v in V2})
                P0 = sorted({tuple(v[i] for i in idxs) for v in V0})
                if len(P0) <= max_paf_vecs:
                    p2m = {p: [p] for p in P2}
                    p0m = {p: [p] for p in P0}
                    t_sub = tuple(target_vec[i] for i in idxs)
                    pq, _, _ = mitm_even_m(P2, P0, p2m, p0m, t_sub)
                    if pq == 0:
                        return {
                            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                            "status": "FALSIFIED_EXACT_PAF_3D",
                            "detail": (
                                f"EXACT 3D PAF OBSTRUCTION mod d={d} on (P0, s={shift_reps[div_shifts[0]-1]}, s={shift_reps[div_shifts[1]-1]}): "
                                f"|P2|={len(P2)}, |P0|={len(P0)}, 0 feasible quadruples!"
                            ),
                            "elapsed": time.time() - t0,
                        }
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "PASSED_L2_SKIPPED_MITM",
                "detail": f"|seqs2|={len(seqs2)}, |seqs0|={len(seqs0)}, |V2|={len(V2)}, |V0|={len(V0)}",
                "elapsed": time.time() - t0,
            }

        pq, sq, wit = mitm_even_m(V2, V0, paf2_map, paf0_map, target_vec)
        if pq == 0:
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "FALSIFIED_EXACT_PAF",
                "detail": f"EXACT PAF OBSTRUCTION mod d={d}: |V2|={len(V2)}, |V0|={len(V0)}, 0 feasible quadruples!",
                "elapsed": time.time() - t0,
            }
        verify_witness_on_Zd(n, m, d, cosets, wit, row_sum_quad)
        return {
            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
            "status": "SURVIVED_EXACT_PAF",
            "paf_quads": pq, "seq_quads": sq,
            "detail": f"|V2|={len(V2)}, |V0|={len(V0)} -> {pq} PAF-quads, {sq} seq-quads",
            "elapsed": time.time() - t0,
        }
    else:
        # Odd m: row_sum_quad = (1, 1, 1, 1)
        ab1, seqs1 = enumerate_for_s(1, None)
        if ab1:
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "PASSED_L2_ONLY",
                "detail": f"Single-seq space > {max_single_tuples}",
                "elapsed": time.time() - t0,
            }
        paf1_map = defaultdict(list)
        for a in seqs1:
            paf1_map[compute_paf_vector(a, csizes, paf_mats_used)].append(a)
        V1 = sorted(paf1_map.keys())
        if not V1:
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "FALSIFIED_SINGLE",
                "detail": "Empty achievable set |V1|=0",
                "elapsed": time.time() - t0,
            }
        min_p0 = V1[0][0]
        V1 = [v for v in V1 if v[0] <= T0 - 3 * min_p0]
        if not V1:
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "FALSIFIED_P0_BOUND",
                "detail": "Tightened P0 bound killed all candidates",
                "elapsed": time.time() - t0,
            }
        if len(V1) > 1500 and K_d >= 2:
            psd_ok = build_rigorous_psd_filter(n, d, U, shift_reps)
            last_sz = shift_sizes[-1]
            def keep_vec1(v):
                rem = 1 - v[0] - sum(shift_sizes[t] * v[t + 1] for t in range(K_d - 1))
                assert rem % last_sz == 0
                return psd_ok(v + (rem // last_sz,))
            V1 = [v for v in V1 if keep_vec1(v)]
            if not V1:
                return {
                    "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                    "status": "FALSIFIED_PSD_BOUND",
                    "detail": f"Rigorous PSD <= {n} bound killed all candidates",
                    "elapsed": time.time() - t0,
                }
        if len(V1) > max_paf_vecs:
            div_shifts = [idx + 1 for idx, s in enumerate(shift_reps[:-1]) if math.gcd(s, d) > 1]
            if len(div_shifts) >= 2:
                idxs = (0, div_shifts[0], div_shifts[1])
                P1 = sorted({tuple(v[i] for i in idxs) for v in V1})
                if len(P1) <= max_paf_vecs:
                    p1m = {p: [p] for p in P1}
                    t_sub = tuple(target_vec[i] for i in idxs)
                    pq, _, _ = mitm_odd_m(P1, p1m, t_sub)
                    if pq == 0:
                        return {
                            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                            "status": "FALSIFIED_EXACT_PAF_3D",
                            "detail": (
                                f"EXACT 3D PAF OBSTRUCTION mod d={d} on (P0, s={shift_reps[div_shifts[0]-1]}, s={shift_reps[div_shifts[1]-1]}): "
                                f"|P1|={len(P1)}, 0 feasible quadruples!"
                            ),
                            "elapsed": time.time() - t0,
                        }
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "PASSED_L2_SKIPPED_MITM",
                "detail": f"|seqs1|={len(seqs1)}, |V1|={len(V1)}",
                "elapsed": time.time() - t0,
            }

        pq, sq, wit = mitm_odd_m(V1, paf1_map, target_vec)
        if pq == 0:
            return {
                "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
                "status": "FALSIFIED_EXACT_PAF",
                "detail": f"EXACT PAF OBSTRUCTION mod d={d}: |V1|={len(V1)}, 0 feasible quadruples!",
                "elapsed": time.time() - t0,
            }
        verify_witness_on_Zd(n, m, d, cosets, wit, row_sum_quad)
        return {
            "d": d, "r_d": r_d, "K_d": K_d, "T0": T0, "Td": Td,
            "status": "SURVIVED_EXACT_PAF",
            "paf_quads": pq, "seq_quads": sq,
            "detail": f"|V1|={len(V1)} -> {pq} PAF-quads, {sq} seq-quads",
            "elapsed": time.time() - t0,
        }


def main():
    targets = [
        # Non-cyclic r = 36
        (4792, 1197, [5, 58],       "1. n=4792, m=1197=3^2*7*19, gens=[5,58]"),
        (4792, 1197, [47, 4],       "2. n=4792, m=1197=3^2*7*19, gens=[47,4]"),
        (7180, 1794, [29, 25],      "3. n=7180, m=1794=2*3*13*23, gens=[29,25]"),
        (9388, 2346, [25, 35],      "4. n=9388, m=2346=2*3*17*23, gens=[25,35]"),
        (5308, 1326, [43, 35, 53],  "5. n=5308, m=1326=2*3*13*17, gens=[43,35,53]"),
        # Cyclic 37 <= r <= 48
        (2396,  598, [25],          "6. n=2396, m=598=2*13*23, g=25"),
        (9388, 2346, [59],          "7. n=9388, m=2346=2*3*17*23, g=59"),
        (4412, 1102, [17],          "8. n=4412, m=1102=2*19*29, g=17"),
        (7180, 1794, [95],          "9. n=7180, m=1794=2*3*13*23, g=95"),
    ]

    summary = []
    for n, m, gens, label in targets:
        U = generate_subgroup(m, gens)
        orbits = get_orbits_U(m, U)
        shift_reps, _ = get_shift_orbits_U(m, U)
        r = len(orbits)
        K = len(shift_reps)
        sols2 = check_2_compression(n, m, orbits, (2, 0, 0, 0)) if m % 2 == 0 else None
        print("=" * 100, flush=True)
        print(f"{label}  (|U|={len(U)}, r={r}, K={K}, 2-comp={len(sols2) if sols2 is not None else 'odd m'})", flush=True)
        print("=" * 100, flush=True)

        bad_shifts, all_shifts = check_shift_bounds_U(n, m, U, orbits)
        max_D_rec = max(all_shifts, key=lambda x: x["D"])
        print(f"  [Shift Bounds] max D(t)={max_D_rec['D']} at t={max_D_rec['t']} (m/2={m/2})", flush=True)
        for rec in bad_shifts:
            eff = rec["exact_min"] if rec["exact_min"] is not None else rec["triv_min"]
            print(
                f"    *** FALSIFIED BY SHIFT t={rec['t']} (gcd={rec['gcd']}): "
                f"D={rec['D']}, triv_min={rec['triv_min']}, exact_MaxCut_min={rec['exact_min']}, 4*min={4*eff} > -4 ***",
                flush=True,
            )
        if bad_shifts:
            print(f"--> VERDICT: *** FALSIFIED BY SHIFT BOUND ***\n", flush=True)
            summary.append((label, "FALSIFIED (Shift Bound)", f"t={bad_shifts[0]['t']}, D={bad_shifts[0]['D']}, exact_min={bad_shifts[0]['exact_min']}"))
            continue

        divs = [d for d in range(2, m) if m % d == 0]
        falsified_reason = None
        tightest = None
        for d in divs:
            res = analyze_divisor_exact_U(n, m, U, orbits, d, sols2)
            print(
                f"  d={d:4d} (r_d={res['r_d']:2d}, K_d={res['K_d']:2d}, T0={res['T0']:4d}, Td={res['Td']:5d}) "
                f"[{res['status']:22s}] ({res['elapsed']:.2f}s) {res['detail']}",
                flush=True,
            )
            if res["status"].startswith("FALSIFIED"):
                falsified_reason = f"d={d}: {res['status']} ({res['detail']})"
                break
            if res["status"] == "SURVIVED_EXACT_PAF":
                if tightest is None or (res["seq_quads"], -d) < (tightest["seq_quads"], -tightest["d"]):
                    tightest = res
        if falsified_reason:
            print(f"--> VERDICT: *** FALSIFIED *** ({falsified_reason})\n", flush=True)
            summary.append((label, "FALSIFIED (Composite PAF)", falsified_reason))
        else:
            tinfo = f"tightest d={tightest['d']} ({tightest['paf_quads']} PAF-quads, {tightest['seq_quads']} seq-quads)" if tightest else "none"
            print(f"--> VERDICT: SURVIVED ({tinfo})\n", flush=True)
            summary.append((label, "SURVIVED", tinfo))

    print("\n" + "=" * 100, flush=True)
    print("FINAL SUMMARY OF ALL 9 NEW TARGETS", flush=True)
    print("=" * 100, flush=True)
    for lbl, st, info in summary:
        print(f"  {lbl:50s} | {st:26s} | {info}", flush=True)


if __name__ == "__main__":
    main()
