#!/usr/bin/env python3
"""
Expanded Exact-Integer Algebraic Falsification Scanner for Open Hadamard Orders (n <= 10,000).

Scans two expanded regimes across all 85 live UNKNOWN orders (from deep_falsify.UNKNOWN_85):
  (a) Non-cyclic subgroups U <= Z_m^* with r <= 36 orbits for BOTH:
      - Theorem 5 (b = 4, m = (n - 4) / 4)
      - Theorem 3 (b = 0, m = n / 4)
  (b) Cyclic subgroups U = <g> <= Z_m^* with 37 <= r <= 48 orbits for Theorem 5 (b = 4).

All algebraic checks use 100% exact integer arithmetic (zero floating-point).
Does not modify deep_falsify.py; imports shared exact helpers from it.
"""

import itertools
import math
import time
from collections import defaultdict, deque

from deep_falsify import (
    UNKNOWN_85,
    factor,
    ord_mod,
    reachable_sums,
    sum_of_4_in_lists,
    sum_of_4_tuples,
    get_target_row_sum_quadruples,
    check_2_compression,
    check_D_compression_L2,
    find_cyclic_subgroups_with_r_le,
    full_falsify_candidate,
)


# ---------------------------------------------------------------------------
# General Subgroup U <= Z_m^* Enumeration & Orbit Helpers
# ---------------------------------------------------------------------------

def get_orbits_U(m, U):
    """Return orbits of an arbitrary subgroup U <= Z_m^* acting on Z_m."""
    seen = [False] * m
    orbits = []
    for x in range(m):
        if not seen[x]:
            orb = sorted({(x * u) % m for u in U})
            for v in orb:
                seen[v] = True
            orbits.append(orb)
    return orbits


def get_shift_reps_U(m, U):
    """Return shift orbit representatives of (+/- U) on Z_m \\ {0}."""
    U_pm = U | {(m - u) % m for u in U}
    seen = [False] * m
    seen[0] = True
    reps = []
    for d in range(1, m):
        if not seen[d]:
            reps.append(d)
            for u in U_pm:
                seen[(d * u) % m] = True
    return reps


def get_shift_count_U(m, U):
    """Return number of distinct autocorrelation shift orbits K of (+/- U) on Z_m \\ {0}."""
    return len(get_shift_reps_U(m, U))


def check_diag_const(m, U, b, orbits, d_reps=None):
    """Exact Stage-0 diagonal-constant obstruction check:
    For any shift d in {1..m-1}, let D(d) = #{x in Z_m : orb(x) == orb(x + d)}.
    Since f(x)*f(x+d) = +1 whenever orb(x) == orb(x+d), every U-invariant sequence satisfies
      PAF_f(d) >= 2*D(d) - m,
    so sum_{s=0}^3 PAF_s(d) >= 4 * (2*D(d) - m).
    Any shift d with 4 * (2*D(d) - m) > -b is strictly impossible.
    Returns (ok, msg)."""
    if d_reps is None:
        d_reps = get_shift_reps_U(m, U)
    orb_id = [0] * m
    for idx, o in enumerate(orbits):
        for x in o:
            orb_id[x] = idx
    max_diag = 0
    worst_d = None
    for d in d_reps:
        diag = 0
        for x in range(m):
            if orb_id[x] == orb_id[(x + d) % m]:
                diag += 1
        if diag > max_diag:
            max_diag = diag
            worst_d = d
        lb = 4 * (2 * diag - m)
        if lb > -b:
            return False, f"FALSIFIED by Stage-0 diag_const at d={d} (D(d)={diag}, m={m}, 4*(2*D-m)={lb} > {-b})"
    return True, f"passed diag_const (max D(d)={max_diag} at d={worst_d})"


def is_cyclic_subgroup(m, U):
    """Exact check whether finite abelian subgroup U <= Z_m^* is cyclic.
    Returns (True, smallest_generator) or (False, None)."""
    sz = len(U)
    if sz == 1:
        return True, 1
    primes = list(factor(sz).keys())
    for u in sorted(U):
        if all(pow(u, sz // p, m) != 1 for p in primes):
            return True, u
    return False, None


def find_minimal_generators(m, U):
    """Find a small generating set for subgroup U <= Z_m^*."""
    is_cyc, gen = is_cyclic_subgroup(m, U)
    if is_cyc:
        return [gen]
    U_sorted = sorted(U)
    gen_sub = {1}
    gens = []
    while len(gen_sub) < len(U):
        best_g = None
        best_ord = 1
        for g in U_sorted:
            if g in gen_sub:
                continue
            # Order of g modulo gen_sub
            k = 1
            cur = g
            while cur not in gen_sub:
                cur = (cur * g) % m
                k += 1
            if k > best_ord:
                best_ord = k
                best_g = g
        gens.append(best_g)
        # Expand gen_sub by powers of best_g
        pows = [1]
        for _ in range(best_ord - 1):
            pows.append((pows[-1] * best_g) % m)
        new_sub = set()
        for a in gen_sub:
            for pw in pows:
                new_sub.add((a * pw) % m)
        gen_sub = new_sub
    assert gen_sub == U
    return gens


def maximal_subgroups_of_abelian(m, H, min_size=1):
    """Yield all maximal subgroups M < H (of prime index p | |H|) with |M| >= min_size.
    Uses exact linear algebra over F_p on H / H^p."""
    sz = len(H)
    if sz <= min_size:
        return
    H_sorted = sorted(H)
    primes = sorted(factor(sz).keys())

    for p in primes:
        if sz // p < min_size:
            continue
        # H^p = {h^p mod m : h in H}
        Hp = {pow(h, p, m) for h in H_sorted}
        # Build F_p-coordinates for H / H^p
        coords = {b: () for b in Hp}
        B = set(Hp)
        while len(B) < sz:
            x = None
            for cand in H_sorted:
                if cand not in B:
                    x = cand
                    break
            x_pows = [1]
            for _ in range(p - 1):
                x_pows.append((x_pows[-1] * x) % m)
            new_coords = {}
            for b, c in coords.items():
                for k, xp in enumerate(x_pows):
                    new_coords[(b * xp) % m] = c + (k,)
            coords = new_coords
            B = set(coords.keys())

        d = len(next(iter(coords.values())))
        assert len(Hp) * (p ** d) == sz

        # Projective vectors in P^{d-1}(F_p)
        for first_nz in range(d):
            tail_len = d - 1 - first_nz
            for tail in itertools.product(range(p), repeat=tail_len):
                v = (0,) * first_nz + (1,) + tail
                M = frozenset(
                    h for h, c in coords.items()
                    if sum(vi * ci for vi, ci in zip(v, c)) % p == 0
                )
                assert len(M) == sz // p
                yield M


def find_all_subgroups_with_r_le(m, max_r=36):
    """Enumerate ALL distinct subgroups U <= Z_m^* with r(U) <= max_r orbits on Z_m.
    Uses top-down BFS from G = Z_m^* via maximal subgroups, exploiting exact monotonicity:
    U1 <= U2 => r(U2) <= r(U1).
    Returns list of (U_frozenset, r, is_cyclic, gens)."""
    G = frozenset(x for x in range(1, m) if math.gcd(x, m) == 1)
    gcd_u = {u: math.gcd(u - 1, m) for u in G}

    def orbit_count(H):
        tot = sum(gcd_u[u] for u in H)
        assert tot % len(H) == 0
        return tot // len(H)

    r_G = orbit_count(G)
    if r_G > max_r:
        return []

    # Necessary lower bound: r >= 1 + (m - 1) / |U| => |U| >= ceil((m - 1) / (max_r - 1))
    min_u = (m - 1 + max_r - 2) // (max_r - 1)

    visited = {G: r_G}
    queue = deque([G])

    while queue:
        H = queue.popleft()
        for M in maximal_subgroups_of_abelian(m, H, min_size=min_u):
            if M in visited:
                continue
            r_M = orbit_count(M)
            if r_M <= max_r:
                visited[M] = r_M
                queue.append(M)

    results = []
    for U, r in visited.items():
        is_cyc, gen = is_cyclic_subgroup(m, U)
        gens = [gen] if is_cyc else find_minimal_generators(m, U)
        results.append((U, r, is_cyc, gens))
    results.sort(key=lambda x: (x[1], -len(x[0]), x[3]))
    return results


# ---------------------------------------------------------------------------
# Generalized Exact Algebraic Falsification Checks for Arbitrary U <= Z_m^*
# ---------------------------------------------------------------------------

def check_4_compression_exact_U(n, m, U, b, orbits, row_sum_quad):
    """Exact 4-compression simultaneous (k=m/2, k=m/4) PSD check when 4 | m,
    generalized to arbitrary subgroup U <= Z_m^*."""
    if m % 4 != 0:
        return True, "N/A"
    sum_s2 = sum(s * s for s in row_sum_quad)
    T0 = (sum_s2 + 3 * n) // 4
    by_m4 = defaultdict(list)
    for o in orbits:
        reps = tuple(sorted({x % 4 for x in o}))
        by_m4[reps].append(len(o) // len(reps))
    has_3 = any(u % 4 == 3 for u in U)
    if has_3:
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
            return False, "FALSIFIED by 4-compression exact (m/2, m/4) PSD obstruction (3 in U mod 4)"
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
            return False, "FALSIFIED by 4-compression exact (m/2, m/4) PSD obstruction (U mod 4 = {1})"
        return True, "passed 4-comp (exact (m/2, m/4) PSD)"


def check_prime_q_compression_exact_U(n, m, U, b, orbits, q, row_sum_quad, max_tuples=40000):
    """Exact q-compression PSD/PAF check for an odd prime q | m,
    generalized to arbitrary subgroup U <= Z_m^*."""
    U_q = {u % q for u in U}
    h = len(U_q)
    assert (q - 1) % h == 0
    e = (q - 1) // h
    sum_s2 = sum(s * s for s in row_sum_quad)
    T0 = (sum_s2 + (q - 1) * n) // q
    Td = (sum_s2 - n) // q

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

    # Step 2: e = 1 (surjects onto F_q^*)
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

    # Step 4: e >= 3 — exact full PAF vector meet-in-the-middle when tractable
    U_q_pm = U_q | {(q - u) % q for u in U_q}
    seen_d = [False] * q
    seen_d[0] = True
    d_reps = []
    for d in range(1, q):
        if not seen_d[d]:
            d_reps.append(d)
            for u in U_q_pm:
                seen_d[(d * u) % q] = True

    res_to_cid = [0] * q
    for idx, c in enumerate(cosets, start=1):
        for r in c:
            res_to_cid[r] = idx

    num_c = e + 1
    paf_mats = []
    for d in d_reps:
        mat = [[0] * num_c for _ in range(num_c)]
        for x in range(q):
            ci = res_to_cid[x]
            cj = res_to_cid[(x + d) % q]
            mat[ci][cj] += 1
        paf_mats.append(mat)

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
        max_len = max(len(x) for x in lists)
        if max_len <= 1500:
            wit = sum_of_4_tuples(lists, target_vec)
            if wit is None:
                return False, f"FALSIFIED by q={q} (e={e}, h={h}) exact full PAF vector obstruction", l2_info
            return True, f"passed q={q} (e={e}, h={h}, exact full PAF, |V|={[len(x) for x in lists]})", l2_info
        else:
            return True, f"passed q={q} (e={e}, h={h}, exact L2; full PAF skipped |V|={max_len})", l2_info
    return True, f"passed q={q} (e={e}, h={h}, exact L2; tuple space > {max_tuples})", l2_info


def check_2q_simultaneous_e2_U(n, m, U, b, orbits, q, row_sum_quad):
    """Exact simultaneous (k=m/2, k=m/q, k=(q+2)m/(2q)) PSD check when 2q | m and e=2,
    generalized to arbitrary subgroup U <= Z_m^*."""
    if m % 2 != 0:
        return True, "N/A"
    U_q = {u % q for u in U}
    h = len(U_q)
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


def full_falsify_subgroup(n, m, U, gens, b, extra_divisor_l2=False):
    """Run all exact algebraic falsification checks on (n, m, U, b) for arbitrary U <= Z_m^*.
    Stage 0: exact diag_const lower bound 4*(2*D(d) - m) <= -b for all shifts d.
    Returns (passed, summary_dict)."""
    orbits = get_orbits_U(m, U)
    r = len(orbits)
    d_reps = get_shift_reps_U(m, U)
    K = len(d_reps)
    U_size = len(U)

    # Stage 0: Exact diag_const obstruction check
    ok_diag, msg_diag = check_diag_const(m, U, b, orbits, d_reps=d_reps)
    if not ok_diag:
        return False, {
            "n": n, "m": m, "gens": gens, "b": b, "U": U_size, "r": r, "K": K,
            "reason": msg_diag,
            "stage0_killed": True,
        }

    row_quads = get_target_row_sum_quadruples(n, m, b, orbits)
    if not row_quads:
        return False, {
            "n": n, "m": m, "gens": gens, "b": b, "U": U_size, "r": r, "K": K,
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
            ok_4, msg_4 = check_4_compression_exact_U(n, m, U, b, orbits, rq)
            rep["checks"].append(msg_4)
            if not ok_4:
                ok_rq = False
                rep["fail_reason"] = msg_4

        # 2. Exact q-compression for every odd prime q | m
        if ok_rq:
            for q in odd_primes:
                ok_q, msg_q, info_q = check_prime_q_compression_exact_U(n, m, U, b, orbits, q, rq)
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
                ok_sim, msg_sim = check_2q_simultaneous_e2_U(n, m, U, b, orbits, q, rq)
                if msg_sim != "N/A":
                    rep["checks"].append(msg_sim)
                    if not ok_sim:
                        ok_rq = False
                        rep["fail_reason"] = msg_sim
                        break
                else:
                    rep["checks"].append(f"passed 2q={2*q} (exact L2)")

        # 4. Optional bonus higher composite divisor L2 checks (e.g. 8, p^2, 4q, q1*q2)
        if ok_rq and extra_divisor_l2:
            # Find proper divisors D | m not already checked as 2, 4, q, 2q
            already = {1, 2, 4} | set(odd_primes) | {2 * q for q in odd_primes}
            divs = [1]
            for p, e in fac.items():
                divs = [d * (p ** k) for d in divs for k in range(e + 1)]
            extra_divs = sorted(d for d in divs if d not in already and d < m)
            for D in extra_divs:
                # Only run if number of cosets of U mod D on Z_D is small (<= 10)
                coset_set = {tuple(sorted({x % D for x in o})) for o in orbits}
                if len(coset_set) <= 10:
                    ok_D, info_D = check_D_compression_L2(n, m, b, orbits, D, rq)
                    if not ok_D:
                        ok_rq = False
                        msg_D = f"FALSIFIED by D={D} exact L2-norm obstruction (T0={info_D['T0']}, max_single={info_D['max_single_P0']})"
                        rep["checks"].append(msg_D)
                        rep["fail_reason"] = msg_D
                        break
                    else:
                        rep["checks"].append(f"passed D={D} (exact L2, cosets={len(coset_set)})")

        quad_reports.append(rep)
        if ok_rq:
            surviving_quads.append(rep)

    if surviving_quads:
        return True, {
            "n": n, "m": m, "gens": gens, "b": b, "U": U_size, "r": r, "K": K,
            "surviving_quads": surviving_quads,
            "all_reports": quad_reports,
        }
    else:
        return False, {
            "n": n, "m": m, "gens": gens, "b": b, "U": U_size, "r": r, "K": K,
            "reason": quad_reports[0].get("fail_reason", "FALSIFIED"),
            "all_reports": quad_reports,
        }


def fmt_fac(m):
    return "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factor(m).items()))


def sanity_check_enumerator():
    """Verify top-down subgroup enumerator matches deep_falsify.find_cyclic_subgroups_with_r_le."""
    for m in [1892, 1666, 1102, 738]:
        cyc_old = find_cyclic_subgroups_with_r_le(m, max_r=36)
        all_new = find_all_subgroups_with_r_le(m, max_r=36)
        cyc_new = [(u, r) for (u, r, is_c, _) in all_new if is_c]
        old_sets = set()
        for g, u_sz, r in cyc_old:
            elems = set()
            cur = 1
            while cur not in elems:
                elems.add(cur)
                cur = (cur * g) % m
            old_sets.add((frozenset(elems), r))
        new_sets = set(cyc_new)
        assert old_sets == new_sets, f"Mismatch for m={m}: old={len(old_sets)}, new={len(new_sets)}"
    print("[Sanity check] Top-down subgroup enumerator matches deep_falsify cyclic enumerator 100%!", flush=True)


def print_survivors(title, survivors, is_cyclic_mode=False):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if not survivors:
        print("  (NONE)")
        return
    # Rank by (r, K, n)
    survivors.sort(key=lambda x: (x["r"], x["K"], x["n"]))
    for idx, info in enumerate(survivors, 1):
        fac_str = fmt_fac(info["m"])
        gens_str = f"g={info['g']}" if is_cyclic_mode else f"gens={info['gens']}"
        n_rq = len(info["surviving_quads"])
        if info["m"] % 2 == 0:
            all_2c = []
            for q in info["surviving_quads"]:
                all_2c.extend(q.get("2_comp_sols", []))
            n_2c = len(all_2c)
            if n_2c <= 3:
                c2_str = f"{n_2c} {all_2c}"
            else:
                c2_str = f"{n_2c}"
        else:
            c2_str = "N/A"
        print(
            f"  #{idx:2d}  n={info['n']:4d}  m={info['m']:4d} ({fac_str:14s})  b={info['b']}  "
            f"{gens_str:18s}  |U|={info['U']:3d}  r={info['r']:2d}  K={info['K']:2d}  "
            f"rq={n_rq}  2-comp={c2_str}"
        )


def main():
    t0 = time.time()
    sanity_check_enumerator()

    # =========================================================================
    # REGIME (a): Non-cyclic subgroups U <= Z_m^* with r <= 36 (b=4 and b=0)
    # =========================================================================
    print("\n" + "=" * 90)
    print("REGIME (a): NON-CYCLIC SUBGROUPS U <= Z_m^* WITH r <= 36 (Theorem 5 b=4 AND Theorem 3 b=0)")
    print("=" * 90, flush=True)

    for b, label in [(4, "Theorem 5 (b=4)"), (0, "Theorem 3 (b=0)")]:
        t_reg = time.time()
        total_noncyc = 0
        stage0_killed = 0
        falsified = 0
        survivors = []

        for n in UNKNOWN_85:
            m = (n - b) // 4
            subgroups = find_all_subgroups_with_r_le(m, max_r=36)
            for U, r, is_cyc, gens in subgroups:
                if is_cyc:
                    continue
                total_noncyc += 1
                passed, info = full_falsify_subgroup(n, m, U, gens, b, extra_divisor_l2=False)
                if passed:
                    passed_extra, info_extra = full_falsify_subgroup(n, m, U, gens, b, extra_divisor_l2=True)
                    info["extra_passed"] = passed_extra
                    info["extra_info"] = info_extra
                    survivors.append(info)
                else:
                    falsified += 1
                    if info.get("stage0_killed"):
                        stage0_killed += 1

        elapsed = time.time() - t_reg
        print(
            f"\n[{label} Non-Cyclic r <= 36] Checked {total_noncyc} non-cyclic (n, m, U) in {elapsed:.1f}s | "
            f"{falsified} FALSIFIED ({stage0_killed} by Stage-0 diag_const) | {len(survivors)} SURVIVED",
            flush=True,
        )
        print_survivors(f"SURVIVORS: {label} Non-Cyclic r <= 36 (ranked by (r, K))", survivors, is_cyclic_mode=False)

        if survivors:
            print("\n  --- Bonus higher-divisor L2 check on survivors ---")
            for info in survivors:
                ep = info["extra_passed"]
                status = "SURVIVES extra D-L2" if ep else f"CAUGHT by extra D-L2: {info['extra_info'].get('reason')}"
                print(f"    n={info['n']} m={info['m']} gens={info['gens']} r={info['r']} K={info['K']} -> {status}")

    # =========================================================================
    # REGIME (b): Cyclic subgroups U = <g> with 37 <= r <= 48 (Theorem 5, b=4)
    # =========================================================================
    print("\n" + "=" * 90)
    print("REGIME (b): CYCLIC SUBGROUPS U = <g> WITH 37 <= r <= 48 (Theorem 5, b=4)")
    print("=" * 90, flush=True)

    t_reg = time.time()
    total_cyc_37_48 = 0
    stage0_killed_b = 0
    falsified_b = 0
    survivors_b = []

    for n in UNKNOWN_85:
        m = (n - 4) // 4
        subgroups = find_cyclic_subgroups_with_r_le(m, max_r=48)
        for g, u_sz, r in subgroups:
            if r < 37:
                continue
            total_cyc_37_48 += 1
            U = frozenset(pow(g, k, m) for k in range(u_sz))
            passed, info = full_falsify_subgroup(n, m, U, [g], 4, extra_divisor_l2=False)
            if passed:
                info["g"] = g
                passed_extra, info_extra = full_falsify_subgroup(n, m, U, [g], 4, extra_divisor_l2=True)
                info["extra_passed"] = passed_extra
                info["extra_info"] = info_extra
                survivors_b.append(info)
            else:
                falsified_b += 1
                if info.get("stage0_killed"):
                    stage0_killed_b += 1

    elapsed = time.time() - t_reg
    print(
        f"\n[Theorem 5 (b=4) Cyclic 37 <= r <= 48] Checked {total_cyc_37_48} cyclic (n, m, <g>) in {elapsed:.1f}s | "
        f"{falsified_b} FALSIFIED ({stage0_killed_b} by Stage-0 diag_const) | {len(survivors_b)} SURVIVED",
        flush=True,
    )
    print_survivors("SURVIVORS: Theorem 5 (b=4) Cyclic 37 <= r <= 48 (ranked by (r, K))", survivors_b, is_cyclic_mode=True)

    if survivors_b:
        print("\n  --- Bonus higher-divisor L2 check on Cyclic 37 <= r <= 48 survivors ---")
        for info in survivors_b:
            ep = info["extra_passed"]
            status = "SURVIVES extra D-L2" if ep else f"CAUGHT by extra D-L2: {info['extra_info'].get('reason')}"
            print(f"    n={info['n']} m={info['m']} g={info['g']} r={info['r']} K={info['K']} -> {status}")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
