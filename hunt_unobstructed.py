#!/usr/bin/env python3
"""
Hunt for the smallest-r TRULY UNOBSTRUCTED Hadamard targets across all 85 live unknown orders:
  1. Theorem 5 (b=4, n=4m+4), cyclic 49 <= r <= 72
  2. Theorem 3 (b=0, n=4m),   cyclic 37 <= r <= 64

Applies:
  - Fast exact algebraic falsifier from deep_falsify.py
  - Exact diag_const + Shift MaxCut lower bound (generalized for both b=4 and b=0)
  - Exact composite d-compression PAF + 50-digit rigorous Decimal PSD <= n filter (for Theorem 5)
"""

import math
import time
from collections import defaultdict
from deep_falsify import (
    UNKNOWN_85,
    factor,
    carmichael_lambda,
    get_orbits,
    get_shift_count,
    ord_mod,
    full_falsify_candidate,
    check_2_compression,
)
from check_composites_v2 import (
    generate_subgroup,
    get_orbits_U,
    get_shift_orbits_U,
    analyze_divisor_exact_U,
)


def find_cyclic_subgroups_in_range(m, min_r, max_r):
    """Find all distinct cyclic subgroups U = <g> <= Z_m^* with min_r <= r <= max_r."""
    lam = carmichael_lambda(m)
    min_u = (m - 1 + max_r - 2) // (max_r - 1)
    if lam < min_u:
        return []
    seen = {}
    for g in range(1, m):
        if math.gcd(g, m) != 1:
            continue
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
        if key in seen:
            continue
        r = sum(math.gcd(u - 1, m) for u in elems) // len(elems)
        if min_r <= r <= max_r:
            seen[key] = (g, len(elems), r)
    return sorted(seen.values(), key=lambda x: (x[2], -x[1], x[0]))


def check_shift_bounds_general(n, m, b, U, orbits, max_comp_size=14):
    """Exact diag_const + Shift MaxCut lower bound for b=4 (target -4) or b=0 (target 0)."""
    target_sum = -4 if b == 4 else 0
    elem_to_orb = [0] * m
    for idx, o in enumerate(orbits):
        for x in o:
            elem_to_orb[x] = idx
    reps, _ = get_shift_orbits_U(m, U)
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
        triv_min = 2 * D - m
        if 4 * triv_min > target_sum:
            return False, f"diag_const t={t} D={D} triv_min={triv_min}"
        # Connected components
        undir = defaultdict(set)
        for u in adj:
            for v in adj[u]:
                undir[u].add(v)
                undir[v].add(u)
        vis = set()
        comps = []
        ok_size = True
        for u in undir:
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
                if len(comp) > max_comp_size:
                    ok_size = False
                    break
                comps.append(comp)
        if ok_size:
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
                    for a, b_idx, w in edges:
                        sa = (mask >> (a - 1)) & 1 if a > 0 else 0
                        sb = (mask >> (b_idx - 1)) & 1 if b_idx > 0 else 0
                        if sa != sb:
                            cut += w
                    if cut > best:
                        best = cut
                total_mc += best
            exact_min = m - 2 * total_mc
            if 4 * exact_min > target_sum:
                return False, f"ShiftMaxCut t={t} D={D} exact_min={exact_min}"
    return True, "ok"


def m_structure_tag(m):
    fac = factor(m)
    primes = sorted(fac.keys())
    if len(primes) == 1 and fac[primes[0]] == 1:
        return "PRIME"
    if primes[0] == 2 and len(primes) == 2 and fac[primes[1]] == 1:
        return f"2^{fac[2]}*p"
    return "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(fac.items()))


def scan_theorem5(min_r=49, max_r=72):
    print("=" * 100, flush=True)
    print(f"PART 1: THEOREM 5 (b=4, n=4m+4) CYCLIC {min_r} <= r <= {max_r} ACROSS ALL 85 OPEN ORDERS", flush=True)
    print("=" * 100, flush=True)
    cands = []
    for n in UNKNOWN_85:
        m = (n - 4) // 4
        subs = find_cyclic_subgroups_in_range(m, min_r, max_r)
        for g, u_sz, r in subs:
            cands.append((r, n, m, g, u_sz))
    cands.sort()
    print(f"Found {len(cands)} cyclic (n, m, g) candidates with {min_r} <= r <= {max_r}.", flush=True)

    stage1_survivors = []
    for r, n, m, g, u_sz in cands:
        passed, info = full_falsify_candidate(n, m, g, 4)
        if not passed:
            continue
        U = generate_subgroup(m, [g])
        orbits = get_orbits_U(m, U)
        ok_s, msg_s = check_shift_bounds_general(n, m, 4, U, orbits)
        if not ok_s:
            continue
        K = get_shift_count(m, g)
        tag = m_structure_tag(m)
        stage1_survivors.append((r, K, n, m, g, u_sz, tag))
        print(f"  [Stage1 PASS] r={r:2d} K={K:2d} n={n} m={m} ({tag:12s}) g={g:4d} |U|={u_sz}", flush=True)

    print(f"\nStage 1 (deep_falsify + Shift MaxCut) survivors: {len(stage1_survivors)} / {len(cands)}", flush=True)
    print("Running exact composite PAF + 50-digit PSD filter on Stage 1 survivors...", flush=True)

    final_survivors = []
    for r, K, n, m, g, u_sz, tag in stage1_survivors:
        U = generate_subgroup(m, [g])
        orbits = get_orbits_U(m, U)
        sols2 = check_2_compression(n, m, orbits, (2, 0, 0, 0)) if m % 2 == 0 else None
        divs = [d for d in range(2, m) if m % d == 0]
        killed = None
        for d in divs:
            res = analyze_divisor_exact_U(n, m, U, orbits, d, sols2, max_single_tuples=250000, max_paf_vecs=4000)
            if res["status"].startswith("FALSIFIED"):
                killed = f"d={d} ({res['status']})"
                break
        if killed:
            print(f"  [KILLED]   r={r:2d} K={K:2d} n={n} m={m} ({tag:12s}) g={g:4d} -> {killed}", flush=True)
        else:
            print(f"  [SURVIVED] r={r:2d} K={K:2d} n={n} m={m} ({tag:12s}) g={g:4d} |U|={u_sz} (2-comp={len(sols2) if sols2 is not None else 'odd'})", flush=True)
            final_survivors.append((r, K, n, m, g, u_sz, tag))
    print(f"\n--> THEOREM 5 ({min_r}<=r<={max_r}) FINAL UNOBSTRUCTED SURVIVORS: {len(final_survivors)}", flush=True)
    return final_survivors


def scan_theorem3(min_r=37, max_r=64):
    print("\n" + "=" * 100, flush=True)
    print(f"PART 2: THEOREM 3 (b=0, n=4m) CYCLIC {min_r} <= r <= {max_r} ACROSS ALL 85 OPEN ORDERS", flush=True)
    print("=" * 100, flush=True)
    cands = []
    for n in UNKNOWN_85:
        m = n // 4
        subs = find_cyclic_subgroups_in_range(m, min_r, max_r)
        for g, u_sz, r in subs:
            cands.append((r, n, m, g, u_sz))
    cands.sort()
    print(f"Found {len(cands)} cyclic (n, m, g) candidates with {min_r} <= r <= {max_r}.", flush=True)

    survivors = []
    for r, n, m, g, u_sz in cands:
        passed, info = full_falsify_candidate(n, m, g, 0)
        if not passed:
            continue
        U = generate_subgroup(m, [g])
        orbits = get_orbits_U(m, U)
        ok_s, msg_s = check_shift_bounds_general(n, m, 0, U, orbits)
        if not ok_s:
            continue
        K = get_shift_count(m, g)
        tag = m_structure_tag(m)
        rq_cnt = len(info["surviving_quads"])
        rqs = [q["row_sums"] for q in info["surviving_quads"][:3]]
        survivors.append((r, K, n, m, g, u_sz, tag, rq_cnt, rqs))
        print(f"  [SURVIVED] r={r:2d} K={K:2d} n={n} m={m} ({tag:12s}) g={g:4d} |U|={u_sz} row_quads={rq_cnt} {rqs}", flush=True)
    print(f"\n--> THEOREM 3 ({min_r}<=r<={max_r}) SURVIVORS: {len(survivors)} / {len(cands)}", flush=True)
    return survivors


if __name__ == "__main__":
    t0 = time.time()
    scan_theorem5(49, 72)
    scan_theorem3(37, 64)
    print(f"\nTotal elapsed: {time.time() - t0:.2f}s", flush=True)
