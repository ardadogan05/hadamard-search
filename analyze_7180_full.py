#!/usr/bin/env python3
"""
Exact joint (mod 46, mod 78) compression analyzer and independent verifier
for order n = 7180, m = 1794 = 46 * 39, g = 29 (|U| = 66, r = 60, b = 4).

Keeps analyze_7180_joint.py untouched and provides:
  1. Exact lossless canonical enumeration of mod-46 compressed quadruples
     (up to row permutation of rows 1..3, per-row negation of rows 1..3,
     per-row translation by m/2 = 897, and per-row reversal x -> -x).
  2. Independent exact integer verification of:
     - CRT 2 x 3 x 10 orbit decomposition on Z_1794,
     - All 46 shifts of Z_46 and all 78 shifts of Z_78 (11 shift orbits),
     - Shared 60-orbit sign consistency between mod-46 and mod-78 views.
  3. Driver/cross-checker for the single-threaded C++20 engine analyze_7180_full.cpp.
"""

import argparse
import os
import random
import subprocess
import time
from collections import defaultdict
from itertools import product

from analyze_7180_joint import MOD46_WITNESS, compatible_energy, orbits, possible_halves, verify_mod46
from check_composites_v2 import generate_subgroup, get_shift_orbits_U
from deep_falsify import check_2_compression
import decompress_r54 as dr

N, M, G = 7180, 1794, 29
TARGET_E78 = N - 4 * (M // 78)       # 7088
SCALED_E = (TARGET_E78 - 8 * 39) // 44  # 154
MAX_A = SCALED_E // 10               # 15
TARGET_PAF78 = -4 * (M // 78)        # -92
EVEN_SHIFTS = [26, 6, 12, 2, 4]
ODD_SHIFTS = [39, 13, 3, 15, 1, 5]


def canon_a2(a):
    return min(a, (a[0], a[2], a[1]))


def canon_a0(a):
    r = (a[0], a[2], a[1])
    return min(a, r, tuple(-x for x in a), tuple(-x for x in r))


def canon_row2(e, o):
    """Exact order-4 per-row symmetry group for row_sum = +2 (shift by m/2 and reversal)."""
    er, or_ = (e[0], e[2], e[1]), (o[0], o[2], o[1])
    return min((e, o), (o, e), (er, or_), (or_, er))


def canon_row0(e, o):
    """Exact order-8 per-row symmetry group for row_sum = 0 (negation, shift by m/2, reversal)."""
    er, or_ = (e[0], e[2], e[1]), (o[0], o[2], o[1])
    ne, no = tuple(-x for x in e), tuple(-x for x in o)
    ner, nor = tuple(-x for x in er), tuple(-x for x in or_)
    return min(
        (e, o), (o, e), (er, or_), (or_, er),
        (ne, no), (no, ne), (ner, nor), (nor, ner),
    )


def build_mod78_structures():
    orbits23 = orbits(23)
    orbits39 = orbits(39)
    weights = [len(o) for o in orbits39]
    orb39_map = {r: j for j, orb in enumerate(orbits39) for r in orb}

    U = generate_subgroup(M, [G])
    U78 = sorted({u % 78 for u in U})
    sreps, ssizes = get_shift_orbits_U(78, U78)
    assert sorted(EVEN_SHIFTS + ODD_SHIFTS) == sorted(sreps)

    even_mats = {}
    for s in EVEN_SHIFTS:
        mat = [[0] * 10 for _ in range(10)]
        for x in range(0, 78, 2):
            mat[orb39_map[x % 39]][orb39_map[(x + s) % 39]] += 1
        even_mats[s] = [(i, j, mat[i][j]) for i in range(10) for j in range(10) if mat[i][j]]

    odd_mats = {}
    for s in ODD_SHIFTS:
        mat = [[0] * 10 for _ in range(10)]
        for x in range(0, 78, 2):
            mat[orb39_map[x % 39]][orb39_map[(x + s) % 39]] += 1
        for x in range(1, 78, 2):
            mat[orb39_map[(x + s) % 39]][orb39_map[x % 39]] += 1
        odd_mats[s] = [(i, j, mat[i][j]) for i in range(10) for j in range(10) if mat[i][j]]

    return orbits23, orbits39, weights, orb39_map, sreps, ssizes, even_mats, odd_mats


def verify_mod78_brute_force(mod78_rows, orb39_map):
    """Independently verify row sums and ALL 78 periodic autocorrelations on Z_78."""
    seqs = []
    for e_vec, o_vec in mod78_rows:
        seq = [e_vec[orb39_map[x % 39]] if x % 2 == 0 else o_vec[orb39_map[x % 39]] for x in range(78)]
        seqs.append(seq)
    sums = tuple(sum(s) for s in seqs)
    if sums != (2, 0, 0, 0):
        raise RuntimeError(f"Incorrect mod-78 row sums: {sums}")
    for s in range(78):
        paf = sum(sum(a[x] * a[(x + s) % 78] for x in range(78)) for a in seqs)
        expected = TARGET_E78 if s == 0 else TARGET_PAF78
        if paf != expected:
            raise RuntimeError(f"Incorrect mod-78 PAF at shift {s}: {paf} != {expected}")
    return True


def verify_crt_60_orbit_consistency(mod46_rows, mod78_rows, weights):
    """Verify each (mod46_half, mod78_half) comes from an exact 3x10 assignment of +-1 signs."""
    full_mask = (1 << len(weights)) - 1
    mask_weight = [sum(w for i, w in enumerate(weights) if (m >> i) & 1) for m in range(full_mask + 1)]
    masks = defaultdict(list)
    for m, w in enumerate(mask_weight):
        masks[2 * w - 39].append(m)

    for r in range(4):
        for h in range(2):
            s_sum, u_sum, v_sum = mod46_rows[r][h]
            v78 = mod78_rows[r][h]
            # Recover s_i in {-1, 1} and t_i in {-1, 0, 1} uniquely from v78[i] = s_i + 22*t_i
            s_vec = []
            t_vec = []
            for val in v78:
                found = None
                for t_i in (-1, 0, 1):
                    s_i = val - 22 * t_i
                    if s_i in (-1, 1):
                        found = (s_i, t_i)
                        break
                if found is None:
                    raise RuntimeError(f"Invalid mod-78 coordinate {val}")
                s_vec.append(found[0])
                t_vec.append(found[1])
            if sum(w * s for w, s in zip(weights, s_vec)) != s_sum:
                raise RuntimeError("Singleton orbit sum mismatch between mod-46 and mod-78")
            if 2 * sum(w * t for w, t in zip(weights, t_vec)) != u_sum + v_sum:
                raise RuntimeError("u+v orbit sum mismatch between mod-46 and mod-78")
            # Check existence of d_i in {-1, +1} on {i : t_i == 0} with 2*sum(w_i * d_i) == u_sum - v_sum
            zero_weights = [w for w, t in zip(weights, t_vec) if t == 0]
            target_diff = (u_sum - v_sum) // 2
            reachable = {0}
            for w in zero_weights:
                reachable = {x + w for x in reachable} | {x - w for x in reachable}
            if target_diff not in reachable:
                raise RuntimeError("No valid 60-orbit sign assignment realizes both mod-46 and mod-78")
    return True


def check_single_witness_python(rows, weights, even_mats, odd_mats):
    """Independent pure-Python exact checker for one mod-46 witness across all 11 mod-78 shifts."""
    width = len(weights)
    full_mask = (1 << width) - 1
    mask_weight = [sum(w for i, w in enumerate(weights) if (m >> i) & 1) for m in range(full_mask + 1)]
    masks = defaultdict(list)
    for m, w in enumerate(mask_weight):
        masks[2 * w - 39].append(m)

    halves = [possible_halves(profile, masks, mask_weight, full_mask, width, MAX_A)
              for row in rows for profile in row]
    if any(not h for h in halves):
        return "unreachable_crt"
    dp = {(0, 0)}
    for h in halves:
        dp = compatible_energy(dp, h, MAX_A)
        if not dp:
            return "energy(7088)"
    if not any(11 * a + b == SCALED_E for a, b in dp):
        return "energy(7088)"

    allowed = []
    for i, h in enumerate(halves):
        others = {(0, 0)}
        for j, o in enumerate(halves):
            if i != j:
                others = compatible_energy(others, o, MAX_A)
        allowed.append([
            v for (a, b), vs in h.items()
            if any(a + c <= MAX_A and 11 * (a + c) + b + d == SCALED_E for c, d in others)
            for v in vs
        ])

    # Individual even shifts
    for s in EVEN_SHIFTS:
        cur = {(0, 0)}
        for hv in allowed:
            h_vals = {
                (sum(w * x * x for w, x in zip(weights, v)),
                 sum(c * v[i] * v[j] for i, j, c in even_mats[s]))
                for v in hv
            }
            cur = {(e1 + e2, p1 + p2) for e1, p1 in cur for e2, p2 in h_vals if e1 + e2 <= TARGET_E78}
            if not cur:
                break
        if (TARGET_E78, TARGET_PAF78) not in cur:
            return f"even_{s}"

    # Individual odd shifts
    for s in ODD_SHIFTS:
        cur = {(0, 0)}
        for ev, od in zip(allowed[::2], allowed[1::2]):
            rp = {
                (sum(w * (x * x + y * y) for w, x, y in zip(weights, e, o)),
                 sum(c * e[i] * o[j] for i, j, c in odd_mats[s]))
                for e, o in product(ev, od)
            }
            cur = {(e1 + e2, p1 + p2) for e1, p1 in cur for e2, p2 in rp if e1 + e2 <= TARGET_E78}
            if not cur:
                break
        if (TARGET_E78, TARGET_PAF78) not in cur:
            return f"odd_{s}"

    return "SURVIVED_ALL_INDIVIDUAL"


def export_canonical_mod46(out_path):
    """Export exact canonical mod-46 single rows and canonical d=23 quads for analyze_7180_full.cpp."""
    t0 = time.time()
    U, orbs, cosets, shift_reps, paf_mats, quads = dr.get_all_d_quads(N, M, G, 23)
    canon_quads = sorted({(canon_a2(q[0]), *sorted(canon_a0(x) for x in q[1:])) for q in quads})
    uniq_a2 = sorted({q[0] for q in canon_quads})
    uniq_a0 = sorted({q[k] for q in canon_quads for k in (1, 2, 3)})
    a2_idx = {a: i for i, a in enumerate(uniq_a2)}
    a0_idx = {a: i for i, a in enumerate(uniq_a0)}

    orbits39 = orbits(39)
    weights = [len(o) for o in orbits39]
    full_mask = (1 << len(weights)) - 1
    mask_weight = [sum(w for i, w in enumerate(weights) if (m >> i) & 1) for m in range(full_mask + 1)]
    even_R_set = {2 * w - 39 for w in mask_weight}

    sols2 = check_2_compression(N, M, orbs, (2, 0, 0, 0))
    allowed_d2 = sorted({abs(d0) for (d0, _, _, _) in sols2})
    allowed_d0 = sorted({dv for (_, d1, d2, d3) in sols2 for dv in (d1, d2, d3)})
    csizes = [len(c) for c in cosets]
    psd_ok = dr.build_exact_psd_filter(N, 23, U, shift_reps)

    rows2_by_a = defaultdict(set)
    for a in uniq_a2:
        B_lists = [
            sorted(ev - ov for ev in even_R_set for ov in even_R_set
                   if ev + ov == a[i] and csizes[i] * (ev - ov) ** 2 <= N)
            for i in range(3)
        ]
        for df in allowed_d2:
            for b in dr.enumerate_exact(csizes, B_lists, df, N):
                fp = dr.compute_paf_vector(b, csizes, paf_mats)
                if psd_ok(fp):
                    e = tuple((a[i] + b[i]) // 2 for i in range(3))
                    o = tuple((a[i] - b[i]) // 2 for i in range(3))
                    r = canon_row2(e, o)
                    assert canon_a2(tuple(r[0][i] + r[1][i] for i in range(3))) == a
                    rows2_by_a[a].add((r, fp))

    rows0_by_a = defaultdict(set)
    for a in uniq_a0:
        B_lists = [
            sorted(ev - ov for ev in even_R_set for ov in even_R_set
                   if ev + ov == a[i] and csizes[i] * (ev - ov) ** 2 <= N)
            for i in range(3)
        ]
        for df in allowed_d0:
            for b in dr.enumerate_exact(csizes, B_lists, df, N):
                fp = dr.compute_paf_vector(b, csizes, paf_mats)
                if psd_ok(fp):
                    e = tuple((a[i] + b[i]) // 2 for i in range(3))
                    o = tuple((a[i] - b[i]) // 2 for i in range(3))
                    r = canon_row0(e, o)
                    assert canon_a0(tuple(r[0][i] + r[1][i] for i in range(3))) == a
                    rows0_by_a[a].add((r, fp))

    # Double-check MOD46_WITNESS is present!
    w0 = canon_row2(*MOD46_WITNESS[0])
    w123 = sorted(canon_row0(*r) for r in MOD46_WITNESS[1:])
    wa0 = canon_a2(tuple(w0[0][i] + w0[1][i] for i in range(3)))
    wa123 = sorted(canon_a0(tuple(w[0][i] + w[1][i] for i in range(3))) for w in w123)
    wa = (wa0, *wa123)
    assert wa in set(canon_quads), "MOD46_WITNESS d=23 quad missing!"
    assert any(r == w0 for r, _ in rows2_by_a[wa0]), "MOD46_WITNESS row 0 missing!"
    for w, a in zip(w123, wa123):
        assert any(r == w for r, _ in rows0_by_a[a]), "MOD46_WITNESS row 1..3 missing!"

    with open(out_path, "w") as f:
        f.write(f"{len(uniq_a2)} {len(uniq_a0)} {len(canon_quads)}\n")
        for a in uniq_a2:
            lst = sorted(rows2_by_a[a])
            f.write(f"{len(lst)}\n")
            for (e, o), fp in lst:
                f.write(f"{e[0]} {e[1]} {e[2]} {o[0]} {o[1]} {o[2]} {fp[0]} {fp[1]}\n")
        for a in uniq_a0:
            lst = sorted(rows0_by_a[a])
            f.write(f"{len(lst)}\n")
            for (e, o), fp in lst:
                f.write(f"{e[0]} {e[1]} {e[2]} {o[0]} {o[1]} {o[2]} {fp[0]} {fp[1]}\n")
        for a1, a2, a3, a4 in canon_quads:
            f.write(f"{a2_idx[a1]} {a0_idx[a2]} {a0_idx[a3]} {a0_idx[a4]}\n")

    tot_r2 = sum(len(v) for v in rows2_by_a.values())
    tot_r0 = sum(len(v) for v in rows0_by_a.values())
    print(
        f"Exported {out_path} in {time.time() - t0:.2f}s: "
        f"{len(canon_quads)} canon d=23 quads (from {len(quads)} raw), "
        f"{tot_r2} canon rows2, {tot_r0} canon rows0. "
        f"MOD46_WITNESS verified at canon quad index {canon_quads.index(wa)}."
    )


def verify_survivors_file(path):
    orbits23, orbits39, weights, orb39_map, _, _, _, _ = build_mod78_structures()
    if not os.path.exists(path):
        print(f"No survivors file at {path}")
        return 0
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    idx = 0
    verified = 0
    while idx < len(lines):
        assert lines[idx] == "WITNESS"
        idx += 1
        mod46_rows = []
        mod78_rows = []
        for _ in range(4):
            p46 = list(map(int, lines[idx].split()[1:]))
            p78e = tuple(map(int, lines[idx + 1].split()[1:]))
            p78o = tuple(map(int, lines[idx + 2].split()[1:]))
            idx += 3
            mod46_rows.append((tuple(p46[:3]), tuple(p46[3:])))
            mod78_rows.append((p78e, p78o))
        verify_mod46(tuple(mod46_rows), orbits23)
        verify_mod78_brute_force(mod78_rows, orb39_map)
        verify_crt_60_orbit_consistency(mod46_rows, mod78_rows, weights)
        verified += 1
    print(f"Independently verified {verified} joint (mod 46, mod 78) witnesses in {path}.")
    return verified


def self_test():
    orbits23, orbits39, weights, orb39_map, sreps, ssizes, even_mats, odd_mats = build_mod78_structures()
    # 1. Verify CRT 2 x 3 x 10 orbit partition equals full Z_1794 orbits
    partition = {
        frozenset(x for x in range(M) if x % 2 == parity and x % 23 in row and x % 39 in col)
        for parity, row, col in product(range(2), orbits23, orbits39)
    }
    assert partition == {frozenset(o) for o in orbits(M)}
    assert len(partition) == 60

    # 2. Verify all 11 mod-78 shift bilinear forms against brute-force Z_78 PAF
    rng = random.Random(2026)
    for _ in range(5):
        e = [rng.randint(-23, 23) for _ in range(10)]
        o = [rng.randint(-23, 23) for _ in range(10)]
        seq = [e[orb39_map[x % 39]] if x % 2 == 0 else o[orb39_map[x % 39]] for x in range(78)]
        for s in EVEN_SHIFTS:
            direct = sum(seq[x] * seq[(x + s) % 78] for x in range(78))
            via_mat = (
                sum(c * e[i] * e[j] for i, j, c in even_mats[s])
                + sum(c * o[i] * o[j] for i, j, c in even_mats[s])
            )
            assert direct == via_mat
        for s in ODD_SHIFTS:
            direct = sum(seq[x] * seq[(x + s) % 78] for x in range(78))
            via_mat = sum(c * e[i] * o[j] for i, j, c in odd_mats[s])
            assert direct == via_mat

    # 3. Verify MOD46_WITNESS passes mod-46 and dies at even_26 in pure Python
    verify_mod46(MOD46_WITNESS, orbits23)
    res = check_single_witness_python(MOD46_WITNESS, weights, even_mats, odd_mats)
    assert res == "even_26", f"Expected MOD46_WITNESS to die at even_26, got {res}"
    print("[SELF-TEST PASSED] CRT partition (60 orbits), all 11 mod-78 shift matrices, and MOD46_WITNESS check verified.")


def main():
    parser = argparse.ArgumentParser(description="Joint (mod 46, mod 78) exact analyzer for 7180 g=29")
    parser.add_argument("--self-test", action="store_true", help="run independent algebraic and brute-force self-tests")
    parser.add_argument("--export", help="export canonical mod-46 dataset to PATH")
    parser.add_argument("--verify-survivors", help="independently verify exported joint witnesses from PATH")
    parser.add_argument("--run", action="store_true", help="compile C++ engine, export data, and run energy-only + stratified scan")
    parser.add_argument("--per-quad-cap", type=int, default=20, help="per-quad cap for --run stratified scan")
    args = parser.parse_args()

    if args.self_test or not (args.export or args.verify_survivors or args.run):
        self_test()
    if args.export:
        export_canonical_mod46(args.export)
    if args.verify_survivors:
        verify_survivors_file(args.verify_survivors)
    if args.run:
        data_path = "/tmp/data_7180.txt"
        bin_path = "/tmp/analyze_7180_full"
        surv_path = "/tmp/survivors_7180.txt"
        export_canonical_mod46(data_path)
        subprocess.run(
            ["g++", "-std=c++20", "-O3", "-march=native", "analyze_7180_full.cpp", "-o", bin_path],
            check=True,
        )
        subprocess.run([bin_path, "--data", data_path, "--energy-only"], check=True)
        subprocess.run(
            [bin_path, "--data", data_path, "--per-quad-cap", str(args.per_quad_cap), "--out", surv_path],
            check=True,
        )
        verify_survivors_file(surv_path)


if __name__ == "__main__":
    main()
