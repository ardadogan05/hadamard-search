#!/usr/bin/env python3
"""
Independent exact integer verifier for Hadamard matrices constructed from
Legendre quadruples / Goethals-Seidel sequences (Theorem 5 of arXiv:2610.27175v1).

Given a witness JSON file produced by a Goethals-Seidel solver:
  1. Reconstructs the orbits of the multiplier group modulo m and expands the
     4 sequences A, B, C, D of length m.
  2. Independently verifies O(m^2) periodic autocorrelations:
         sum_{s=0}^3 PAF_s(d) == -b   for all 1 <= d < m.
  3. Explicitly constructs the full n x n Hadamard matrix H (n = 4m + b)
     using the bordered Goethals-Seidel array (b = 4) or unbordered GS (b = 0).
  4. Verifies with exact integer arithmetic that:
       - every entry H[i][j] is in {-1, +1}
       - H @ H.T == n * I_n  (exact integer matrix multiplication).

Usage:
  python3 verify.py <witness.json>
  python3 verify.py --self-test
"""

import json
import math
import sys
import time


def build_orbits(m: int, g: int | list[int]):
    generators = [g] if isinstance(g, int) else g
    elem_to_orb = [-1] * m
    orbits = []
    for x in range(m):
        if elem_to_orb[x] == -1:
            oid = len(orbits)
            orb = [x]
            elem_to_orb[x] = oid
            for residue in orb:
                for generator in generators:
                    y = residue * generator % m
                    if elem_to_orb[y] == -1:
                        elem_to_orb[y] = oid
                        orb.append(y)
            orbits.append(orb)
    return orbits, elem_to_orb


def make_circulant(seq):
    """Circulant m x m matrix X with X[i][j] = seq[(j - i) % m]."""
    m = len(seq)
    return [[seq[(j - i) % m] for j in range(m)] for i in range(m)]


def mat_mul_small(A, B):
    """Exact integer matrix multiplication for small matrices."""
    rows = len(A)
    cols = len(B[0])
    inner = len(B)
    return [
        [sum(A[i][k] * B[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]


def mat_transpose(A):
    return [list(col) for col in zip(*A)]


def apply_R_right(X):
    """Right-multiply m x m matrix X by back-diagonal R: (X*R)[i][j] = X[i][m-1-j]."""
    return [row[::-1] for row in X]


def negate_mat(X):
    return [[-v for v in row] for row in X]


def build_goethals_seidel(seqs):
    """
    Build the 4m x 4m Goethals-Seidel array G from 4 sequences of length m:
        G = [  A     B*R     C*R     D*R   ]
            [ -B*R   A      -D^T*R   C^T*R ]
            [ -C*R   D^T*R   A      -B^T*R ]
            [ -D*R  -C^T*R   B^T*R   A     ]
    """
    m = len(seqs[0])
    A, B, C, D = [make_circulant(s) for s in seqs]
    BR = apply_R_right(B)
    CR = apply_R_right(C)
    DR = apply_R_right(D)
    BtR = apply_R_right(mat_transpose(B))
    CtR = apply_R_right(mat_transpose(C))
    DtR = apply_R_right(mat_transpose(D))

    blocks = [
        [A, BR, CR, DR],
        [negate_mat(BR), A, negate_mat(DtR), CtR],
        [negate_mat(CR), DtR, A, negate_mat(BtR)],
        [negate_mat(DR), negate_mat(CtR), BtR, A],
    ]

    G = []
    for br in range(4):
        for i in range(m):
            row = []
            for bc in range(4):
                row.extend(blocks[br][bc][i])
            G.append(row)
    return G


def build_hadamard_theorem5(seqs, b: int):
    """
    Construct the full n x n Hadamard matrix H (n = 4m + b) from 4 circulant
    generating sequences of length m.

    For b = 4 (Theorem 5 bordered Goethals-Seidel construction):
        H = [ M_4          E (x) 1_m^T ]
            [ F (x) 1_m    G           ]
    where:
      - G is the 4m x 4m Goethals-Seidel array with block row-sum matrix S_G:
            S_G = [  sA   sB   sC   sD ]
                  [ -sB   sA  -sD   sC ]
                  [ -sC   sD   sA  -sB ]
                  [ -sD  -sC   sB   sA ]
        satisfying S_G * S_G^T = (sA^2 + sB^2 + sC^2 + sD^2) I_4 = 4 I_4.
      - M_4, E, F are 4 x 4 {-1, +1} Hadamard matrices satisfying:
            M_4 * F^T + E * S_G^T = 0.
        * Even m ({|sA|,|sB|,|sC|,|sD|} = {2,0,0,0}):
          S_G = 2 * P where P is a 4 x 4 signed permutation matrix.
          Take M_4 = H_4 (Sylvester 4x4 Hadamard), F = C_4 = J_4 - 2*I_4,
          and E = -(1/2) * M_4 * F^T * P.
          Since (1/2) * H_4 * C_4 has all entries in {-1, +1}, E is a valid
          4 x 4 {-1, +1} Hadamard matrix and M_4 * F^T + E * (2 P^T) = 0.
        * Odd m (|sA| = |sB| = |sC| = |sD| = 1):
          S_G itself is a 4 x 4 {-1, +1} Hadamard matrix.
          Take E = S_G, F = H_4, M_4 = -H_4.
          Then M_4 * F^T + E * S_G^T = -4*I_4 + 4*I_4 = 0.
    """
    m = len(seqs[0])
    G = build_goethals_seidel(seqs)
    if b == 0:
        return G
    if b != 4:
        raise ValueError(f"Unsupported border size b={b} (expected 0 or 4)")

    sA, sB, sC, sD = [sum(s) for s in seqs]
    sum_sq = sA * sA + sB * sB + sC * sC + sD * sD
    if sum_sq != 4:
        raise ValueError(
            f"Row sums {(sA, sB, sC, sD)} do not satisfy sum(s_i^2) == 4 (got {sum_sq})"
        )

    S_G = [
        [sA, sB, sC, sD],
        [-sB, sA, -sD, sC],
        [-sC, sD, sA, -sB],
        [-sD, -sC, sB, sA],
    ]

    H4 = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ]
    C4 = [
        [-1, 1, 1, 1],
        [1, -1, 1, 1],
        [1, 1, -1, 1],
        [1, 1, 1, -1],
    ]

    if m % 2 == 0:
        # S_G = 2 * P with P a signed permutation matrix
        P = [[v // 2 for v in row] for row in S_G]
        M4 = H4
        F = C4
        # E = -(1/2) * M4 * F^T * P
        MFt = mat_mul_small(M4, mat_transpose(F))
        half_neg_MFt = [[-v // 2 for v in row] for row in MFt]
        E = mat_mul_small(half_neg_MFt, P)
    else:
        # S_G is a {-1, +1} Hadamard matrix
        E = S_G
        F = H4
        M4 = negate_mat(H4)

    # Sanity check border equations over exact integers
    chk = mat_mul_small(M4, mat_transpose(F))
    esgt = mat_mul_small(E, mat_transpose(S_G))
    for i in range(4):
        for j in range(4):
            if chk[i][j] + esgt[i][j] != 0:
                raise AssertionError("Border orthogonality equation M4*F^T + E*S_G^T != 0")

    # Assemble full (4m + 4) x (4m + 4) matrix H
    H = []
    # Top 4 rows: [M4, E (x) 1_m^T]
    for r in range(4):
        row = list(M4[r])
        for c in range(4):
            row.extend([E[r][c]] * m)
        H.append(row)

    # Bottom 4m rows: [F (x) 1_m, G]
    for br in range(4):
        f_row = F[br]
        for i in range(m):
            H.append(list(f_row) + G[br * m + i])

    return H


def verify_sequences_paf(seqs, b: int):
    """Independent exact O(m^2) periodic autocorrelation check."""
    m = len(seqs[0])
    for s_idx, seq in enumerate(seqs):
        if len(seq) != m:
            return False, f"Sequence {s_idx} has length {len(seq)}, expected {m}"
        for x in seq:
            if x not in (-1, 1):
                return False, f"Sequence {s_idx} has non-{{-1,+1}} entry {x}"

    target = -b
    for d in range(1, m):
        ac = 0
        for seq in seqs:
            p = 0
            for x in range(m):
                p += seq[x] * seq[(x + d) % m]
            ac += p
        if ac != target:
            return False, f"PAF sum mismatch at shift d={d}: got {ac}, expected {target}"
    return True, "PAF check passed"


def verify_hadamard_exact(H, expected_n: int, also_naive_threshold: int = 128):
    """
    Exact integer verification of H:
      - shape is expected_n x expected_n
      - all entries in {-1, +1}
      - H @ H.T == expected_n * I_n (exact integer inner products)
    Uses exact bit-packed integer Hamming distance:
      for {-1,+1} rows u, v with bitmasks bu, bv (bit j = 1 iff entry == +1),
      sum_k u[k]*v[k] == n - 2 * (bu ^ bv).bit_count()  (exact identity).
    For n <= also_naive_threshold, also cross-checks with naive O(n^3) integer matmul.
    """
    n = len(H)
    if n != expected_n:
        return False, f"Row count {n} != expected {expected_n}"

    masks = [0] * n
    for i, row in enumerate(H):
        if len(row) != n:
            return False, f"Row {i} has length {len(row)} != {n}"
        mask = 0
        for j, v in enumerate(row):
            if v == 1:
                mask |= (1 << j)
            elif v != -1:
                return False, f"Invalid entry H[{i}][{j}] = {v} (not in {{-1, +1}})"
        masks[i] = mask

    # Exact integer H @ H.T == n * I_n check
    # Diagonal: every row has n entries in {-1,+1}, so row_i . row_i == n.
    # Off-diagonal: row_i . row_j == n - 2 * (masks[i] ^ masks[j]).bit_count().
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            dot = n - 2 * (mi ^ masks[j]).bit_count()
            if dot != 0:
                return False, f"(H @ H.T)[{i}][{j}] = {dot} != 0"

    # Cross-check with explicit O(n^3) integer multiplication for small matrices
    if n <= also_naive_threshold:
        for i in range(n):
            ri = H[i]
            for j in range(n):
                rj = H[j]
                dot = sum(ri[k] * rj[k] for k in range(n))
                exp = n if i == j else 0
                if dot != exp:
                    return False, f"Naive matmul mismatch at ({i},{j}): got {dot}, expected {exp}"

    return True, f"EXACT PASS: {n}x{n} Hadamard matrix verified (all entries in {{-1,+1}}, H @ H.T == {n}*I)"


def expand_witness(data):
    if not isinstance(data, dict):
        raise ValueError("Witness must be a JSON object")
    n, m, b = (data.get(key) for key in ("n", "m", "b"))
    if any(type(value) is not int for value in (n, m, b)):
        raise ValueError("n, m and b must be integers")
    if m <= 0 or b not in (0, 4):
        raise ValueError("Expected m > 0 and b equal to 0 or 4")

    if n != 4 * m + b:
        raise ValueError(f"n={n} != 4*m+b={4*m+b}")

    generators = data.get("gens", [data.get("g")])
    if not isinstance(generators, list) or any(
        type(g) is not int or math.gcd(g, m) != 1 for g in generators
    ):
        raise ValueError("Each multiplier generator must be an integer coprime to m")
    orbits, elem_to_orb = build_orbits(m, generators)
    r = len(orbits)
    signs = data.get("orbit_signs")
    if not isinstance(signs, list) or len(signs) != 4 or any(
        not isinstance(row, list) or len(row) != r for row in signs
    ):
        raise ValueError(f"orbit_signs shape mismatch (expected 4 x {r})")
    if any(type(sign) is not int or sign not in (-1, 1) for row in signs for sign in row):
        raise ValueError("orbit_signs entries must be integers equal to -1 or 1")
    return n, m, b, [[row[i] for i in elem_to_orb] for row in signs]


def verify_witness_file(path: str):
    t0 = time.perf_counter()
    try:
        with open(path, "r") as f:
            n, m, b, seqs = expand_witness(json.load(f))
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return False

    ok_paf, msg_paf = verify_sequences_paf(seqs, b)
    print(f"[1/3] Sequence PAF check (m={m}, b={b}): {msg_paf}")
    if not ok_paf:
        return False

    H = build_hadamard_theorem5(seqs, b)
    t1 = time.perf_counter()
    print(f"[2/3] Constructed full {n}x{n} matrix H in {t1 - t0:.2f}s")

    ok_H, msg_H = verify_hadamard_exact(H, n)
    t2 = time.perf_counter()
    print(f"[3/3] Matrix check ({t2 - t1:.2f}s): {msg_H}")
    return ok_H


def run_self_test():
    print("=== Running verify.py self-tests ===")

    # Test 1: m = 2, n = 12 (b = 4, even m)
    seqs_12 = [
        [1, 1],
        [1, -1],
        [1, -1],
        [1, -1],
    ]
    ok, msg = verify_sequences_paf(seqs_12, 4)
    assert ok, msg
    H12 = build_hadamard_theorem5(seqs_12, 4)
    ok, msg = verify_hadamard_exact(H12, 12)
    assert ok, msg
    print(f"  [PASS] m=2 -> n=12: {msg}")

    # Test 2: m = 4, n = 20 (b = 4, even m)
    seqs_20 = [
        [1, 1, 1, -1],
        [1, 1, -1, -1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
    ]
    ok, msg = verify_sequences_paf(seqs_20, 4)
    assert ok, msg
    H20 = build_hadamard_theorem5(seqs_20, 4)
    ok, msg = verify_hadamard_exact(H20, 20)
    assert ok, msg
    print(f"  [PASS] m=4 -> n=20: {msg}")

    # Test 3: m = 4, n = 20 with permuted & negated row sums s = (0, 0, -2, 0)
    seqs_20_perm = [
        [1, 1, -1, -1],
        [1, 1, -1, -1],
        [-1, -1, -1, 1],
        [1, -1, 1, -1],
    ]
    ok, msg = verify_sequences_paf(seqs_20_perm, 4)
    assert ok, msg
    H20p = build_hadamard_theorem5(seqs_20_perm, 4)
    ok, msg = verify_hadamard_exact(H20p, 20)
    assert ok, msg
    print(f"  [PASS] m=4 -> n=20 (s=(0,0,-2,0)): {msg}")

    # Test 4: m = 3, n = 16 (b = 4, odd m, s = (1, 1, 1, 1))
    seqs_16 = [
        [1, 1, -1],
        [1, 1, -1],
        [1, 1, -1],
        [1, 1, -1],
    ]
    ok, msg = verify_sequences_paf(seqs_16, 4)
    assert ok, msg
    H16 = build_hadamard_theorem5(seqs_16, 4)
    ok, msg = verify_hadamard_exact(H16, 16)
    assert ok, msg
    print(f"  [PASS] m=3 -> n=16 (odd m): {msg}")

    # Test 5: Negative test - flipping a single entry in H20 must fail!
    H20_bad = [list(row) for row in H20]
    H20_bad[7][13] *= -1
    ok, msg = verify_hadamard_exact(H20_bad, 20)
    assert not ok, "Corrupted matrix unexpectedly passed!"
    print(f"  [PASS] Negative test (flipped 1 entry in H20): correctly rejected ({msg})")

    # Test 6: An unbordered solver_v3 witness uses both listed generators.
    witness = {
        "n": 12, "m": 3, "b": 0, "g": 1, "gens": [1, 2],
        "orbit_signs": [[1, 1], [-1, 1], [-1, 1], [-1, 1]],
    }
    n, _, b, seqs = expand_witness(witness)
    ok, msg = verify_sequences_paf(seqs, b)
    assert ok, msg
    ok, msg = verify_hadamard_exact(build_hadamard_theorem5(seqs, b), n)
    assert ok, msg
    print(f"  [PASS] Multi-generator witness: {msg}")

    witness["orbit_signs"][0][0] = 1.5
    try:
        expand_witness(witness)
    except ValueError:
        print("  [PASS] Fractional orbit sign rejected")
    else:
        raise AssertionError("Fractional orbit sign unexpectedly passed")

    print("=== All self-tests PASSED ===")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 verify.py <witness.json> | --self-test")
        sys.exit(1)
    if sys.argv[1] == "--self-test":
        run_self_test()
        sys.exit(0)
    ok = verify_witness_file(sys.argv[1])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
