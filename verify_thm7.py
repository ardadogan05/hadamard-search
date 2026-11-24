#!/usr/bin/env python3
"""
Exact-Integer Verifier & Builder for Theorem 7 (b = 12, k = 4, G = Z_2^2 x S)
from Egan-Kotsireas-London-O Cathain (arXiv:2610.27175v2).

Given four {+-1}-valued functions X, Y, Z, W on G = Z_2^2 x S (|S| = s, v = 4s, n = 16s + 12)
whose coset sums on G/S = Z_2^2 satisfy:
    a_f(00) - a_f(c) = d_f   for all c in {01, 10, 11}
with {|d_X|, |d_Y|, |d_Z|, |d_W|} = {0, 2, 2, 2}, and whose periodic autocorrelations satisfy:
    sum_f P_f(d) = -12   for d in S \\ {0}
    sum_f P_f(d) = +4    for d in G \\ S,

constructs the full n x n Hadamard matrix:
    H = [[K,        E @ Pi^T],
         [Pi @ F,   M       ]]
and verifies H_{ij} in {+-1} and H @ H^T = n * I_n over exact integers.
"""

import sys


# Universal exact border matrices (K: 12x12, E: 12x16, F: 16x12) for
# normalized coset differences (d_X, d_Y, d_Z, d_W) = (0, 2, 2, 2) on Z_2^2.
_BMAP = [
    (1, -1,  1, -1),
    (1,  1, -1, -1),
    (1, -1, -1,  1),
]

_F_ORBS = [
    ((0, 0, 1, 2), (1, -1, -1, -1)),
    ((0, 0, 1, 2), (1,  1,  1, -1)),
    ((0, 1, 2, 2), (1, -1,  1,  1)),
    ((0, 1, 2, 2), (1,  1, -1,  1)),
]

_E_ORBS = [
    ((0, 2, 1, 0), (1,  1, -1, -1)),
    ((0, 2, 1, 0), (1,  1,  1,  1)),
    ((0, 2, 2, 1), (1, -1, -1,  1)),
    ((0, 2, 2, 1), (1, -1,  1, -1)),
]


def _expand_vec(colors, signs):
    out = []
    for c, sg in zip(colors, signs):
        out.extend(sg * x for x in _BMAP[c])
    return out


def build_border_KEF():
    """Return exact integer matrices (K, E, F) of sizes (12x12, 12x16, 16x12)
    for normalized coset differences (d_X, d_Y, d_Z, d_W) = (0, 2, 2, 2)."""
    F_cols = []
    for colors, signs in _F_ORBS:
        for shift in range(3):
            c_sh = tuple((c + shift) % 3 for c in colors)
            F_cols.append(_expand_vec(c_sh, signs))
    F = [[F_cols[col][row] for col in range(12)] for row in range(16)]

    E = []
    for colors, signs in _E_ORBS:
        for shift in range(3):
            c_sh = tuple((c + shift) % 3 for c in colors)
            E.append(_expand_vec(c_sh, signs))

    # Mbar_eff on R^4 kron 1_4^perp for (d_X, d_Y, d_Z, d_W) = (0, 2, 2, 2):
    # Note Theorem 3 GS array is:
    #   [ A   BR   CR   DR ]
    #   [-BR  A   D^TR -C^TR]
    #   [-CR -D^TR A   B^TR]
    #   [-DR  C^TR -B^TR A ]
    # For G/S = Z_2^2, Rbar = I_4 and each Abar, Bbar, Cbar, Dbar is symmetric (d_f I_4 + b_f J_4).
    # So on 1_4^perp, Mbar = 2 * Q4_GS kron I_4 where:
    Q4_GS = [
        [ 0,  1,  1,  1],
        [-1,  0,  1, -1],
        [-1, -1,  0,  1],
        [-1,  1, -1,  0],
    ]
    # Wait: check sign convention of Q4_GS vs Q4 used when finding E, F!
    # In Q4_GS, rows 1..3 have +D^T, -C^T (opposite sign on off-diag 2x2 from Q4).
    # Notice negating the sign of (Y, Z, W) or transposing Q4_GS:
    # Let's compute K = -(1/16) E @ Mbar^T @ F directly and verify K in {+-1}^{12x12}!
    # Actually: Q4_GS^T = -Q4_GS, and swapping B, C, D or negating D matches Q4!
    # Specifically, if (d_X, d_Y, d_Z, d_W) = (0, 2, 2, 2) with GS array having
    #   [ A   BR   CR   DR ]
    #   [-BR  A   -D^TR C^TR]  (note: negating column/block signs or using (0, 2, -2, -2)!)
    # Let's compute directly for the exact GS array in Theorem 3:
    #   Row 0: ( A,  BR,  CR,  DR) -> ( 0,  1,  1,  1)
    #   Row 1: (-BR,  A, D^TR,-C^TR) -> (-1,  0,  1, -1)
    #   Row 2: (-CR,-D^TR, A, B^TR) -> (-1, -1,  0,  1)
    #   Row 3: (-DR, C^TR,-B^TR, A) -> (-1,  1, -1,  0)
    # Note: Q4_GS is obtained from Q4 by negating (d_Y, d_Z, d_W) -> (d_Y, -d_Z, d_W)?
    # Check: if d_X=0, d_Y=2, d_Z=2, d_W=-2:
    #   Row 0: ( 0,  2,  2, -2)
    #   Row 1: (-2,  0, -2, -2)
    # Wait: Q4 had Row 0 = (0,1,1,1), Row 1 = (-1,0,-1,1), Row 2 = (-1,1,0,-1), Row 3 = (-1,-1,1,0),
    # which is literally Q4_GS with blocks (2, 3) swapped!
    # So we simply build M with the exact Theorem 3 GS sign convention (matching Q4_GS),
    # or equivalently permute blocks (2, 3) in E and F!
    perm = [0, 1, 3, 2]
    E_p = []
    for row in E:
        r_new = []
        for blk in perm:
            r_new.extend(row[4 * blk : 4 * blk + 4])
        E_p.append(r_new)
    F_p = []
    for blk in perm:
        F_p.extend(F[4 * blk : 4 * blk + 4])

    # Compute K = -(1/8) E_p @ (Q4_GS^T kron I_4) @ F_p
    # First M_F = (Q4_GS^T kron I_4) @ F_p (16 x 12)
    M_F = [[0] * 12 for _ in range(16)]
    for i in range(4):
        for j in range(4):
            qij = Q4_GS[j][i]  # Q4_GS^T[i][j]
            if qij == 0:
                continue
            for r in range(4):
                for c in range(12):
                    M_F[4 * i + r][c] += qij * F_p[4 * j + r][c]

    K = [[0] * 12 for _ in range(12)]
    for i in range(12):
        for j in range(12):
            val = -sum(E_p[i][t] * M_F[t][j] for t in range(16))
            assert val % 8 == 0, f"val={val} not divisible by 8"
            K[i][j] = val // 8
            assert K[i][j] in (-1, 1), f"K[{i}][{j}]={K[i][j]} not in +-1"

    return K, E_p, F_p


def build_and_verify_thm7(s, add_S, neg_S, seqs_4):
    """Build full n x n (n = 16s + 12) Hadamard matrix from 4 sequences on G = Z_2^2 x S
    and verify H_{ij} in {+-1} and H @ H^T = n * I_n over exact integers.

    Args:
      s: |S|
      add_S: callable (a, b) -> a + b in S (indexed 0..s-1)
      neg_S: callable (a) -> -a in S (indexed 0..s-1)
      seqs_4: list of 4 dicts/lists of length 4*s, where index c*s + x corresponds to (c, x) in Z_2^2 x S
              (c in {0,1,2,3} with XOR addition on Z_2^2, x in 0..s-1).
    """
    v = 4 * s
    n = 4 * v + 12

    # 1. Check coset sums and normalize (X, Y, Z, W) so d_X = 0, d_Y = d_Z = d_W = +2
    norm_seqs = [None, None, None, None]
    d2_slot = 1
    for seq in seqs_4:
        assert len(seq) == v
        assert all(val in (-1, 1) for val in seq)
        csums = [sum(seq[c * s : (c + 1) * s]) for c in range(4)]
        assert csums[1] == csums[2] == csums[3], f"Non-sigma3 coset sums: {csums}"
        df = csums[0] - csums[1]
        if df == 0:
            assert norm_seqs[0] is None, "Multiple sequences with d_f == 0"
            norm_seqs[0] = list(seq)
        elif df == 2:
            norm_seqs[d2_slot] = list(seq)
            d2_slot += 1
        elif df == -2:
            norm_seqs[d2_slot] = [-val for val in seq]
            d2_slot += 1
        else:
            raise AssertionError(f"Invalid coset difference d_f = {df}")
    assert d2_slot == 4 and norm_seqs[0] is not None

    # Precompute group subtraction (y - x) and (-y - x) on G = Z_2^2 x S
    def grp_sub(idx_y, idx_x):
        cy, y = divmod(idx_y, s)
        cx, x = divmod(idx_x, s)
        return (cy ^ cx) * s + add_S(y, neg_S(x))

    def grp_neg(idx_x):
        cx, x = divmod(idx_x, s)
        return cx * s + neg_S(x)

    neg_idx = [grp_neg(i) for i in range(v)]

    # Build group-developed matrices A, B, C, D and their R-products:
    # [f]_{i, j} = f(j - i)
    # ([f] R)_{i, j} = f((-j) - i)
    # ([f]^T R)_{i, j} = f(i - (-j)) = f(i + j)
    def make_mat(seq, mode):
        M = [[0] * v for _ in range(v)]
        if mode == "A":
            for i in range(v):
                for j in range(v):
                    M[i][j] = seq[grp_sub(j, i)]
        elif mode == "BR":
            for i in range(v):
                for j in range(v):
                    M[i][j] = seq[grp_sub(neg_idx[j], i)]
        elif mode == "BTR":
            for i in range(v):
                for j in range(v):
                    M[i][j] = seq[grp_sub(i, neg_idx[j])]
        return M

    A = make_mat(norm_seqs[0], "A")
    BR = make_mat(norm_seqs[1], "BR")
    CR = make_mat(norm_seqs[2], "BR")
    DR = make_mat(norm_seqs[3], "BR")
    BTR = make_mat(norm_seqs[1], "BTR")
    CTR = make_mat(norm_seqs[2], "BTR")
    DTR = make_mat(norm_seqs[3], "BTR")

    K, E, F = build_border_KEF()

    # Assemble full n x n matrix H = [[K, E @ Pi^T], [Pi @ F, M]]
    H = [[0] * n for _ in range(n)]
    # Top-left K (12 x 12)
    for i in range(12):
        for j in range(12):
            H[i][j] = K[i][j]
    # Top-right E @ Pi^T (12 x 4v): column (blk*v + c*s + x) gets E[i][blk*4 + c]
    for i in range(12):
        for blk in range(4):
            for c in range(4):
                val = E[i][4 * blk + c]
                col0 = 12 + blk * v + c * s
                for x in range(s):
                    H[i][col0 + x] = val
    # Bottom-left Pi @ F (4v x 12): row (blk*v + c*s + x) gets F[blk*4 + c][j]
    for blk in range(4):
        for c in range(4):
            row0 = 12 + blk * v + c * s
            for x in range(s):
                for j in range(12):
                    H[row0 + x][j] = F[4 * blk + c][j]
    # Bottom-right M (4v x 4v): exact Theorem 3 GS array
    gs_blocks = [
        [(1, A),  ( 1, BR),  ( 1, CR),  ( 1, DR)],
        [(-1, BR),( 1, A),   ( 1, DTR), (-1, CTR)],
        [(-1, CR),(-1, DTR), ( 1, A),   ( 1, BTR)],
        [(-1, DR),( 1, CTR), (-1, BTR), ( 1, A)],
    ]
    for bi in range(4):
        for bj in range(4):
            sgn, blk_mat = gs_blocks[bi][bj]
            r0 = 12 + bi * v
            c0 = 12 + bj * v
            for i in range(v):
                row_src = blk_mat[i]
                row_dst = H[r0 + i]
                if sgn == 1:
                    for j in range(v):
                        row_dst[c0 + j] = row_src[j]
                else:
                    for j in range(v):
                        row_dst[c0 + j] = -row_src[j]

    # Exact integer verification of H_{ij} in {+-1} and H @ H^T = n * I_n
    for i in range(n):
        for j in range(n):
            assert H[i][j] in (-1, 1)
    for i in range(n):
        ri = H[i]
        for j in range(i, n):
            rj = H[j]
            dot = sum(a * b for a, b in zip(ri, rj))
            expected = n if i == j else 0
            if dot != expected:
                raise AssertionError(f"Orthogonality failed at ({i}, {j}): got {dot}, expected {expected}")
    return H


def self_test():
    # Self-test on s = 1 (v = 4, n = 28):
    # A = (1, 1, 1, 1) -> d = 0; B = C = D = (-1, 1, 1, 1) -> d = -2 (normalized to (1, -1, -1, -1), d = +2)
    seqs = [
        [1, 1, 1, 1],
        [-1, 1, 1, 1],
        [-1, 1, 1, 1],
        [-1, 1, 1, 1],
    ]
    H = build_and_verify_thm7(1, lambda a, b: 0, lambda a: 0, seqs)
    print(f"[PASS] verify_thm7.py self-test s=1 (n={len(H)}): H in {{+-1}} and H @ H^T = {len(H)} * I_{len(H)} VERIFIED EXACTLY!")


if __name__ == "__main__":
    self_test()
