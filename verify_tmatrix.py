#!/usr/bin/env python3
"""
Exact-Integer Verifier for Circulant T-Matrices & Cooper-Wallis Hadamard Construction H(4*t*w).

1. Verifies T-matrices (T1, T2, T3, T4) of length t:
   - Each position x in Z_t has exactly one i in {0,1,2,3} with T_i[x] in {-1, +1} and the others 0.
   - sum_{i=0}^3 PAF(T_i, d) == 0 for all 1 <= d < t over exact integers.
2. Implements Cooper-Wallis construction H(4*t*w) using verified symmetric circulant
   Williamson matrices WT(w) for w in {1, 3, 5, 7, 9}.
3. Self-tests on t=3, w=5 -> H(60) by constructing the full 60x60 matrix and checking
   H_{ij} in {+-1} and H @ H^T = 60 * I_60 over exact integers.
"""

import json
import sys


# Verified symmetric circulant Williamson matrices WT(w)
WILLIAMSON_WT = {
    1: [
        [1],
        [1],
        [1],
        [1],
    ],
    3: [
        [1, 1, 1],
        [1, -1, -1],
        [1, -1, -1],
        [1, -1, -1],
    ],
    5: [
        [-1, 1, 1, 1, 1],
        [-1, 1, 1, 1, 1],
        [1, 1, -1, -1, 1],
        [1, -1, 1, 1, -1],
    ],
    7: [
        [-1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, -1, 1, -1],
        [1, 1, -1, -1, -1, -1, 1],
        [1, -1, -1, 1, 1, -1, -1],
    ],
    9: [
        [-1, 1, -1, -1, -1, -1, -1, -1, 1],
        [1, 1, -1, -1, -1, -1, -1, -1, 1],
        [1, -1, 1, 1, -1, -1, 1, 1, -1],
        [1, -1, 1, 1, -1, -1, 1, 1, -1],
    ],
}


def verify_williamson(w):
    """Exact integer verification of symmetric circulant WT(w)."""
    W = WILLIAMSON_WT[w]
    assert len(W) == 4
    for seq in W:
        assert len(seq) == w
        assert all(x in (-1, 1) for x in seq)
        for i in range(1, w):
            assert seq[i] == seq[w - i], f"WT({w}) not symmetric at i={i}"
    for d in range(w):
        paf = sum(seq[i] * seq[(i + d) % w] for seq in W for i in range(w))
        expected = 4 * w if d == 0 else 0
        assert paf == expected, f"WT({w}) failed at shift d={d}: got {paf}, expected {expected}"
    return True


def expand_tmatrix_from_orbits(t, g, orbit_mat, orbit_sign):
    """Expand U-invariant T-matrix witness into four {0, +-1} sequences of length t."""
    seen = [False] * t
    orbits = []
    for x in range(t):
        if not seen[x]:
            orb = []
            cur = x
            while not seen[cur]:
                seen[cur] = True
                orb.append(cur)
                cur = (cur * g) % t
            orbits.append(orb)
    assert len(orbits) == len(orbit_mat) == len(orbit_sign)
    T = [[0] * t for _ in range(4)]
    for oi, orb in enumerate(orbits):
        m_idx = orbit_mat[oi]
        sgn = orbit_sign[oi]
        assert m_idx in (0, 1, 2, 3) and sgn in (-1, 1)
        for x in orb:
            T[m_idx][x] = sgn
    return T


def verify_tmatrix_seqs(T):
    """Exact integer verification that T = [T0, T1, T2, T3] is a valid circulant T-matrix."""
    assert len(T) == 4
    t = len(T[0])
    for i in range(4):
        assert len(T[i]) == t
    # Check disjoint supports and entries in {0, +-1}
    for x in range(t):
        nz = [T[i][x] for i in range(4) if T[i][x] != 0]
        assert len(nz) == 1 and nz[0] in (-1, 1), f"Invalid support at x={x}: {[T[i][x] for i in range(4)]}"
    # Check sum_i PAF(T_i, d) == 0 for all 1 <= d < t
    for d in range(1, t):
        s = 0
        for i in range(4):
            ti = T[i]
            s += sum(ti[x] * ti[(x + d) % t] for x in range(t))
        if s != 0:
            raise AssertionError(f"T-matrix PAF failed at d={d}: got {s}, expected 0")
    row_sums = [sum(T[i]) for i in range(4)]
    assert sum(x * x for x in row_sums) == t
    return t, row_sums


def build_cooper_wallis_hadamard(T, w, full_check=True):
    """Build exact Cooper-Wallis Hadamard matrix H of order n = 4*t*w from T-matrix T and WT(w)."""
    t, _ = verify_tmatrix_seqs(T)
    verify_williamson(w)
    W = WILLIAMSON_WT[w]
    tw = t * w
    n = 4 * tw

    # Group-developed sequences A, B, C, D over Z_t x Z_w (index x*w + y):
    # A =  T0*W0 + T1*W1 + T2*W2 + T3*W3
    # B = -T0*W1 + T1*W0 + T2*W3 - T3*W2
    # C = -T0*W2 - T1*W3 + T2*W0 + T3*W1
    # D = -T0*W3 + T1*W2 - T2*W1 + T3*W0
    def make_seq(terms):
        seq = [0] * tw
        for x in range(t):
            for sgn, ti, wi in terms:
                tv = T[ti][x]
                if tv != 0:
                    c = sgn * tv
                    wseq = W[wi]
                    base = x * w
                    for y in range(w):
                        seq[base + y] = c * wseq[y]
                    break
        return seq

    a = make_seq([(1, 0, 0), (1, 1, 1), (1, 2, 2), (1, 3, 3)])
    b = make_seq([(-1, 0, 1), (1, 1, 0), (1, 2, 3), (-1, 3, 2)])
    c = make_seq([(-1, 0, 2), (-1, 1, 3), (1, 2, 0), (1, 3, 1)])
    d = make_seq([(-1, 0, 3), (1, 1, 2), (-1, 2, 1), (1, 3, 0)])

    # Verify exact Z_t x Z_w autocorrelation sum == 0 for all (dx, dy) != (0, 0)
    for dx in range(t):
        for dy in range(w):
            if dx == 0 and dy == 0:
                continue
            tot = 0
            for seq in (a, b, c, d):
                for x in range(t):
                    x2 = (x + dx) % t
                    b1, b2 = x * w, x2 * w
                    for y in range(w):
                        tot += seq[b1 + y] * seq[b2 + (y + dy) % w]
            assert tot == 0, f"Cooper-Wallis PAF failed at ({dx},{dy}): {tot}"

    if not full_check:
        return (a, b, c, d)

    # Construct full n x n Goethals-Seidel matrix H over Z_t x Z_w
    def sub_g(i, j):
        x1, y1 = divmod(i, w)
        x2, y2 = divmod(j, w)
        return ((x1 - x2) % t) * w + ((y1 - y2) % w)

    def neg_g(i):
        x, y = divmod(i, w)
        return ((-x) % t) * w + ((-y) % w)

    neg_idx = [neg_g(i) for i in range(tw)]

    def mat_A(seq):
        return [[seq[sub_g(j, i)] for j in range(tw)] for i in range(tw)]

    def mat_BR(seq):
        return [[seq[sub_g(neg_idx[j], i)] for j in range(tw)] for i in range(tw)]

    def mat_BTR(seq):
        return [[seq[sub_g(i, neg_idx[j])] for j in range(tw)] for i in range(tw)]

    A = mat_A(a)
    BR, CR, DR = mat_BR(b), mat_BR(c), mat_BR(d)
    BTR, CTR, DTR = mat_BTR(b), mat_BTR(c), mat_BTR(d)

    gs = [
        [(1, A),   (1, BR),   (1, CR),   (1, DR)],
        [(-1, BR), (1, A),    (1, DTR),  (-1, CTR)],
        [(-1, CR), (-1, DTR), (1, A),    (1, BTR)],
        [(-1, DR), (1, CTR),  (-1, BTR), (1, A)],
    ]
    H = [[0] * n for _ in range(n)]
    for bi in range(4):
        for bj in range(4):
            sgn, blk = gs[bi][bj]
            r0, c0 = bi * tw, bj * tw
            for i in range(tw):
                src = blk[i]
                dst = H[r0 + i]
                if sgn == 1:
                    for j in range(tw):
                        dst[c0 + j] = src[j]
                else:
                    for j in range(tw):
                        dst[c0 + j] = -src[j]

    for i in range(n):
        ri = H[i]
        for j in range(i, n):
            dot = sum(x * y for x, y in zip(ri, H[j]))
            expected = n if i == j else 0
            assert dot == expected, f"H @ H^T failed at ({i},{j}): {dot} != {expected}"
    return H


def self_test():
    for w in sorted(WILLIAMSON_WT.keys()):
        verify_williamson(w)
    print(f"[PASS] All Williamson matrices WT({sorted(WILLIAMSON_WT.keys())}) verified exactly!")

    # Small T-matrix of order t = 3:
    # T0 = [1, 1, 0], T1 = [0, 0, 1], T2 = [0, 0, 0], T3 = [0, 0, 0]?
    # Check PAF(1): T0 gives 1, T1 gives 0 -> sum = 1 != 0.
    # Proper T-matrix of order t = 3: T0 = [1, 0, 0], T1 = [0, 1, -1], T2 = [0, 0, 0], T3 = [0, 0, 0]!
    # Check PAF(1) of T1: 0*1 + 1*(-1) + (-1)*0 = -1 != 0.
    # Try t = 3: s1^2+s2^2+s3^2+s4^2 = 3 => (1, 1, 1, 0)!
    # T0 = [1, 0, 0], T1 = [0, 1, 0], T2 = [0, 0, 1], T3 = [0, 0, 0]!
    # Check PAF(1): 0 + 0 + 0 + 0 = 0! PAF(2): 0!
    T3 = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]
    H60 = build_cooper_wallis_hadamard(T3, 5, full_check=True)
    print(f"[PASS] Cooper-Wallis self-test t=3, w=5 -> H({len(H60)}): H in {{+-1}} and H @ H^T = 60 * I_60 VERIFIED EXACTLY!")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        with open(sys.argv[1]) as f:
            data = json.load(f)
        T = expand_tmatrix_from_orbits(data["t"], data["g"], data["orbit_mat"], data["orbit_sign"])
        t, rsums = verify_tmatrix_seqs(T)
        w = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
        print(f"[PASS] T-matrix t={t} verified! row_sums={rsums}, sum_sq={sum(x*x for x in rsums)}")
        build_cooper_wallis_hadamard(T, w, full_check=(4 * t * w <= 200))
        print(f"[PASS] Cooper-Wallis H({4*t*w}) (w={w}) group-developed Z_{t} x Z_{w} orthogonality VERIFIED EXACTLY!")
    else:
        self_test()
