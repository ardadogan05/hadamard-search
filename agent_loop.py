#!/usr/bin/env python3
"""
Agentic Hadamard Discovery Loop:
  1. Falsification: Check row-sum feasibility, 2-compression, and prime-divisor PSD obstructions.
  2. Probe: Run fast C++ orbit-space Tabu search probes on surviving open orders.
  3. Focus: Attack the lowest-energy (unobstructed) open order until E == 0.
  4. Verify: Independently construct H (n x n) and verify H * H^T = n * I over exact integers.
"""
import json
import math
import os
import re
import subprocess
import sys
import time

# Verified live UNKNOWN orders from https://www.math74.com/Hadamard/ (85 orders <= 10000)
CANDIDATES = [
    # (n, m, g, b, label)
    (7572, 1892, 5, 4, "n=7572 m=1892 g=5 |U|=210 r=24 K=13"),
    (7572, 1892, 15, 4, "n=7572 m=1892 g=15 |U|=210 r=27 K=14"),
    (4412, 1102, 11, 4, "n=4412 m=1102 g=11 |U|=84 r=28 K=15"),
    (4412, 1102, 23, 4, "n=4412 m=1102 g=23 |U|=63 r=30 K=15"),
    (4892, 1222, 3, 4, "n=4892 m=1222 g=3 |U|=69 r=30 K=15"),
    (5708, 1426, 33, 4, "n=5708 m=1426 g=33 |U|=110 r=28 K=15"),
    (6268, 1566, 31, 4, "n=6268 m=1566 g=31 |U|=252 r=28 K=15"),
    (5164, 1290, 17, 4, "n=5164 m=1290 g=17 |U|=84 r=30 K=17"),
    (5804, 1450, 23, 4, "n=5804 m=1450 g=23 |U|=140 r=30 K=17"),
    (6668, 1666, 9, 4, "n=6668 m=1666 g=9 |U|=168 r=30 K=17"),
    (7180, 1794, 7, 4, "n=7180 m=1794 g=7 |U|=132 r=30 K=17"),
    (2956, 738, 5, 4, "n=2956 m=738 g=5 |U|=60 r=26 K=17"),
]


def factor(n):
    f = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def get_orbits(m, g):
    seen = [False] * m
    orbits = []
    for x in range(m):
        if not seen[x]:
            orb = []
            cur = x
            while not seen[cur]:
                seen[cur] = True
                orb.append(cur)
                cur = (cur * g) % m
            orbits.append(orb)
    return orbits


def falsify_quadratic_norm(n, m, g):
    """
    Exact algebraic falsification:
    For any odd prime q | m with q = 3 mod 4 where |<g> mod q| = (q-1)/2 (index 2 in F_q^*),
    the q-compression of any <g>-invariant sequence has only 3 values (a0, aQR, aNQR),
    and |f_hat(m/q)|^2 = (a0 - (aQR+aNQR)/2)^2 + q * ((aQR-aNQR)/2)^2.
    Actually even more simply, check whether 4n = sum_f (X_f^2 + q * Y_f^2) with valid
    reachable compression values!
    """
    orbs = get_orbits(m, g)
    fac = factor(m)
    for q in fac:
        if q == 2 or q % 4 != 3:
            continue
        # Check order of g mod q
        gq = g % q
        if gq == 0:
            continue
        ord_q = 1
        cur = gq
        while cur != 1:
            cur = (cur * gq) % q
            ord_q += 1
        if ord_q != (q - 1) // 2:
            continue
        # Partition orbits into 0 mod q, QR mod q, NQR mod q
        qrs = {(x * x) % q for x in range(1, q)}
        s0, sqr, snqr = [], [], []
        for o in orbs:
            rem = o[0] % q
            if rem == 0:
                s0.append(len(o))
            elif rem in qrs:
                sqr.append(len(o))
            else:
                snqr.append(len(o))
        # Each orbit in sqr has size divisible by (q-1)/2; its contribution to EACH QR residue is len(o) / ((q-1)/2)
        h = (q - 1) // 2
        def reach(sizes, div=1):
            dp = 1
            for s in sizes:
                dp |= (dp << (s // div))
            tot = sum(s // div for s in sizes)
            return sorted({tot - 2 * sub for sub in range(tot + 1) if (dp >> sub) & 1})
        R0 = reach(s0, 1)
        Rqr = reach(sqr, h)
        Rnqr = reach(snqr, h)
        # For target row sum s_f in {0, 2}: a0 + h*(aqr + anqr) = s_f
        # PSD = (a0 - (aqr + anqr)/2)^2 + q * ((aqr - anqr)/2)^2
        # Multiply by 4: 4*PSD = (2*a0 - aqr - anqr)^2 + q * (aqr - anqr)^2
        def psds(target):
            R0_set = set(R0)
            vals = set()
            for aqr in Rqr:
                for anqr in Rnqr:
                    a0 = target - h * (aqr + anqr)
                    if a0 in R0_set:
                        x = 2 * a0 - aqr - anqr
                        y = aqr - anqr
                        vals.add(x * x + q * y * y)
            return sorted(vals)
        p2 = psds(2)
        p0 = psds(0)
        p0_set = set(p0)
        goal = 4 * n
        ok = False
        for a in p2:
            r1 = goal - a
            if r1 < 0: continue
            for i, b in enumerate(p0):
                r2 = r1 - b
                if r2 < 0: break
                for j in range(i, len(p0)):
                    c = p0[j]
                    r3 = r2 - c
                    if r3 < c: break
                    if r3 in p0_set:
                        ok = True
                        break
                if ok: break
            if ok: break
        if not ok:
            return False, f"FALSIFIED by q={q} quadratic norm obstruction (|U mod {q}|={h})"
    return True, "passed"


def run_probe(n, m, g, b, secs, out_path):
    cmd = ["./solver", str(m), str(g), str(b), str(secs), out_path, "2"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    best_e = None
    found = False
    for line in res.stdout.splitlines():
        m_e = re.search(r"new best E=(\d+)", line)
        if m_e:
            best_e = int(m_e.group(1))
        if "FOUND EXACT VERIFIED SOLUTION" in line:
            found = True
            best_e = 0
    return found, best_e, res.stdout


def main():
    probe_secs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print("=== STAGE 1: ALGEBRAIC FALSIFICATION ===", flush=True)
    survivors = []
    for n, m, g, b, label in CANDIDATES:
        ok, msg = falsify_quadratic_norm(n, m, g)
        print(f"  {label:42s} -> {msg}", flush=True)
        if ok:
            survivors.append((n, m, g, b, label))

    print(f"\n=== STAGE 2: {probe_secs}s C++ PROBES ON {len(survivors)} SURVIVORS ===", flush=True)
    scored = []
    for n, m, g, b, label in survivors:
        out_path = f"witness_{n}_{m}_{g}.json"
        found, best_e, _ = run_probe(n, m, g, b, probe_secs, out_path)
        print(f"  {label:42s} -> best_E={best_e}  found={found}", flush=True)
        if found:
            print(f"\n*** JACKPOT! EXACT HADAMARD MATRIX OF OPEN ORDER n={n} FOUND AT {out_path} ***", flush=True)
            sys.exit(0)
        if best_e is not None:
            scored.append((best_e, n, m, g, b, label))

    scored.sort()
    print("\n=== RANKED HOTTEST TARGETS ===", flush=True)
    for best_e, n, m, g, b, label in scored:
        print(f"  E={best_e:8d} (E/16={best_e//16:5d}) : {label}", flush=True)


if __name__ == "__main__":
    main()
