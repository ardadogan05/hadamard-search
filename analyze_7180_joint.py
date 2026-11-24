#!/usr/bin/env python3
"""Check whether one modulus-46 witness for 7180 g=29 extends through modulus 78."""

from collections import defaultdict
from itertools import product

from verify import build_orbits


N, M, G = 7180, 1794, 29
# Each triple uses increasing projected orbit representatives modulo 23.
# The two triples of a row are its even and odd parts modulo 46.
MOD46_WITNESS = (
    ((-23, -5, 5), (-19, -3, 7)),
    ((-5, 1, -3), (27, 1, -1)),
    ((-23, 1, 3), (23, 1, -5)),
    ((-7, 5, -5), (-37, -3, 7)),
)


def orbits(modulus):
    return sorted(tuple(sorted(orbit)) for orbit in build_orbits(modulus, G)[0])


def verify_mod46(rows, orbits23):
    residue_orbit = {x: j for j, orbit in enumerate(orbits23) for x in orbit}
    sequences = [
        [row[x % 2][residue_orbit[x % 23]] for x in range(46)]
        for row in rows
    ]
    if tuple(map(sum, sequences)) != (2, 0, 0, 0):
        raise RuntimeError("Incorrect row sums in the mod-46 witness")
    for shift in range(46):
        paf = sum(sum(x * y for x, y in zip(a, a[shift:] + a[:shift])) for a in sequences)
        if paf != (N if shift == 0 else 0) - 4 * (M // 46):
            raise RuntimeError(f"Incorrect mod-46 PAF at shift {shift}")


def possible_halves(profile, masks, mask_weight, full_mask, width, max_a):
    """Enumerate mod-78 half vectors compatible with three mod-46 row totals."""
    singleton, first, second = profile
    values = defaultdict(set)
    for left, right in product(masks[first], masks[second]):
        agreement = full_mask ^ (left ^ right)
        a = mask_weight[agreement]
        if a > max_a:
            continue
        for base in masks[singleton]:
            b = a - 2 * mask_weight[agreement & (left ^ base)]
            values[a, b].add(tuple(
                (2 * (base >> i & 1) - 1)
                + 22 * ((left >> i & 1) + (right >> i & 1) - 1)
                for i in range(width)
            ))
    return values


def compatible_energy(left, right, max_a):
    return {(a + c, b + d) for a, b in left for c, d in right if a + c <= max_a}


def check_witness():
    orbits23, orbits39 = orbits(23), orbits(39)
    partition = {
        frozenset(x for x in range(M) if x % 2 == parity and x % 23 in row and x % 39 in col)
        for parity, row, col in product(range(2), orbits23, orbits39)
    }
    if partition != {frozenset(orbit) for orbit in orbits(M)}:
        raise RuntimeError("Unexpected CRT multiplier-orbit structure")
    if list(map(len, orbits23)) != [1, 11, 11]:
        raise RuntimeError("Unexpected modulus-23 orbit sizes")

    verify_mod46(MOD46_WITNESS, orbits23)
    weights = list(map(len, orbits39))
    width = len(weights)
    full_mask = (1 << width) - 1
    mask_weight = [
        sum(weight for i, weight in enumerate(weights) if mask >> i & 1)
        for mask in range(full_mask + 1)
    ]
    masks = defaultdict(list)
    for mask, weight in enumerate(mask_weight):
        masks[2 * weight - 39].append(mask)

    # Each mod-78 coordinate is s+22t with s=±1 and t in {-1,0,1}.
    # The required energy gives 11*A+B=154, where A=sum(t²), B=sum(s*t).
    # With |B| <= A, A is at most 15.
    target_energy = N - 4 * (M // 78)
    scaled_energy = (target_energy - 8 * 39) // 44
    max_a = scaled_energy // 10
    halves = [
        possible_halves(profile, masks, mask_weight, full_mask, width, max_a)
        for row in MOD46_WITNESS for profile in row
    ]
    if any(not half for half in halves):
        raise RuntimeError("A mod-46 parity half fails CRT reachability")

    allowed_halves = []
    for i, half in enumerate(halves):
        others = {(0, 0)}
        for j, other in enumerate(halves):
            if i != j:
                others = compatible_energy(others, other, max_a)
        allowed_halves.append([
            vector for (a, b), vectors in half.items()
            if any(a + c <= max_a and 11 * (a + c) + b + d == scaled_energy for c, d in others)
            for vector in vectors
        ])

    # A shift of 39 interchanges the parity halves at each residue modulo 39.
    joint_pafs = {(0, 0)}
    for even, odd in zip(allowed_halves[::2], allowed_halves[1::2]):
        row_pafs = {
            (sum(w * (x*x + y*y) for w, x, y in zip(weights, e, o)),
             2 * sum(w * x*y for w, x, y in zip(weights, e, o)))
            for e, o in product(even, odd)
        }
        joint_pafs = {
            (energy + p0, paf + p39) for energy, paf in joint_pafs for p0, p39 in row_pafs
            if energy + p0 <= target_energy
        }
    target_shift = -4 * (M // 78)
    attainable = sorted(paf for energy, paf in joint_pafs if energy == target_energy)
    print(f"Energy-compatible half-vector counts: {list(map(len, allowed_halves))}")
    print(f"Attainable shift-39 PAFs at energy {target_energy}: {attainable}")
    if target_shift in attainable:
        print(f"The sample passes the modulus-78 energy and shift-39 check ({target_shift}).")
    else:
        print(f"The sample cannot extend to modulus 78 (required shift-39 PAF {target_shift}).")


if __name__ == "__main__":
    check_witness()
