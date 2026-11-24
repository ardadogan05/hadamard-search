#!/usr/bin/env python3
"""Recheck the three historical Theorem 5 cyclic multiplier exclusions.

For length m, the signed sequences must have row sums (2, 0, 0, 0)
up to signs and permutation, and their PAFs must sum to -4 at each
nonzero shift. These checks concern only the specified multiplier families.
"""

from collections import Counter
from itertools import product


def orbits(modulus, multiplier):
    seen = set()
    for residue in range(modulus):
        orbit = []
        while residue not in seen:
            seen.add(residue)
            orbit.append(residue)
            residue = residue * multiplier % modulus
        if orbit:
            yield orbit


def diagonal_bound(m, multiplier, shift):
    labels = {x: i for i, orbit in enumerate(orbits(m, multiplier)) for x in orbit}
    diagonal = sum(labels[x] == labels[(x + shift) % m] for x in range(m))
    return diagonal, 4 * (2 * diagonal - m)


def compressed_paf_vectors(m, multiplier, d):
    cells = list(orbits(d, multiplier))
    cell_id = {x: i for i, cell in enumerate(cells) for x in cell}
    values = [{0} for cell in cells]
    # Each original orbit projects onto one cell and contributes equally
    # to every residue in that cell. Its sign can be chosen independently.
    for orbit in orbits(m, multiplier):
        i = cell_id[orbit[0] % d]
        weight = sum(x % d == cells[i][0] for x in orbit)
        values[i] = {value + sign * weight for value in values[i] for sign in (-1, 1)}

    # PAF is invariant under multiplication and reversal of a shift.
    shifts = [cell[0] for cell in cells[1:] if cell[0] <= min(-x % d for x in cell)]
    correlations = [
        Counter((cell_id[x], cell_id[(x + shift) % d]) for x in range(d))
        for shift in shifts
    ]
    norm_target = 4 * m + 4 - 4 * (m // d)
    options = [
        [(value, len(cell) * value, len(cell) * value**2) for value in sorted(choices)]
        for cell, choices in zip(cells, values)
    ]
    vectors = {0: set(), 2: set()}
    for assignment in product(*options):
        row_sum = sum(entry[1] for entry in assignment)
        if row_sum not in vectors:
            continue
        norm = sum(entry[2] for entry in assignment)
        if norm > norm_target:
            continue
        paf = tuple(
            sum(weight * assignment[i][0] * assignment[j][0] for (i, j), weight in counts.items())
            for counts in correlations
        )
        vectors[row_sum].add((norm, *paf))
    target = (norm_target, *([-4 * (m // d)] * len(shifts)))
    return target, vectors[2], vectors[0]


def has_paf_quadruple(target, row_two, row_zero):
    if not row_two or not row_zero:
        return False
    row_zero = sorted(row_zero)
    min_zero = row_zero[0][0]
    min_two = min(row_two)[0]
    pairs = set()
    for i, left in enumerate(row_zero):
        for j in range(i, len(row_zero)):
            right = row_zero[j]
            if left[0] + right[0] > target[0] - min_two - min_zero:
                break
            pairs.add(tuple(a + b for a, b in zip(left, right)))
    for two in row_two:
        for zero in row_zero:
            if two[0] + zero[0] > target[0] - 2 * min_zero:
                break
            if tuple(a - b - c for a, b, c in zip(target, two, zero)) in pairs:
                return True
    return False


def main():
    for m, multiplier, shift in ((1892, 15, 946), (1666, 9, 238)):
        diagonal, lower_bound = diagonal_bound(m, multiplier, shift)
        if lower_bound <= -4:
            raise RuntimeError('Expected diagonal obstruction was not reproduced')
        print(f'n={4*m+4}, g={multiplier}, shift={shift}: D={diagonal}, PAF sum >= {lower_bound}')

    target, row_two, row_zero = compressed_paf_vectors(1892, 9, 473)
    if has_paf_quadruple(target, row_two, row_zero):
        raise RuntimeError('Expected compression obstruction was not reproduced')
    print(f'n=7572, g=9, d=473: {len(row_two)} row-2 and {len(row_zero)} row-0 PAF vectors, no matching quadruple')


if __name__ == '__main__':
    main()
