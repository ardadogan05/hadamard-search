#!/usr/bin/env python3
"""Exact completion of one row from its odd moments for prime-length GS search.

Let p be the sequence length, H the odd-order multiplier group, h=|H|,
q=(p-1)/h. In F_p[t]/(t^p), the substitution x -> exp(t) maps the cyclic
convolution algebra F_p[x]/(x^p-1) into the truncated power series algebra.
The truncated exponential is multiplicative for residues modulo p because
all degrees below p have invertible factorials. Its coefficient at degree
k in a transformed sequence is sum_a f(a)*a**k/k!. H-invariance makes the
coefficients vanish unless h divides k. With z=t**h the transformed row is
A(z)=E(z)+O(z), split into even and odd degrees. Negating positions maps
z to -z, so autocorrelation maps to E²-O² modulo z**(q+1).

The Goethals-Seidel target reduces to zero in F_p: every diagonal entry of
its combined autocorrelation is 4p and every off-diagonal entry is zero.
Given three complete rows and the fourth row's odd part and nonzero row
sum, the fourth even part is a unique formal square root with that chosen
constant term. Invert the moment transform and check for signs +/-1.

This reduction is exact for sign vectors of prime length. The transform is
invertible, and the combined autocorrelation is divisible by four at each
shift. At a nonzero shift its value lies in (-4p,4p) if at least one row is
nonconstant: invariance by that shift would force a row to be constant,
and all products cannot be -1 on an odd cycle. Thus a multiple of both p
and four must vanish. For the 4412 target q=58, so 29 even coefficients are
recovered instead of chosen. Rejection concerns the supplied partial rows.
"""

import argparse
import json
from math import isqrt
from pathlib import Path

from verify import build_orbits


def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, isqrt(n) + 1))


class OrbitMoments:
    def __init__(self, p, generator):
        if p < 3 or not is_prime(p) or generator % p == 0:
            raise ValueError('expected an odd prime length and a unit multiplier')
        self.p = p
        self.orbits, _ = build_orbits(p, generator)
        self.h = len(self.orbits[1])
        if self.h % 2 == 0:
            raise ValueError('expected an odd-order multiplier')
        self.q = len(self.orbits) - 1
        self.nodes = [pow(orbit[0], self.h, p) for orbit in self.orbits[1:]]
        if len(set(self.nodes)) != self.q:
            raise ValueError('expected distinct multiplier orbit nodes')
        factorials = [1]
        for k in range(1, p):
            factorials.append(k * factorials[-1] % p)
        self.factorials = factorials[::self.h]
        self.inverse_factorials = [pow(f, -1, p) for f in self.factorials]

    def encode(self, signs):
        if len(signs) != self.q + 1 or any(type(x) is not int or x not in (-1, 1) for x in signs):
            raise ValueError(f'expected {self.q + 1} signs in orbit order')
        return [
            (signs[0] + self.h * sum(signs[1:])) % self.p,
            *(
                self.h * self.inverse_factorials[j]
                * sum(sign * pow(node, j, self.p) for node, sign in zip(self.nodes, signs[1:])) % self.p
                for j in range(1, self.q + 1)
            ),
        ]

    def decode(self, coefficients):
        p = self.p
        moments = [c * f % p for c, f in zip(coefficients, self.factorials)]
        # The nodes form the q-th roots of unity. Indices 1,...,q give
        # a full inverse DFT, using index q in place of index 0.
        scale = pow(self.h * self.q, -1, p)
        nonzero = [
            scale * sum(moments[j] * pow(node, -j, p) for j in range(1, self.q + 1)) % p
            for node in self.nodes
        ]
        return [(moments[0] - self.h * sum(nonzero)) % p, *nonzero]

    def complete(self, other_rows, odd_coefficients, row_sum):
        p = self.p
        if len(other_rows) != 3 or len(odd_coefficients) != self.q // 2 or row_sum % p == 0:
            raise ValueError('expected three rows, the missing odd moments, and a nonzero row sum modulo p')
        others = [self.encode(row) for row in other_rows]
        if (sum(row[0] ** 2 for row in others) + row_sum**2) % p:
            return None
        odd = [0] * (self.q + 1)
        odd[1::2] = odd_coefficients
        result = odd.copy()
        result[0] = row_sum % p
        inverse_derivative = pow(2 * row_sum, -1, p)
        for k in range(2, self.q + 1, 2):
            other_paf = sum(
                (-1) ** j * row[j] * row[k - j]
                for row in others for j in range(k + 1)
            )
            square_odd = sum(odd[j] * odd[k - j] for j in range(1, k, 2))
            known_even = sum(result[j] * result[k - j] for j in range(2, k, 2))
            result[k] = (square_odd - other_paf - known_even) * inverse_derivative % p
        decoded = self.decode(result)
        if any(a not in (1, p - 1) for a in decoded):
            return None
        signs = [1 if a == 1 else -1 for a in decoded]
        if signs[0] + self.h * sum(signs[1:]) != row_sum:
            return None
        if all(len(set(row)) == 1 for row in [*other_rows, signs]):
            return None
        return signs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('candidate', type=Path, nargs='?', help='optional b=0 candidate JSON')
    parser.add_argument('--row', type=int, choices=range(1, 5), default=1, help='row to complete (1–4)')
    args = parser.parse_args()
    if args.candidate is None:
        model = OrbitMoments(1103, 749)
        print(f'n=4412, h={model.h}, q={model.q}: {model.q // 2} even coefficients determined by the three other rows and remaining odd part')
        return 0

    data = json.loads(args.candidate.read_text())
    if data['b'] != 0 or len(data['orbit_signs']) != 4:
        parser.error('expected four b=0 orbit sign rows')
    model = OrbitMoments(data['m'], data['g'])
    rows = data['orbit_signs']
    row = rows[args.row - 1]
    others = rows[:args.row - 1] + rows[args.row:]
    row_sum = row[0] + model.h * sum(row[1:])
    completed = model.complete(others, model.encode(row)[1::2], row_sum)
    if completed is None:
        print(f'No signed completion for row {args.row} with the supplied odd moments and row sum {row_sum}.')
        return 1
    print(f'Exact completion for row {args.row} with row sum {row_sum}.')
    print(json.dumps(completed))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
