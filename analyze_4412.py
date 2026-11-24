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


class BinaryOrbitProjection:
    """Generate fourth-row candidates by a characteristic-two norm equation.

    Use the minus supports D_i. Their four difference counts sum to
    p*delta_0 + (sum_i |D_i| - p)*1. At a primitive p-th root zeta in
    characteristic two the constant vector vanishes, leaving
    sum_i D_i(zeta)*D_i(zeta^-1) = 1.

    The multiplier and Frobenius (x -> x²) give two orbits on nonzero
    frequencies, exchanged by inversion. Over F_2, the multiplier-fixed
    algebra therefore has factors F_2 at 1 and two fields of degree q/2
    at zeta and zeta^-1. The evaluation map is inverted by binary Gaussian
    elimination. The fourth row's size fixes its evaluation at 1. Its
    remaining norm is fixed by the other rows, so any nonzero choice for
    one field component determines the opposite component by division.
    The zero norm permits either component to vanish. Row size and exact
    integer PAF must still be checked after this modular reduction.
    """

    # Irreducible polynomials with degrees ord_p(2), encoded as bit masks.
    MODULI = {
        1103: 0x20000005,
        487: 0xD85EA17C286ABB83052F4B898DB4C3BB531B7E1B8666E18E4F005623C3A81,
    }

    def __init__(self, model):
        if model.p not in self.MODULI:
            raise ValueError('binary projection supports prime lengths 487 and 1103')
        self.model = model
        self.modulus = self.MODULI[model.p]
        self.degree = self.modulus.bit_length() - 1
        exponent = ((1 << self.degree) - 1) // model.p
        root = self.power(2, exponent)
        if root == 1 or self.power(root, model.p) != 1:
            raise ValueError('field generator did not give a primitive p-th root')
        roots = [1]
        for _ in range(1, model.p):
            roots.append(self.multiply(roots[-1], root))
        self.columns = []
        self.pivots = {}
        forward_pivots = {}
        for i, orbit in enumerate(model.orbits):
            a = b = 0
            for x in orbit:
                a ^= roots[x]
                b ^= roots[-x % model.p]
            self.columns.append((a, b))
            self._insert(forward_pivots, a, 1 << i)
            if not self._insert(self.pivots, self._pack(1, a, b), 1 << i):
                raise ValueError('projection is not injective on orbit signs')
        self.forward_rank = len(forward_pivots)
        if self.forward_rank != model.q // 2:
            raise ValueError('expected two field factors of degree q/2')

    @staticmethod
    def _insert(pivots, vector, mask):
        while vector:
            bit = vector.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = vector, mask
                return True
            pivot, origin = pivots[bit]
            vector ^= pivot
            mask ^= origin
        return False

    def multiply(self, a, b):
        product = 0
        high_bit = 1 << self.degree
        while b:
            if b & 1:
                product ^= a
            b >>= 1
            a <<= 1
            if a & high_bit:
                a ^= self.modulus
        return product

    def power(self, base, exponent):
        result = 1
        while exponent:
            if exponent & 1:
                result = self.multiply(result, base)
            base = self.multiply(base, base)
            exponent >>= 1
        return result

    def _pack(self, parity, forward, backward):
        return parity | (forward << 1) | (backward << (self.degree + 1))

    def project(self, signs):
        if len(signs) != len(self.columns) or any(type(sign) is not int or sign not in (-1, 1) for sign in signs):
            raise ValueError(f'expected {len(self.columns)} signs in orbit order')
        forward = backward = 0
        for sign, (a, b) in zip(signs, self.columns):
            if sign == -1:
                forward ^= a
                backward ^= b
        return forward, backward

    def remaining_norm(self, other_rows):
        if len(other_rows) != 3:
            raise ValueError('expected three complete rows')
        norm = 1
        for row in other_rows:
            a, b = self.project(row)
            norm ^= self.multiply(a, b)
        return norm

    def mask_from_packed(self, value):
        """Apply the linear inverse to packed parity and field coordinates."""
        mask = 0
        while value:
            pivot = self.pivots.get(value.bit_length() - 1)
            if pivot is None:
                return None
            basis, origin = pivot
            value ^= basis
            mask ^= origin
        return mask

    def complete(self, norm, projection, row_sum, *, reverse=False):
        """Complete one projection. At zero norm try both orientations."""
        size = (self.model.p - row_sum) // 2
        if not 0 <= size <= self.model.p or row_sum != self.model.p - 2 * size or (norm and not projection):
            return None
        if not 0 <= projection < 1 << self.degree:
            return None
        # Both components inhabit the same subfield, so inversion can use
        # its order even when the ambient field is larger.
        other = self.multiply(norm, self.power(projection, (1 << self.forward_rank) - 2)) if norm else 0
        forward, backward = (other, projection) if reverse else (projection, other)
        if self.multiply(forward, backward) != norm:
            return None
        mask = self.mask_from_packed(self._pack(size % 2, forward, backward))
        if mask is None:
            return None
        signs = [-1 if mask >> i & 1 else 1 for i in range(self.model.q + 1)]
        if signs[0] + self.model.h * sum(signs[1:]) == row_sum:
            return signs
        return None


def export_binary_scan(path, model, other_rows, row_sum):
    """Write the exact fixed-triple inputs for scan_binary_4412.cpp."""
    if model.p != 1103 or model.h != 19:
        raise ValueError('the binary scanner requires length 1103 and multiplier order 19')
    if sum((row[0] + model.h * sum(row[1:])) ** 2 for row in other_rows) + row_sum**2 != 4 * model.p:
        raise ValueError('row-sum squares must add to 4412')
    binary = BinaryOrbitProjection(model)
    norm = binary.remaining_norm(other_rows)
    if not norm:
        raise ValueError('the binary scanner requires a nonzero residual norm')
    inverse_columns = [binary.mask_from_packed(1 << bit) for bit in range(1 + 2 * binary.degree)]
    if any(col is None for col in inverse_columns):
        raise ValueError('the inverse does not cover all packed coordinates')

    orbit_ids = [0] * model.p
    for i, orbit in enumerate(model.orbits):
        for x in orbit:
            orbit_ids[x] = i
    supports = [[int(row[i] == -1) for i in orbit_ids] for row in other_rows]
    size = (model.p - row_sum) // 2
    if not 0 <= size <= model.p or model.p - 2 * size != row_sum:
        raise ValueError('invalid missing row sum')
    threshold = sum(sum(support) for support in supports) + size - model.p
    shifts = []
    seen = {0}
    for orbit in model.orbits[1:]:
        d = orbit[0]
        if orbit_ids[d] in seen:
            continue
        seen.update((orbit_ids[d], orbit_ids[-d % model.p]))
        residual = threshold - sum(
            support[x] * support[(x + d) % model.p]
            for support in supports for x in range(model.p)
        )
        shifts.append((d, residual))

    lines = [
        f'{norm} {size}',
        ' '.join(map(str, inverse_columns)),
        ' '.join(map(str, orbit_ids)),
        *(f'{shift} {residual}' for shift, residual in shifts),
    ]
    path.write_text('\n'.join(lines) + '\n')
    print(f'Wrote fixed-triple scan input to {path} (norm={norm:#x}, size={size}, shifts={len(shifts)}).')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('candidate', type=Path, nargs='?', help='optional b=0 candidate JSON')
    parser.add_argument('--row', type=int, choices=range(1, 5), default=1, help='row to complete (1–4)')
    parser.add_argument('--binary', action='store_true', help='complete from one binary field projection')
    parser.add_argument('--export-binary-scan', type=Path, help='write fixed-triple input for scan_binary_4412.cpp')
    args = parser.parse_args()
    if args.candidate is None:
        if args.export_binary_scan:
            parser.error('--export-binary-scan requires a candidate JSON')
        model = OrbitMoments(1103, 749)
        if args.binary:
            binary = BinaryOrbitProjection(model)
            projection_count = 1 << binary.forward_rank
            print(f'n=4412, h={model.h}, q={model.q}: two {binary.forward_rank}-bit projections')
            print(f'Fourth-row bounds before size and PAF checks: {projection_count - 1} at nonzero norm, {2 * projection_count - 1} at zero norm.')
        else:
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
    if args.export_binary_scan:
        try:
            export_binary_scan(args.export_binary_scan, model, others, row_sum)
        except ValueError as error:
            parser.error(str(error))
        return 0
    if args.binary:
        binary = BinaryOrbitProjection(model)
        forward, backward = binary.project(row)
        norm = binary.remaining_norm(others)
        completed = binary.complete(norm, forward or backward, row_sum, reverse=(forward == 0))
        if completed is None:
            print(f'No row-size-compatible binary completion for row {args.row}.')
            return 1
        print(f'Binary candidate for row {args.row} with row sum {row_sum}. Verify its integer PAF to establish a solution.')
        print(json.dumps(completed))
        return 0

    completed = model.complete(others, model.encode(row)[1::2], row_sum)
    if completed is None:
        print(f'No signed completion for row {args.row} with the supplied odd moments and row sum {row_sum}.')
        return 1
    print(f'Exact completion for row {args.row} with row sum {row_sum}.')
    print(json.dumps(completed))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
