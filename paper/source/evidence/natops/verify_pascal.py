#!/usr/bin/env python3
"""Exact Pascal-code checks; no datasets, learner imports or experimental fits.

Run with Python's standard library. The report is written beside this script.
All path tensor coordinates through degree three are computed directly by
enumerating nondecreasing segment indices, including the factorial weights
for repeated indices. This calculation is separate from the linear-map and
Chen-product implementations used to transform/combine block summaries.

Channels are ordered by code copy and then by original channel. Global
windows use R >= 2 and 1 <= P <= R. The formula at R = P = 1 is used only
for a singleton block, where its sole weight is binom(0,0) / binom(0,0) = 1.
"""

from collections import Counter
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations_with_replacement, product
from math import comb, factorial
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent


def binomial(n, k):
    """Falling-factorial binomial for integer n, including negative shifts."""
    if k < 0:
        return F(0)
    value = F(1)
    for j in range(k):
        value *= F(n - j, j + 1)
    return value


def denominators(length, copies):
    assert 1 <= copies <= length
    return [F(comb(length - 1, p)) for p in range(copies)]


def normalized_code(r, length, copies):
    assert 0 <= r < length
    return [F(comb(r, p), comb(length - 1, p))
            for p in range(copies)]


def lift(increments, copies, normalized=True):
    length = len(increments)
    if normalized:
        assert 1 <= copies <= length
    return [[weight * F(value)
             for weight in (normalized_code(r, length, copies) if normalized
                            else [binomial(r, p) for p in range(copies)])
             for value in row]
            for r, row in enumerate(increments)]


def identity(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def matmul(a, b):
    assert len(a[0]) == len(b)
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), F(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def matvec(a, x):
    return [sum((v * w for v, w in zip(row, x)), F(0)) for row in a]


def inverse(a):
    """Exact Gauss-Jordan inversion, independent of the coding formulas."""
    n = len(a)
    rows = [list(map(F, row)) + identity(n)[i] for i, row in enumerate(a)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [v / scale for v in rows[j]]
        for i in range(n):
            if i != j:
                scale = rows[i][j]
                rows[i] = [v - scale * w for v, w in zip(rows[i], rows[j])]
    return [row[n:] for row in rows]


def shift(copies, h):
    return [[binomial(h, p - q) if q <= p else F(0)
             for q in range(copies)] for p in range(copies)]


def normalize_matrix(length, copies):
    d = denominators(length, copies)
    return [[F(i == j) / d[i] for j in range(copies)]
            for i in range(copies)]


def channel_map(code_map, channels):
    """Kronecker product of the code map with the channel identity."""
    return [[code_map[p][q] if c == e else F(0)
             for q in range(len(code_map[0])) for e in range(channels)]
            for p in range(len(code_map)) for c in range(channels)]


def direct_signature(increments, depth=3):
    """Direct piecewise-linear iterated integrals, not a Chen recursion."""
    dimension = len(increments[0])
    out = {(): F(1)}
    for degree in range(1, depth + 1):
        assignments = []
        for indices in combinations_with_replacement(range(len(increments)), degree):
            weight = F(1)
            for multiplicity in Counter(indices).values():
                weight /= factorial(multiplicity)
            assignments.append((indices, weight))
        for word in product(range(dimension), repeat=degree):
            value = F(0)
            for indices, weight in assignments:
                term = weight
                for r, c in zip(indices, word):
                    term *= increments[r][c]
                value += term
            out[word] = value
    return out


def tensor_map(signature, linear, depth=3):
    """Apply L^{tensor k} to each complete stored tensor level."""
    target_dimension = len(linear)
    options = [[(j, x) for j, x in enumerate(row) if x] for row in linear]
    out = {(): F(1)}
    for degree in range(1, depth + 1):
        for word in product(range(target_dimension), repeat=degree):
            value = F(0)
            for choices in product(*(options[i] for i in word)):
                term = signature[tuple(j for j, x in choices)]
                for j, x in choices:
                    term *= x
                value += term
            out[word] = value
    return out


def chen(left, right, dimension, depth=3):
    """Truncated tensor multiplication, separate from direct_signature."""
    return {word: sum((left.get(word[:j], F(0)) * right.get(word[j:], F(0))
                      for j in range(len(word) + 1)), F(0))
            for degree in range(depth + 1)
            for word in product(range(dimension), repeat=degree)}


def pascal_from_monomial(length, copies):
    """Coefficients of product_j ((R-1)u-j)/(R-1-j) in powers of u."""
    rows = []
    coefficients = [F(1)]
    for p in range(copies):
        rows.append(coefficients + [F(0)] * (copies - len(coefficients)))
        if p + 1 < copies:
            updated = [F(0)] * (len(coefficients) + 1)
            for k, value in enumerate(coefficients):
                updated[k] -= F(p, length - 1 - p) * value
                updated[k + 1] += F(length - 1, length - 1 - p) * value
            coefficients = updated
    return rows


def signed_fixture(length, channels):
    return [[F(((r + 1) * (c + 2)) % 7 - 3, c + 1)
             for c in range(channels)] for r in range(length)]


def verify():
    counts = Counter()

    def check(group, actual, expected):
        counts[group] += 1
        assert actual == expected, (group, actual, expected)

    def compare_tensors(group, actual, expected):
        check(group + '.word_sets', set(actual), set(expected))
        for word in actual:
            check(group + '.coordinates', actual[word], expected[word])

    for length in range(2, 9):
        for copies in range(1, length + 1):
            d = denominators(length, copies)
            for r in range(length):
                code = normalized_code(r, length, copies)
                check('normalized_definition', code,
                      [binomial(r, p) / d[p] for p in range(copies)])
                check('constant_copy', code[0], F(1))
                if copies >= 2:
                    check('linear_copy', code[1], F(r, length - 1))
                check('weights_in_unit_interval', all(0 <= a <= 1 for a in code), True)
            check('first_code', normalized_code(0, length, copies), [F(1)] + [F(0)] * (copies - 1))
            check('last_code', normalized_code(length - 1, length, copies), [F(1)] * copies)

    for copies in range(1, 7):
        for h in range(-4, 5):
            a = shift(copies, h)
            for r in range(7):
                check('pascal_shift', matvec(a, [binomial(r, p) for p in range(copies)]),
                      [binomial(r + h, p) for p in range(copies)])
            check('shift_inverse', matmul(a, shift(copies, -h)), identity(copies))
            for k in range(-4, 5):
                check('shift_group', matmul(a, shift(copies, k)), shift(copies, h + k))

    span_cases = [(2, 1, 2), (3, 2, 2), (5, 3, 2), (4, 4, 1), (6, 3, 1)]
    for length, copies, channels in span_cases:
        values = signed_fixture(length, channels)
        conversion = pascal_from_monomial(length, copies)
        reverse = inverse(conversion)
        check('basis_inverse', matmul(conversion, reverse), identity(copies))
        monomial = [[F(r, length - 1) ** p * x
                     for p in range(copies) for x in row]
                    for r, row in enumerate(values)]
        pascal = lift(values, copies)
        forward_map = channel_map(conversion, channels)
        reverse_map = channel_map(reverse, channels)
        check('basis_increment_map', [matvec(forward_map, v) for v in monomial], pascal)
        ordinary_signature = direct_signature(values)
        pascal_signature = direct_signature(pascal)
        monomial_signature = direct_signature(monomial)
        compare_tensors('basis_signature_naturality', tensor_map(monomial_signature, forward_map), pascal_signature)
        compare_tensors('basis_signature_inverse', tensor_map(pascal_signature, reverse_map), monomial_signature)
        projection = [[F(i == j) for j in range(copies * channels)] for i in range(channels)]
        compare_tensors('constant_copy_projection', tensor_map(pascal_signature, projection), ordinary_signature)
        if copies == 1:
            compare_tensors('P1_ordinary_signature', pascal_signature, ordinary_signature)
        if copies == 2:
            linear_lift = [row + [F(r, length - 1) * x for x in row]
                           for r, row in enumerate(values)]
            compare_tensors('P2_linear_signature', pascal_signature, direct_signature(linear_lift))

    for length in range(2, 8):
        for channels in range(1, 4):
            values = signed_fixture(length, channels)
            coded = lift(values, length)
            moments = [sum((v[p * channels + c] for v in coded), F(0))
                       for p in range(length) for c in range(channels)]
            d = denominators(length, length)
            recovered = [[sum(((-1) ** (p - r) * comb(p, r) * d[p]
                                * moments[p * channels + c]
                                for p in range(r, length)), F(0))
                          for c in range(channels)] for r in range(length)]
            check('full_moment_inversion_signed_vectors', recovered, values)

    block_cases = [(3, 1, 2, [1, 2]), (5, 3, 2, [2, 3]),
                   (6, 4, 1, [3, 3]), (4, 4, 1, [1, 3]),
                   (7, 4, 1, [2, 1, 4])]
    for length, copies, channels, blocks in block_cases:
        assert sum(blocks) == length
        values = signed_fixture(length, channels)
        target = direct_signature(lift(values, copies))
        unnormalized_joined = {(): F(1)}
        normalized_joined = {(): F(1)}
        start = 0
        for block_length in blocks:
            block = values[start:start + block_length]
            # Fixed global normalization follows shifting; no denominator for
            # the short block is needed in this unnormalized route.
            linear = channel_map(matmul(normalize_matrix(length, copies),
                                        shift(copies, start)), channels)
            block_signature = direct_signature(lift(block, copies, normalized=False))
            transformed = tensor_map(block_signature, linear)
            unnormalized_joined = chen(unnormalized_joined, transformed, copies * channels)

            # Independent local normalization uses only K=min(P,L) copies.
            # This rectangular transformation is defined even for L=1.
            short_copies = min(copies, block_length)
            h = [[binomial(start, p - q) * comb(block_length - 1, q)
                  / comb(length - 1, p) if q <= p else F(0)
                  for q in range(short_copies)] for p in range(copies)]
            short_signature = direct_signature(lift(block, short_copies))
            short_transformed = tensor_map(short_signature, channel_map(h, channels))
            compare_tensors('short_block_agrees_with_unnormalized', short_transformed, transformed)
            if block_length == 1:
                check('singleton_no_zero_denominator', [row[0] for row in h],
                      normalized_code(start, length, copies))
            normalized_joined = chen(normalized_joined, short_transformed, copies * channels)
            start += block_length
        compare_tensors('unnormalized_block_chen', unnormalized_joined, target)
        compare_tensors('normalized_short_block_chen', normalized_joined, target)

    return {
        'all_checks_passed': True,
        'arithmetic': 'Python standard-library fractions.Fraction',
        'scope': 'Exact construction and complete tensor levels zero through three; no dataset access or learner fitting',
        'verifier_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'check_counts': dict(sorted(counts.items())),
        'total_checks': sum(counts.values()),
        'signature_basis_cases_R_P_C': span_cases,
        'concatenation_cases_R_P_C_block_lengths': block_cases,
        'negative_shift_range': [-4, 4],
        'full_moment_inversion_R_range': [2, 7],
        'full_moment_inversion_channel_range': [1, 3],
        'conventions': [
            'Global normalized Pascal weights require 1 <= P <= R and R >= 2.',
            'Singleton blocks use only their constant local copy; higher global copies come from the rectangular shift map.',
            'Blocks partition one increment sequence without inserting basepoints or changing original channels or clocks.',
            'Negative shifts use generalized binomial coefficients and express algebraic rebasing.',
            'Tensor naturality is tested against independent direct iterated-integral enumeration, including repeated-segment factorials.',
        ],
    }


if __name__ == '__main__':
    report = verify()
    (ROOT / 'verification_pascal.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
