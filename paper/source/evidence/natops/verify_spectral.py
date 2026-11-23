#!/usr/bin/env python3
"""Check the companion's spectral meanings without data or learned models.

Uses Gram--Schmidt on Pascal columns independently of the implementation's QR,
then compares an incremental tensor calculation with the probability formula.
All computations here are standard-library binary64 arithmetic.
"""
import json
from math import comb, isclose, sqrt
from pathlib import Path


def qr_columns(length, copies):
    columns = []
    for degree in range(copies):
        v = [float(comb(r, degree)) for r in range(length)]
        for q in columns:
            dot = sum(a * b for a, b in zip(v, q))
            v = [a - dot * b for a, b in zip(v, q)]
        norm = sqrt(sum(a * a for a in v))
        columns.append([a / norm for a in v])
    return columns


def area(increments):
    x = y = xy = yx = 0.0
    for dx, dy in increments:
        xy += x * dy + dx * dy / 2
        yx += y * dx + dy * dx / 2
        x += dx
        y += dy
    return (xy - yx) / 2


def verify():
    cases = []
    for samples in (127, 128):
        bins = samples // 2 + 1
        q0, q1, _ = qr_columns(bins, 3)
        u = [r / (bins - 1) for r in range(bins)]
        frequency = [2 * r / samples for r in range(bins)]
        d = sqrt(bins * (bins + 1) / (12 * (bins - 1)))
        assert all(isclose(q, 1 / sqrt(bins), abs_tol=1e-14) for q in q0)
        assert all(isclose(q, (t - .5) / d, abs_tol=1e-14) for q, t in zip(q1, u))
        for name, power in [('distributed', [(r % 7 + 1) ** 2 for r in range(bins)]),
                            ('single_bin', [int(r == bins // 2) for r in range(bins)])]:
            total = sum(power)
            mass = [p / total for p in power]
            expected_gap = sum(abs(a - b) * pa * pb
                               for a, pa in zip(u, mass) for b, pb in zip(u, mass))
            j = area([(p, t * p) for p, t in zip(mass, u)])
            stored = sqrt(2) * area([(a * p, b * p) for a, b, p in zip(q0, q1, mass)])
            expected = sqrt(2) / (4 * sqrt(bins) * d) * expected_gap
            assert isclose(j, expected_gap / 4, abs_tol=1e-14)
            assert isclose(stored, expected, abs_tol=1e-14)
            f_gap = sum(abs(a - b) * pa * pb
                        for a, pa in zip(frequency, mass) for b, pb in zip(frequency, mass))
            assert isclose(f_gap, 2 * (bins - 1) / samples * expected_gap, abs_tol=1e-14)
            amplitude = [sqrt(p) for p in power]
            mean = sum(amplitude) / bins
            variance = sum((a - mean) ** 2 for a in amplitude) / bins
            assert isclose(total, bins * (mean * mean + variance), rel_tol=1e-14)
            cases.append({'samples': samples, 'bins': bins, 'mass': name,
                          'qr_separation_absolute_error': abs(stored - expected)})
    return {'all_checks_passed': True, 'cases': cases,
            'scope': 'QR first-column scale, spectral pair separation, odd/even frequency conversion, amplitude-energy identity; no dataset or learning'}


if __name__ == '__main__':
    report = verify()
    Path(__file__).with_name('verification_spectral.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
