#!/usr/bin/env python3
"""Exact offset-signature identities and dimension checks; no learning results.

Uses Python's standard library and Fraction arithmetic. Run directly to write
verification_offset.json beside this file. Both clock conventions are explicit:
four equally timed increments, and an additional zero-time basepoint slot.
"""
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations, product
from math import factorial
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def signature_two(increments):
    """Tensor Chen products of the signatures exp(z) of straight segments."""
    d = len(increments[0])
    m = [F(0)]*d
    b = [[F(0)]*d for _ in range(d)]
    for values in increments:
        z = list(map(F, values))
        b = [[b[i][j] + m[i]*z[j] + z[i]*z[j]/2
              for j in range(d)] for i in range(d)]
        m = [a + c for a, c in zip(m, z)]
    return m, b


def signature_three(increments):
    """All tensor coordinates through degree three, by straight-segment products."""
    d = len(increments[0])
    words = [w for k in range(4) for w in product(range(d), repeat=k)]
    current = {(): F(1)}
    for values in increments:
        segment = {}
        for word in words:
            value = F(1, factorial(len(word)))
            for i in word:
                value *= F(values[i])
            segment[word] = value
        current = {w: sum((current.get(w[:k], F(0))*segment[w[k:]]
                           for k in range(len(w)+1)), F(0)) for w in words}
    return current


def log_two(increments):
    m, b = signature_two(increments)
    j = [[(b[a][c]-b[c][a])/2 for c in range(len(m))]
         for a in range(len(m))]
    assert all(b[a][c] == j[a][c] + m[a]*m[c]/2
               for a in range(len(m)) for c in range(len(m)))
    return m, j


def lift(increments):
    """Code-major ordering: all ordinary channels, then all weighted channels."""
    r_count = len(increments)
    assert r_count >= 2
    return [list(map(F, z)) + [F(r, r_count-1)*value for value in z]
            for r, z in enumerate(increments)]


def pair(a, b):
    assert a != b
    return min(a, b), max(a, b)


def lifted_edges(d, edges, copies):
    out = {pair(p*d+i, q*d+i) for i in range(d)
           for p, q in combinations(range(copies), 2)}
    out.update(pair(p*d+i, q*d+j) for i, j in edges
               for p in range(copies) for q in range(copies))
    return out


def serial(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    return value


def verify():
    spread = [[F(1, 4), F(z)] for z in [1, 0, 0, 1]]
    central = [[F(1, 4), F(z)] for z in [0, 1, 1, 0]]
    ordinary_spread = signature_two(spread)
    assert ordinary_spread == signature_two(central)
    ordinary_m, ordinary_j = log_two(spread)
    assert ordinary_m == [1, 2] and ordinary_j[0][1] == 0
    spread_m, spread_j = log_two(lift(spread))
    central_m, central_j = log_two(lift(central))
    assert spread_m == central_m == [F(1), F(2), F(1, 2), F(1)]
    assert spread_j[1][3] == F(1, 2)
    assert central_j[1][3] == F(1, 6)

    # Degree three is exactly the first ordinary level separating the running pair.
    degree_three = [signature_three(v) for v in [spread, central]]
    assert all(degree_three[0][w] == degree_three[1][w]
               for w in degree_three[0] if len(w) <= 2)
    assert [t[1, 0, 1] for t in degree_three] == [F(5, 6), F(1, 3)]
    assert [t[0, 0, 1] for t in degree_three] == [F(19, 48), F(13, 48)]
    log_xtx = [t[1, 0, 1] - (t[1,]*t[0, 1]+t[1, 0]*t[1,])/2
               + t[1,]*t[0,]*t[1,]/3 for t in degree_three]
    assert log_xtx == [F(1, 6), F(-1, 3)]

    # General equal-time-grid relation; independent Chen product vs offset area.
    uniform_grid_cases = 0
    for values in [[1, 0, 0, 1], [0, 1, 1, 0],
                   [F(3, 2), F(-2, 3)], [2, -1, 3, -2, 0, 1, 4]]:
        r_count = len(values)
        increments = [[F(1, r_count), F(z)] for z in values]
        t = signature_three(increments)
        _, area = log_two(lift(increments))
        correction = sum((F(z)**2 for z in values), F(0))/(12*(r_count-1))
        assert area[1][3] == F(r_count, 2*(r_count-1))*t[1, 0, 1]-correction
        uniform_grid_cases += 1

    # No universal degree-three recovery of the segmentwise discrete offset area.
    collision = [[F(2), F(1), F(1), F(1)],
                 [F(17, 8), F(5, 8), F(11, 8), F(7, 8)]]
    paths = [[[F(1, 4), z] for z in values] for values in collision]
    assert signature_three(paths[0]) == signature_three(paths[1])
    collision_areas = [log_two(lift(path))[1][1][3] for path in paths]
    assert collision_areas == [F(8, 3), F(509, 192)]
    assert collision_areas[0] != collision_areas[1]

    # Appending a zero slot changes the code normalization, not the conclusion.
    with_basepoint_spread = [[F(0), F(0)]] + spread
    with_basepoint_central = [[F(0), F(0)]] + central
    assert signature_two(with_basepoint_spread) == ordinary_spread
    assert signature_two(with_basepoint_central) == ordinary_spread
    bp_m, bp_spread_j = log_two(lift(with_basepoint_spread))
    bp_other_m, bp_central_j = log_two(lift(with_basepoint_central))
    assert bp_m == bp_other_m == [F(1), F(2), F(5, 8), F(5, 4)]
    assert bp_spread_j[1][3] == F(3, 8)
    assert bp_central_j[1][3] == F(1, 8)

    # Projection and all linear/nonlinear readout identities on a signed fixture.
    fixture = [[F(2), F(-1)], [F(-1, 3), F(2, 5)],
               [F(3, 7), F(-2)], [F(1, 2), F(4, 3)], [F(-2, 5), F(1)]]
    d, r_count = 2, len(fixture)
    u = [F(r, r_count-1) for r in range(r_count)]
    m, j = log_two(lift(fixture))
    _, b = signature_two(lift(fixture))
    original_m, original_b = signature_two(fixture)
    assert m[:d] == original_m
    assert [row[:d] for row in b[:d]] == original_b
    for c in range(d):
        expected = sum(((u[s]-u[r])*fixture[r][c]*fixture[s][c]
                        for r in range(r_count) for s in range(r+1, r_count)), F(0))/2
        assert j[c][d+c] == expected
    c, e = 0, 1
    gap_symmetric = sum(((u[s]-u[r])*(fixture[r][c]*fixture[s][e]
                                     + fixture[r][e]*fixture[s][c])
                         for r in range(r_count) for s in range(r+1, r_count)), F(0))/2
    assert j[c][d+e] + j[e][d+c] == gap_symmetric
    q_ce = sum(((u[s]-u[r])*fixture[r][c]*fixture[s][e]
                for r in range(r_count) for s in range(r+1, r_count)), F(0))
    q_ec = sum(((u[s]-u[r])*fixture[r][e]*fixture[s][c]
                for r in range(r_count) for s in range(r+1, r_count)), F(0))
    moment_product = m[c]*m[d+e] - m[d+c]*m[e]
    assert q_ce == b[c][d+e]-b[d+c][e]
    assert q_ce == gap_symmetric+moment_product/2
    assert q_ce-q_ec == moment_product
    assert q_ce != gap_symmetric  # Omitting the quadratic moment term is wrong here.
    # The other linear log combination weights antisymmetric order by midpoint.
    midpoint_order = sum(((u[r]+u[s])*(fixture[r][c]*fixture[s][e]
                                      - fixture[r][e]*fixture[s][c])
                          for r in range(r_count) for s in range(r+1, r_count)), F(0))/2
    assert j[c][d+e] - j[e][d+c] == midpoint_order

    # Exact fixed-window clock identities, including the zero-time basepoint.
    clock_checks = []
    for r_count in [3, 5, 51]:
        length = r_count-1
        increments = [[F(0)] + [F(3, 2), F(-7, 3)]]
        increments += [[F(1, length)] + [F(((r+2)*(i+2)) % 7-3, i+1)
                                            for i in range(2)]
                       for r in range(1, r_count)]
        d = 3
        moments = [[sum((F(r, length)**p*row[i] for r, row in enumerate(increments)), F(0))
                    for i in range(d)] for p in range(4)]
        coded_m, coded_j = log_two(lift(increments))
        assert coded_m == moments[0] + moments[1]
        assert moments[0][0] == 1 and moments[1][0] == F(r_count, 2*length)
        for i in [1, 2]:
            assert moments[1][i] == coded_j[0][i] + F(r_count, 2*length)*moments[0][i] - increments[0][i]/(2*length)
            assert coded_j[0][d+i] == moments[2][i] - F(r_count, 2*length)*moments[1][i]
            assert coded_j[d][i] == moments[2][i]/2 - F(r_count, 4*length)*moments[0][i]
            assert coded_j[d][d+i] == moments[3][i]/2 - F(r_count, 4*length)*moments[1][i]
        clock_checks.append({"increments_including_basepoint": r_count,
                             "spatial_channels_checked": 2,
                             "first_moment_and_higher_moment_identities": "passed"})

    # Counts count actual retained unordered pairs; no learning matrix is involved.
    d = 25
    full = set(combinations(range(d), 2))
    local = {(0, i) for i in range(1, d)}
    for landmark in range(8):
        local.update(combinations(range(1+3*landmark, 4+3*landmark), 2))
    dimensions = {}
    for copies in [1, 2, 3]:
        for name, edges in [("local", local), ("full", full)]:
            selected = lifted_edges(d, edges, copies)
            formula = copies*d + copies*copies*len(edges) + d*copies*(copies-1)//2
            assert formula == copies*d+len(selected)
            dimensions[f"{name}_P{copies}"] = formula
    assert dimensions["local_P1"] == 73 and dimensions["full_P1"] == 325
    assert dimensions["local_P2"] == 267 and dimensions["full_P2"] == 1275
    offset_local_edges = lifted_edges(d, local, 2)
    ordinary_spatial = {e for e in local if 0 not in e}
    coded_clock_spatial = {pair(p*d, q*d+i) for p in range(2) for q in range(2)
                           for i in range(1, d)}
    clock_rich_control = ordinary_spatial | coded_clock_spatial | {(0, d)}
    assert len(ordinary_spatial) == 24 and len(coded_clock_spatial) == 96
    assert 2*d+len(clock_rich_control) == 171
    assert clock_rich_control < offset_local_edges
    extra_spatial = offset_local_edges-clock_rich_control
    assert len(extra_spatial) == 96
    assert all(a % d != 0 and b % d != 0 for a, b in extra_spatial)
    same_channel = {e for e in extra_spatial if e[0] % d == e[1] % d}
    assert len(same_channel) == 24 and len(extra_spatial-same_channel) == 72
    assert 2*d+len(local) == 98

    # Independently normalized segments require affine code rebasing to compose.
    scalar = [F(2), F(3), F(5), F(7)]
    joint_m1 = sum((F(r, 3)*value for r, value in enumerate(scalar)), F(0))
    reset_segments_m1 = scalar[1]+scalar[3]
    rebased_m1 = scalar[1]/3 + F(2, 3)*(scalar[2]+scalar[3]) + scalar[3]/3
    assert joint_m1 == rebased_m1 == F(34, 3)
    assert reset_segments_m1 != joint_m1

    return serial({
        "status": "new exact-arithmetic offset checks; no learning results",
        "arithmetic": "fractions.Fraction",
        "verifier_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_assertions_passed": True,
        "four_increment_example": {
            "clock_increment_each": F(1, 4), "spatial_increments": [[1,0,0,1], [0,1,1,0]],
            "ordinary_signature_level_one": ordinary_m,
            "ordinary_signature_level_two": ordinary_spread[1], "ordinary_clock_area": F(0),
            "coded_first_levels_in_order_T0_X0_T1_X1": spread_m,
            "same_channel_offset_areas": [spread_j[1][3], central_j[1][3]],
            "claim": "complete ordinary clock-augmented depth-two signatures and coded first levels coincide; offset depth-two signatures differ",
        },
        "higher_ordinary_depth": {
            "minimum_separating_depth_for_running_pair": 3,
            "space_time_space_coordinates": [t[1, 0, 1] for t in degree_three],
            "time_time_space_coordinates": [t[0, 0, 1] for t in degree_three],
            "log_space_time_space_coefficients": log_xtx,
            "uniform_grid_identity_cases": uniform_grid_cases,
            "general_depth_three_collision": {
                "spatial_increments": collision,
                "all_ordinary_coordinates_through_three_equal": True,
                "different_offset_areas": collision_areas,
            },
        },
        "extra_zero_basepoint_convention": {
            "increments": 5, "code": "r/4", "coded_first_levels": bp_m,
            "same_channel_offset_areas": [bp_spread_j[1][3], bp_central_j[1][3]],
            "claim": "same distinction with changed normalization; these are not the four-slot numerical values",
        },
        "general_signed_readout_checks": {
            "ordinary_projection": "passed",
            "same_channel_gap_log_readout": "passed",
            "symmetric_cross_channel_gap_log_readout": gap_symmetric,
            "directed_tensor_gap_readout": q_ce,
            "quadratic_moment_correction": moment_product/2,
            "signed_directed_difference_equals_moment_product": "passed",
            "midpoint_weighted_antisymmetric_log_readout": "passed",
        },
        "clock_moment_checks": clock_checks,
        "nominal_dimensions": dimensions,
        "control_dimensions": {"coded_moments": 50, "moments_plus_ordinary_local": 98,
                               "clock_rich_control": 171, "offset_local": 267,
                               "additional_weighted_spatial_areas": 96,
                               "same_original_channel": 24, "different_coordinates_same_landmark": 72},
        "normalized_segment_composition": {
            "joint_first_moment": joint_m1, "naive_reset_sum": reset_segments_m1,
            "affinely_rebased_sum": rebased_m1,
        },
    })


if __name__ == "__main__":
    result = verify()
    (ROOT / "verification_offset.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
