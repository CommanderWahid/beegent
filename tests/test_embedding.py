#!/usr/bin/env python3
"""Vector helpers and the calibration arithmetic behind CATALOG_MIN_RELEVANCE."""

from beegent.embedding import band, dot, from_blob, intersect, normalise, to_blob


def test_a_vector_is_unit_length_and_round_trips():
    """Normalised at write time, so read-time similarity is a plain dot product."""
    v = normalise([3.0, 4.0])
    assert v == [0.6, 0.8]
    assert abs(dot(v, v) - 1.0) < 1e-9
    assert [round(x, 4) for x in from_blob(to_blob(v))] == [0.6, 0.8]


def test_an_all_zero_vector_does_not_divide_by_zero():
    assert normalise([0.0, 0.0]) == [0.0, 0.0]


def test_the_largest_gap_is_the_boundary():
    """Real measurement: two matches, two non-matches, one obvious separation."""
    assert band([0.446, 0.368, 0.172, 0.167]) == (0.172, 0.368)


def test_a_single_score_has_no_boundary():
    assert band([0.5]) is None
    assert band([]) is None


def test_two_queries_intersect_to_the_TIGHTER_band():
    """Both real queries must be satisfied, so the constraint is the narrower one."""
    assert intersect([(0.172, 0.368), (0.162, 0.509)]) == (0.172, 0.368)


def test_disjoint_bands_are_a_conflict_not_an_average():
    """Averaging would hide that no threshold works - the whole point of reporting it."""
    assert intersect([(0.10, 0.20), (0.40, 0.60)]) is None


def test_touching_bands_are_still_a_conflict():
    """low == high leaves no room to place a threshold strictly between."""
    assert intersect([(0.10, 0.30), (0.30, 0.50)]) is None


def test_a_query_with_no_gap_is_ignored_rather_than_fatal():
    assert intersect([None, (0.20, 0.40)]) == (0.20, 0.40)
    assert intersect([None]) is None
