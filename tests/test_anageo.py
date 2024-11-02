import sympyschool.anageo as ag
import sympy as sp
import pytest


def test_vvv():
    vec = ag.vvv(1, 2, 3)
    assert isinstance(vec, sp.matrices.dense.MutableDenseMatrix)
    assert len(vec) == 3
    assert list(vec) == [1, 2, 3]


def test_vv():
    vec = ag.vv(1, 2)
    assert isinstance(vec, sp.matrices.dense.MutableDenseMatrix)
    assert len(vec) == 2
    assert list(vec) == [1, 2]


def test_angle():
    v1 = ag.vvv(1, 2, 3)
    v2 = ag.vvv(2, -1, 0)
    assert ag.angle(v1, v2) == 90
    assert ag.angle(v1, v2, unit="deg") == 90
    assert ag.angle(v1, v2, unit="rad") == sp.pi/2
    assert ag.angle(v1, v2, "rad") == sp.pi/2

    v1 = ag.vvv(1, 2, 3)
    v2 = ag.vvv(2, 4, 6)
    assert ag.angle(v1, v2) == 0
    assert ag.angle(v1, v2, unit="deg") == 0
    assert ag.angle(v1, v2, unit="rad") == 0
    assert ag.angle(v1, v2, "rad") == 0

    v1 = ag.vvv(1, 0, 0)
    v2 = ag.vvv(1, 1, 0)
    assert ag.angle(v1, v2) == 45
    assert ag.angle(v1, v2, unit="deg") == 45
    assert ag.angle(v1, v2, unit="rad") == sp.pi/4
    assert ag.angle(v1, v2, "rad") == sp.pi/4

    v1 = ag.vvv(1, 2, 3)
    v2 = ag.vvv(3, 2, 1)
    assert sp.simplify(ag.angle(v1, v2, "rad") -
                       sp.acos(sp.Rational(5, 7))) == 0

    a = sp.symbols('a', positive=True)
    v1 = ag.vvv(1, 1, a)
    v2 = ag.vvv(1, 1, 1)
    assert sp.simplify(ag.angle(v1, v2) -
                       (180*sp.acos(
                           sp.sqrt(3)*(a + 2) /
                           (3*sp.sqrt(a**2 + 2)))/sp.pi)) == 0

    try:
        ag.angle(v1, v2, "bla")
        assert False
    except ValueError:
        assert True


def test_is_perpendicular():
    v1 = ag.vvv(1, 2, 3)
    v2 = ag.vvv(2, -1, 0)
    assert ag.is_perpendicular(v1, v2)

    v1 = ag.vvv(1, 2, 3)
    v2 = ag.vvv(2, -1, 1)
    assert not ag.is_perpendicular(v1, v2)

    v1 = ag.vv(1, 2)
    v2 = ag.vv(2, -1)
    assert ag.is_perpendicular(v1, v2)

    v1 = ag.vv(1, 2)
    v2 = ag.vv(2, -2)
    assert not ag.is_perpendicular(v1, v2)

    a = sp.symbols('a')
    v1 = ag.vvv(1, a, 2)
    v2 = ag.vvv(1, 1, 1)
    assert sp.simplify(ag.is_perpendicular(v1, v2)) == sp.Eq(a, -3)

    try:
        v1 = ag.vv(1, 2)
        v2 = ag.vvv(2, -2, -1)
        ag.is_perpendicular(v1, v2)
        assert False
    except ValueError:
        assert True

    try:
        v1 = 5
        v2 = ag.vvv(2, -2, -1)
        ag.is_perpendicular(v1, v2)
        assert False
    except ValueError:
        assert True

    try:
        v1 = ag.vvv(2, -2, -1)
        v2 = 5
        ag.is_perpendicular(v1, v2)
        assert False
    except ValueError:
        assert True


def test_relative_length():
    v1 = ag.vvv(1, 2, 3)
    v2 = ag.vvv(2, 4, 6)
    assert ag.relative_length(v1, v2) == sp.Rational(1, 2)

    v1 = ag.vvv(sp.pi, 2*sp.pi, 3*sp.pi)
    v2 = ag.vvv(1, 2, 3)
    assert ag.relative_length(v1, v2) == sp.pi

    a = sp.symbols('a')
    v1 = ag.vvv(1, 2, a)
    v2 = ag.vvv(2, 4, 2*a)
    try:
        ag.relative_length(v1, v2)
        assert False
    except ValueError:
        assert True


def test_line_init():
    p = ag.vvv(1, 2, 3)
    u = ag.vvv(4, 5, 6)
    line = ag.Line(p, u)
    assert line.p == p
    assert line.u == u

    with pytest.raises(ValueError):
        ag.Line(ag.vvv(1, 2, 3), ag.vvv(0, 0, 0))


def test_fromCoordinateEq():
    e1 = ag.Plane.fromCoordinateEq(1, 1, 0, 4)
    assert e1.n == ag.vvv(1, 1, 0)
    assert (e1.n).dot(e1.p - ag.vvv(4, 0, 0)) == 0

    e2 = ag.Plane.fromCoordinateEq(1, 1, 2, 4)
    assert e2.n == ag.vvv(1, 1, 2)
    assert (e2.n).dot(e2.p - ag.vvv(4, 0, 0)) == 0


def test_orientation_relative_to_line():
    # Identical lines
    l1 = ag.Line(ag.vvv(1, 2, 3), ag.vvv(4, 5, 6))
    l2 = ag.Line(ag.vvv(5, 7, 9), ag.vvv(4, 5, 6))
    assert l1.orientation_relative_to_line(l2) == "The lines are identical."

    # Parallel lines
    l1 = ag.Line(ag.vvv(1, 2, 3), ag.vvv(4, 5, 6))
    l2 = ag.Line(ag.vvv(2, 4, 6), ag.vvv(8, 10, 12))
    assert l1.orientation_relative_to_line(l2) == "The lines are parallel."

    # Intersecting lines
    l1 = ag.Line(ag.vvv(1, 2, 3), ag.vvv(4, 5, 6))
    l2 = ag.Line(ag.vvv(1, 2, 3), ag.vvv(-6, 5, 4))
    assert l1.orientation_relative_to_line(
        l2) == "The lines intersect at (1|2|3)."

    # Skew lines
    l1 = ag.Line(ag.vvv(1, 2, 3), ag.vvv(4, 5, 6))
    l2 = ag.Line(ag.vvv(7, 8, 9), ag.vvv(-6, 5, 4))
    assert l1.orientation_relative_to_line(
        l2) == "The lines are skew (neither parallel nor intersecting)."
