import sympyschool.stochjgst as st
import pytest
import sympy as sp


def test_binompdf():
    assert st.binompdf(n=50, p=0.25, k=10) == pytest.approx(0.0985184099394176)


def test_binomcdf():
    assert st.binomcdf(n=50, p=0.25, k=10) == pytest.approx(0.262202310189509)


def test_vierfeldertafel():
    result = st.vierfeldertafel(abq=0.14, aqb=0.56, bq=0.36,
                                normalized=True, print_latex=False)
    assert result[sp.symbols('ab')] == pytest.approx(0.08)
    assert result[sp.symbols('abq')] == pytest.approx(0.14)
    assert result[sp.symbols('aqb')] == pytest.approx(0.56)
    assert result[sp.symbols('aqbq')] == pytest.approx(0.22)
    assert result[sp.symbols('a')] == pytest.approx(0.22)
    assert result[sp.symbols('aq')] == pytest.approx(0.78)
    assert result[sp.symbols('b')] == pytest.approx(0.64)
    assert result[sp.symbols('bq')] == pytest.approx(0.36)

    result = st.vierfeldertafel(ab=504, aqbq=42, a=630, b=672,
                                normalized=False, print_latex=False)
    assert result[sp.symbols('ab')] == 504
    assert result[sp.symbols('a')] == 630
    assert result[sp.symbols('b')] == 672
    assert result[sp.symbols('abq')] == 126
    assert result[sp.symbols('aqb')] == 168
    assert result[sp.symbols('aqbq')] == 42
    assert result[sp.symbols('aq')] == 210
    assert result[sp.symbols('bq')] == 168
