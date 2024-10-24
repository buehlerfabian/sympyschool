import sympyschool.stochjgst as st
import pytest


def test_binompdf():
    assert st.binompdf(n=50, p=0.25, k=10) == pytest.approx(0.0985184099394176)


def test_binomcdf():
    assert st.binomcdf(n=50, p=0.25, k=10) == pytest.approx(0.262202310189509)
