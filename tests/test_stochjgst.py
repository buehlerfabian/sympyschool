import sympyschool.stochjgst as st
import unittest


class Test_Stochjgst(unittest.TestCase):
    def test_binompdf(self):
        self.assertAlmostEqual(
            st.binompdf(n=50, p=0.25, k=10), 0.0985184099394176)

    def test_binomcdf(self):
        self.assertAlmostEqual(
            st.binomcdf(n=50, p=0.25, k=10), 0.262202310189509)


if __name__ == '__main__':
    unittest.main()
