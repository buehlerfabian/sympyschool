import sympyschool.anageo as ag
import sympy as sp
import unittest


class Test_Anageo_Basic(unittest.TestCase):
    def test_vvv(self):
        vec = ag.vvv(1, 2, 3)
        self.assertIsInstance(vec, sp.matrices.dense.MutableDenseMatrix)
        self.assertEqual(len(vec), 3)
        self.assertListEqual(list(vec), [1, 2, 3])

    def test_vv(self):
        vec = ag.vv(1, 2)
        self.assertIsInstance(vec, sp.matrices.dense.MutableDenseMatrix)
        self.assertEqual(len(vec), 2)
        self.assertListEqual(list(vec), [1, 2])

    def test_angle(self):
        v1 = ag.vvv(1, 2, 3)
        v2 = ag.vvv(2, -1, 0)
        self.assertEqual(ag.angle(v1, v2), 90)
        self.assertEqual(ag.angle(v1, v2, unit="deg"), 90)
        self.assertEqual(ag.angle(v1, v2, unit="rad"), sp.pi/2)
        self.assertEqual(ag.angle(v1, v2, "rad"), sp.pi/2)

        v1 = ag.vvv(1, 2, 3)
        v2 = ag.vvv(2, 4, 6)
        self.assertEqual(ag.angle(v1, v2), 0)
        self.assertEqual(ag.angle(v1, v2, unit="deg"), 0)
        self.assertEqual(ag.angle(v1, v2, unit="rad"), 0)
        self.assertEqual(ag.angle(v1, v2, "rad"), 0)

        v1 = ag.vvv(1, 0, 0)
        v2 = ag.vvv(1, 1, 0)
        self.assertEqual(ag.angle(v1, v2), 45)
        self.assertEqual(ag.angle(v1, v2, unit="deg"), 45)
        self.assertEqual(ag.angle(v1, v2, unit="rad"), sp.pi/4)
        self.assertEqual(ag.angle(v1, v2, "rad"), sp.pi/4)

        v1 = ag.vvv(1, 2, 3)
        v2 = ag.vvv(3, 2, 1)
        self.assertEqual(sp.simplify(ag.angle(v1, v2, "rad") -
                                     sp.acos(sp.Rational(5, 7))), 0)

        a = sp.symbols('a', positive=True)
        v1 = ag.vvv(1, 1, a)
        v2 = ag.vvv(1, 1, 1)
        self.assertEqual(sp.simplify(ag.angle(v1, v2) -
                                     (180*sp.acos(
                                         sp.sqrt(3)*(a + 2) /
                                         (3*sp.sqrt(a**2 + 2)))/sp.pi)), 0)

        with self.assertRaises(ValueError):
            ag.angle(v1, v2, "bla")

    def test_is_perpendicular(self):
        v1 = ag.vvv(1, 2, 3)
        v2 = ag.vvv(2, -1, 0)
        self.assertTrue(ag.is_perpendicular(v1, v2))

        v1 = ag.vvv(1, 2, 3)
        v2 = ag.vvv(2, -1, 1)
        self.assertFalse(ag.is_perpendicular(v1, v2))

        v1 = ag.vv(1, 2)
        v2 = ag.vv(2, -1)
        self.assertTrue(ag.is_perpendicular(v1, v2))

        v1 = ag.vv(1, 2)
        v2 = ag.vv(2, -2)
        self.assertFalse(ag.is_perpendicular(v1, v2))

        a = sp.symbols('a')
        v1 = ag.vvv(1, a, 2)
        v2 = ag.vvv(1, 1, 1)
        self.assertEqual(sp.simplify(ag.is_perpendicular(v1, v2)),
                         sp.Eq(a, -3))

        with self.assertRaises(ValueError):
            v1 = ag.vv(1, 2)
            v2 = ag.vvv(2, -2, -1)
            ag.is_perpendicular(v1, v2)

        with self.assertRaises(ValueError):
            v1 = 5
            v2 = ag.vvv(2, -2, -1)
            ag.is_perpendicular(v1, v2)

        with self.assertRaises(ValueError):
            v1 = ag.vvv(2, -2, -1)
            v2 = 5
            ag.is_perpendicular(v1, v2)

    def test_relative_length(self):
        v1 = ag.vvv(1, 2, 3)
        v2 = ag.vvv(2, 4, 6)
        self.assertEqual(ag.relative_length(v1, v2),
                         sp.Rational(1, 2))

        v1 = ag.vvv(sp.pi, 2*sp.pi, 3*sp.pi)
        v2 = ag.vvv(1, 2, 3)
        self.assertEqual(ag.relative_length(v1, v2),
                         sp.pi)

        a = sp.symbols('a')
        v1 = ag.vvv(1, 2, a)
        v2 = ag.vvv(2, 4, 2*a)
        with self.assertRaises(ValueError):
            ag.relative_length(v1, v2)


if __name__ == '__main__':
    unittest.main()
