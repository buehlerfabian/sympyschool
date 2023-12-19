# import numpy as np
import sympy as sp


def vvv(x1, x2, x3):
    """Returns a 3-dimensional vector.

    Args:
        x1 (float): x1 component
        x2 (float): x2 component
        x3 (float): x3 component

    Returns:
        vector: vector (x1,x2,x3)
    """
    return sp.Matrix([x1, x2, x3])


def vv(x1, x2):
    """Returns a 2-dimensional vector.

    Args:
        x1 (float): x1 component
        x2 (float): x2 component

    Returns:
        vector: vector (x1,x2)
    """
    return sp.Matrix([x1, x2])


def angle(v1, v2, unit="deg"):
    """Returns the angle between 2 vectors.

    Args:
        v1 (vector): first vector
        v2 (vector): second vector
        unit (str, optional): Determines the measuring unit for the
        return value.
            "deg" for degree (0° to 360°), "rad" for radian (0 to 2*pi).
            Defaults to "deg".

    Raises:
        ValueError: if unit is not "deg" or "rad"

    Returns:
        float: value for the angle
    """
    if unit not in ["deg", "rad"]:
        raise ValueError(
            "Unrecognized units {}; expected deg or rad".format(unit))
    if unit == "deg":
        return sp.acos(v1.normalized().dot(v2.normalized())) * 360 / (2*sp.pi)
    else:
        return sp.acos(v1.normalized().dot(v2.normalized()))


def is_perpendicular(v1, v2):
    """Checks if two vectors are perpendicular.

    Args:
        v1 (vector): first vector
        v2 (vector): second vector

    Returns:
        boolean: True if v1 and v2 are perpendicular, False otherwise
        equation: if result depends on condition
    """
    if not is_vector(v1) or not is_vector(v2):
        raise ValueError("Arguments must be vectors.")
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same.")
    return sp.simplify(sp.Eq(v1.dot(v2), 0))


def is_parallel(v1, v2):
    """Checks if two vectors are parallel.

    Args:
        v1 (vector): first vector
        v2 (vector): second vector

    Returns:
        boolean: True, if vectors are parallel;
                 False, otherwise
        equation: if parallelity depends on conditions
    """
    if len(v1) != len(v2):
        raise ValueError('vectors do not have the same dimensions')
    elif len(v1) == 3:
        eqlist = list(set([sp.simplify(sp.Eq(v1.cross(v2)[i], 0))
                      for i in range(len(v1))]))
        if eqlist == [True]:
            return True
        elif False in eqlist:
            return False
        else:
            if True in eqlist:
                eqlist.remove(True)
            return eqlist
    elif len(v1) == 2:
        return sp.simplify(sp.Eq(v1[0]*v2[1]-v1[1]*v2[0], 0))
    else:
        raise ValueError('vector dimension must be 2 or 3')


def relative_length(v1, v2):
    """Returns the length of the first vector relative to the second vector,
    if both are parallel to each other.
    If they are not parallel, false is returned.

    Args:
        v1 (vector): first vector
        v2 (vector): second vector

    Returns:
        number: length of v1 relative to the length of v2
    """
    # TODO raise error if v1, v2 not numeric
    if (not is_parallel(v1, v2)) or v2.norm() == 0:
        return False
    else:
        return v1.norm()/v2.norm()


def is_vector(o):
    """Returns true if object is a vector

    Returns:
        boolean: True if parameter is a vector
    """
    if isinstance(o, sp.matrices.dense.MutableDenseMatrix):
        return True
    else:
        return False


def is_line(o):
    """Checks if given parameter is a line.

    Returns:
        boolean: True if parameter is a line
    """
    if isinstance(o, Line):
        return True
    else:
        return False


def is_plane(o):
    """Checks if given parameter is a plane.

    Returns:
        boolean: True if parameter is a plane
    """
    if isinstance(o, Plane):
        return True
    else:
        return False


class Line:
    def __init__(self, p, u):
        """Creates a line object with position vector p and direction vector u

        Args:
            p (vector): position vector
            u (vector): direction vector
        """
        self.p = p
        self.u = u

    def _repr_latex_(self):
        """LaTeX representation

        Returns:
            string: LaTeX string
        """
        return self.p._repr_latex_()[:-1]+'+t'+self.u._repr_latex_()[1:]

    def get_point(self, t):
        """Returns the position vector to the point corresponding to the
        given parameter value

        Args:
            t (float): parameter value

        Returns:
            vector: position vector
        """
        return self.p+t*self.u

    def is_element(self, p):
        """Checks if point is on the line

        Args:
            p (vector): point

        Returns:
            boolean: true if p is on the line
        """
        return is_parallel(self.u, self.p - p)

    def is_parallel(self, e):
        """Checks if line is parallel to line given in argument

        Args:
            e (line): line

        Returns:
            boolean: true if parallel
        """
        return is_parallel(self.u, e.u)

    def is_identical(self, g2):
        """Checks if line is identical to line given in argument

        Args:
            g2 (line): line

        Returns:
            boolean: true if identical
        """
        eqlist = []
        if type(self.is_parallel(g2)) == list:
            eqlist = [*eqlist, *self.is_parallel(g2)]
        else:
            eqlist.append(self.is_parallel(g2))
        if type(self.is_element(g2.p)) == list:
            eqlist = [*eqlist, *self.is_element(g2.p)]
        else:
            eqlist.append(self.is_element(g2.p))
        eqlist = list(set(eqlist))
        if eqlist == [True]:
            return True
        elif False in eqlist:
            return False
        else:
            if True in eqlist:
                eqlist = [x for x in eqlist if x is not True]
            return eqlist

    def intersection(self, e):
        """Checks if there is an intersection; if true, returns
        intersetion point

        Args:
            e (line): second line

        Returns:
            boolean or vector: False if no intersection,
            otherwise intersection point
        """
        sol = sp.linsolve(
            (sp.Matrix([list(self.u), list(e.u)]).transpose(), -self.p + e.p))
        sol = list(sol)
        if len(sol) > 0:
            return self.get_point(sol[0][0])
        else:
            return False

    def distance_from_point(self, p):
        """Calculates the distance between the line and a point.

        Args:
            p (vector): point

        Returns:
            float: distance
        """
        if not is_vector(p):
            raise TypeError("Parameter must be a point (vector)")
        return (self.p - p).cross(self.u).norm()/self.u.norm()

    def distance_from_line(self, l2):
        """Calculates the distance between the line and another line.

        Args:
            l2 (line): the other line

        Returns:
            float: distance
        """
        if not is_line(l2):
            raise TypeError("Parameter must be a line")
        if not is_parallel(self.u, l2.u):
            n0 = self.u.cross(l2.u)
            n0 = n0/n0.norm()
            return sp.Abs((self.p - l2.p).dot(n0))
        else:
            return self.distance_from_point(l2.p)

    def perpendicular_basepoint(self, o):
        """Calculates the point on the line that is closest to p, i.e.
        the vector connecting both points is perpendicular to the line.

        Args:
            o (point or line): point or line

        Returns:
            point / list of points: point on line closest to p /
            endpoints of shortest segment between both lines
        """
        if (not is_vector(o)) and (not is_line(o)):
            raise TypeError("Parameter must be a point or a line")

        if is_vector(o):
            h = Plane(o, self.u)
            return h.intersection(self)
        else:
            assert is_line(o)
            if self.is_parallel(o):
                raise ValueError("Lines must not be parallel.")
            s, t = sp.symbols('s t')
            p1 = self.get_point(t)
            p2 = o.get_point(s)
            eq1 = (p2-p1).dot(self.u)
            eq2 = (p2-p1).dot(o.u)
            sol = sp.solve([eq1, eq2], [s, t])
            return [p1.subs(sol), p2.subs(sol)]


class Plane:
    def __init__(self, p, n):
        """Creates a plane with given location vector and normal vector

        Args:
            p (vector): location vector
            n (vector): normal vector
        """
        self.p = p
        if n[0].is_integer and n[1].is_integer and n[2].is_integer:
            gcd = sp.gcd(sp.gcd(n[0], n[1]), n[2])
            n = n / gcd
        self.n = n

    @classmethod
    def fromParametricEq(cls, s, u, v):
        """Creates a plane with given location vector and 2 'spannvektoren'

        Args:
            s (vector): location vector
            u (vector): 1. Spannvektor
            v (vector): 2. Spannvektor

        Returns:
            Plane: Plane object
        """
        n = u.cross(v)
        p = s
        return cls(p, n)

    @classmethod
    def fromCoordinateEq(cls, a1, a2, a3, c):
        """Creates a plane from coordinate form a1*x1 + a2*x2 + a3*x3 = c

        Args:
            a1 (float or symbol): n1 (first normal vector component)
            a2 (float or symbol): n2 (second normal vector component)
            a3 (float or symbol): n3 (third normal vector component)
            c (float or symbol): constant

        Returns:
            Plane: Plane object
        """
        if not sp.sympify(a1).is_zero:
            p = vvv(sp.Rational(c, a1), 0, 0)
        elif not sp.sympify(a2).is_zero:
            p = vvv(0, sp.Rational(c, a2), 0)
        else:
            p = vvv(0, 0, sp.Rational(c, a3))

        return cls(p, vvv(a1, a2, a3))

    def getCoordinateEq(self, x1=None, x2=None, x3=None):
        """Returns the equation of the plane.

        Args:
            x1 (symbol, optional): symbol to use for x1-axis. Defaults to x1.
            x2 (symbol, optional): symbol to use for x2-axis. Defaults to x2.
            x3 (symbol, optional): symbol to use for x3-axis. Defaults to x3.

        Returns:
            sympy equation: plane equation
        """
        c = self.n.dot(self.p)
        if (x1 is None):
            x1 = sp.symbols('x1')
        if (x2 is None):
            x2 = sp.symbols('x2')
        if (x3 is None):
            x3 = sp.symbols('x3')
        return sp.Eq(self.n[0]*x1+self.n[1]*x2+self.n[2]*x3, c)

    def _repr_latex_(self):
        return self.getCoordinateEq()._repr_latex_()

    def is_element(self, o):
        """Checks if given object (point or line) is an element of the plane.

        Args:
            o (vector or Line): location vector of a point or line object

        Raises:
        ValueError: if o is not a vector or a line object

        Returns:
            bool or equation or list of equations: true if o is element
            of self or condition in form of equation(s)
        """
        if is_vector(o):
            return sp.Eq((o-self.p).dot(self.n), 0)
        elif is_line(o):
            # return [i for i in self.is_element(o.p)].append(
            # sp.Eq(o.u.dot(self.n),0))
            ret = [self.is_element(o.p), sp.Eq(o.u.dot(self.n), 0)]
            if False in ret:
                return False
            if True in ret:
                ret.remove(True)
            if len(ret) == 0:
                ret = True
            if len(ret) == 1:
                ret = ret[0]
            return ret
        else:
            raise TypeError("Parameter must be a vector(point) or a line")

    def is_identical(self, p):
        """Checks if plane is identical to another plane

        Args:
            p (Plane): Plane to test for identity

        Returns:
            boolean: True if both planes are identical
        """
        if not is_plane(p):
            raise TypeError("Parameter must be a plane.")
        if is_parallel(self.n, p.n):
            k = relative_length(self.n, p.n)
            return sp.simplify(sp.Eq(self.n.dot(self.p), k*p.n.dot(p.p)))
        else:
            return False

    def intersection(self, o):
        """Checks for intersections with a line or another plane.

        Args:
            o (Line or Plane): Line or Plane to check for intersections

        Raises:
            TypeError: _description_

        Returns:
            boolean: True if identical
            Point: intersection point with line
            Line: intersection line with another plane
        """
        if not is_line(o) and not is_plane(o):
            raise TypeError("Parameter must be a line or a plane")
        if is_line(o):
            t = sp.symbols('t')
            result = sp.simplify(self.is_element((o.p + t*o.u)))
            if t in result.free_symbols:    # does result contain symbol t?
                tsol = sp.solve(result, t)
                return o.get_point(tsol[0])
            else:
                return result
        if is_plane(o):
            if is_parallel(self.n, o.n):
                k = relative_length(self.n, o.n)
                return sp.simplify(sp.Eq(self.n.dot(self.p), k*o.n.dot(o.p)))
            else:
                x1, x2, x3 = sp.symbols('x1 x2 x3')
                excl_params = self.getCoordinateEq().free_symbols.union(
                    o.getCoordinateEq().free_symbols) - {x1, x2, x3}
                sol = sp.solve(
                    [self.getCoordinateEq(), o.getCoordinateEq()],
                    exclude=excl_params)
                freevar_set = set.intersection(x1.subs(sol).free_symbols,
                                               x2.subs(sol).free_symbols,
                                               x3.subs(sol).free_symbols)
                if len(freevar_set) == 0:
                    # no free variable in the equations, that means one of
                    # the variables
                    # is completely missing
                    # and therefore free
                    freevar_set = {x1, x2, x3} - set(sol.keys())
                freevar = next(iter(freevar_set))
                p = vvv(x1.subs(sol).subs(freevar, 0),
                        x2.subs(sol).subs(freevar, 0),
                        x3.subs(sol).subs(freevar, 0))
                u = vvv(x1.subs(sol).subs(freevar, 1),
                        x2.subs(sol).subs(freevar, 1),
                        x3.subs(sol).subs(freevar, 1)) - p
                return Line(p, u)

    def perpendicular_basepoint(self, p):
        """Calculates the point on the plane that is closest to p, i.e.
        the vector
        connecting both points is perpendicular to the plane.

        Args:
            p (point): point

        Returns:
            point: point on plane that is closest to p+9
        """
        if not is_vector(p):
            raise TypeError("Parameter must be a vector (point).")
        h = Line(p, self.n)
        return self.intersection(h)

    def distance_from_point(self, p):
        """Returns the distance between plane and point p.

        Args:
            p (point): point

        Returns:
            float: distance between plane and point p
        """
        return sp.Abs(self.n.dot(p - self.p)/self.n.norm())

    def distance_from_line(self, g):
        """Returns the distance between plane and line l.

        Args:
            l (line): line

        Returns:
            float: distance between plane and line l
        """
        if not is_line(g):
            raise TypeError("Parameter l must be a line.")
        if self.n.dot(g.u) == 0:
            return self.distance_from_point(g.p)
        else:
            return 0

    def distance_from_plane(self, p):
        """Returns the distance between self and other plane.

        Args:
            p (Plane): plane to which distance is to be calculated

        Raises:
            TypeError: if p is not a plane

        Returns:
            float: distance between self and other plane
        """
        if not is_plane(p):
            raise TypeError("Parameter p must be a plane.")
        if is_parallel(self.n, p.n):
            return self.distance_from_point(p.p)
        else:
            return 0
