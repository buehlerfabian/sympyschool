from sympy.stats import Binomial, density, P
from sympy import sympify
from sympy.core.relational import LessThan


def binompdf(n, p, k):
    """P(X=k) for a binomial distribution.

    Args:
        n (integer): chain length
        p (float): hit probability
        k (integer): number of hits

    Returns:
        float: P(X=k) for X=B_n;p
    """
    X = Binomial("X", n=n, p=p)
    return density(X)(k)


def binomcdf(n, p, k):
    """P(X<=k) for a binomial distribution.

    Args:
        n (integer): chain length
        p (float): hit probability
        k (integer): number of hits

    Returns:
        float: P(X<=k) for X=B_n;p
    """
    X = Binomial("X", n=n, p=p)
    return P(X <= k)


def binomP(n, p, expr):
    """P(expr) for a binomial distribution.

    Args:
        n (integer): chain length
        p (float): hit probability
        expr (string): i.e. "X==10" or "X<=5" or "(X>=5) & (X<=10)"

    Returns:
        float: P(expr) for X=B_n;p
    """
    term = sympify(expr)
    X = sympify("X")
    if isinstance(term, LessThan):
        subs_term = term.subs(X, Binomial("X", n, p))
        return P(subs_term)
    else:
        return False
