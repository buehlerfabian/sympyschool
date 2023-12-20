from sympy.stats import Binomial, density, P


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
