from sympy.stats import Binomial, density, P
import sympy as sp
from IPython.display import display, Math


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


def vierfeldertafel(ab=None, abq=None, a=None,
                    aqb=None, aqbq=None, aq=None,
                    b=None, bq=None,
                    normalized=True, print_latex=True):
    """
    Constructs a contingency table (Vierfeldertafel) and optionally prints
    it in LaTeX format.
    Parameters:
    ab (float, optional): Value for the cell representing both A and B.
    abq (float, optional): Value for the cell representing A and not B.
    a (float, optional): Value for the total of A.
    aqb (float, optional): Value for the cell representing not A and B.
    aqbq (float, optional): Value for the cell representing neither A nor B.
    aq (float, optional): Value for the total of not A.
    b (float, optional): Value for the total of B.
    bq (float, optional): Value for the total of not B.
    print_latex (bool, optional): If True, prints the LaTeX representation of
        the table. Defaults to True.
    Returns:
    dict: A dictionary of solved values if print_latex is False.
    """
    sa, sb, saq, sbq, sab, sabq, saqb, saqbq = (
        sp.symbols('a b aq bq ab abq aqb aqbq'))
    eqs = [sab+sabq-sa, saqb+saqbq-saq, sab+saqb -
           sb, sabq+saqbq-sbq]
    if normalized:
        eqs.append(sa+saq-1)
        eqs.append(sb+sbq-1)

    params = {'a': a, 'b': b, 'aq': aq, 'bq': bq, 'ab': ab,
              'abq': abq, 'aqb': aqb, 'aqbq': aqbq}
    for symbol, value in params.items():
        if value is not None:
            eqs.append(sp.symbols(symbol) - value)
    s = sp.solve(eqs, [sab, sabq, sa, saqb, saqbq, saq, sb, sbq])

    # Extract the values from the dictionary `s`
    values = [
        [s[sab], s[sabq], s[sa]],
        [s[saqb], s[saqbq], s[saq]],
        [s[sb], s[sbq], s[sa]+s[saq]]
    ]

    # Create the LaTeX table
    table = r"\begin{array}{|c|c|c|}\hline"
    table += rf"{values[0][0]} & {values[0][1]} & {values[0][2]} \\\\ \hline"
    table += rf"{values[1][0]} & {values[1][1]} & {values[1][2]} \\\\ \hline"
    table += rf"{values[2][0]} & {values[2][1]} & {values[2][2]} \\\\ \hline"
    table += r"\end{array}"

    # Print the LaTeX table
    if print_latex:
        display(Math(table))
    else:
        return s


class Hypothesentest_rechts():
    """
    A class to perform a right-tailed hypothesis test for a binomial
    distribution.
    Attributes
    ----------
    n : int
        The number of trials in the binomial distribution.
    p0 : float
        The null hypothesis probability of success.
    alpha : float
        The significance level of the test.
    Methods
    -------
    get_ablehnungsbereich():
        Calculates and returns the critical value (rejection region) for the
        test.
    get_irrtumswahrscheinlichkeit():
        Calculates and returns the type I error probability (alpha) for the
        test.
    get_fehler2(p):
        Calculates and returns the type II error probability (beta) for a
        given alternative hypothesis probability p.
    """

    def __init__(self, n, p0, alpha):
        self.n = n
        self.p0 = p0
        self.alpha = alpha
        self._grenz_k = None

    def get_ablehnungsbereich(self):
        if self._grenz_k is None:
            X = Binomial("X", n=self.n, p=self.p0)
            # lambdify P(X>=k) to avoid sympy's slow evaluation
            k = sp.symbols('k')
            pf = sp.lambdify(k, P(X >= k))

            for m in range(self.n+1):
                if pf(m) <= self.alpha:
                    self._grenz_k = m
                    break
        return self._grenz_k

    def get_irrtumswahrscheinlichkeit(self):
        X = Binomial("X", n=self.n, p=self.p0)
        return P(X >= self.get_ablehnungsbereich())

    def get_fehler2(self, p):
        X = Binomial("X", n=self.n, p=p)
        return P(X < self.get_ablehnungsbereich())


class Hypothesentest_links():
    """
    A class to perform a left-tailed hypothesis test for a binomial
    distribution.
    Attributes
    ----------
    n : int
        The number of trials in the binomial distribution.
    p0 : float
        The null hypothesis probability of success.
    alpha : float
        The significance level of the test.

    Methods
    -------
    get_ablehnungsbereich():
        Calculates and returns the critical value for the rejection region.
    get_irrtumswahrscheinlichkeit():
        Calculates and returns the Type I error probability (alpha).
    get_fehler2(p):
        Calculates and returns the Type II error probability for a given
        alternative hypothesis probability p.
    """

    def __init__(self, n, p0, alpha):
        self.n = n
        self.p0 = p0
        self.alpha = alpha
        self._grenz_k = None

    def get_ablehnungsbereich(self):
        if self._grenz_k is None:
            X = Binomial("X", n=self.n, p=self.p0)
            # lambdify P(X<=k) to avoid sympy's slow evaluation
            k = sp.symbols('k')
            pf = sp.lambdify(k, P(X <= k))

            for m in range(self.n+1):
                if pf(m) > self.alpha:
                    self._grenz_k = m-1
                    break
        return self._grenz_k

    def get_irrtumswahrscheinlichkeit(self):
        X = Binomial("X", n=self.n, p=self.p0)
        return P(X <= self.get_ablehnungsbereich())

    def get_fehler2(self, p):
        X = Binomial("X", n=self.n, p=p)
        return P(X > self.get_ablehnungsbereich())


class Hypothesentest_beidseitig():
    """
    A class to perform a two-sided hypothesis test.
    Attributes:
    -----------
    n : int
        The number of trials.
    p0 : float
        The null hypothesis probability.
    alpha : float
        The significance level.
    Methods:
    --------
    get_ablehnungsbereich_links():
        Returns the rejection region for the left-sided test.
    get_ablehnungsbereich_rechts():
        Returns the rejection region for the right-sided test.
    get_irrtumswahrscheinlichkeit():
        Returns the probability of a type I error (alpha).
    get_fehler2(p):
        Returns the probability of a type II error (beta) for a given
        alternative hypothesis probability p.
    """

    def __init__(self, n, p0, alpha):
        self.n = n
        self.p0 = p0
        self.alpha = alpha
        self._h_links = Hypothesentest_links(n, p0, alpha/2)
        self._h_rechts = Hypothesentest_rechts(n, p0, alpha/2)

    def get_ablehnungsbereich_links(self):
        return self._h_links.get_ablehnungsbereich()

    def get_ablehnungsbereich_rechts(self):
        return self._h_rechts.get_ablehnungsbereich()

    def get_irrtumswahrscheinlichkeit(self):
        X = Binomial("X", n=self.n, p=self.p0)
        return (P(X >= self.get_ablehnungsbereich_rechts()) +
                P(X <= self.get_ablehnungsbereich_links()))

    def get_fehler2(self, p):
        X = Binomial("X", n=self.n, p=p)
        return (P(sp.And(X > self.get_ablehnungsbereich_links(),
                         X < self.get_ablehnungsbereich_rechts())))
