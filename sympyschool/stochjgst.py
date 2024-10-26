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
