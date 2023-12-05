import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator, AutoMinorLocator
from fractions import Fraction as frac


def set_usetex(enable):
    """Enable/disable TeX-Engine for plotting

    Args:
        enable (boolean): when True, TeX-Engine is enabled
    """
    if enable:
        plt.rcParams.update(
            {"text.usetex": True,
             "font.family": "serif",
             "font.size": 11,
             "text.latex.preamble":
             r"\usepackage{mathpazo}\usepackage[locale=DE,"
             "per-mode=fraction,separate-uncertainty=true]{siunitx}"})
    else:
        plt.rcParams.update({"text.usetex": False})


def set_figsize(size):
    """Sets figure size to preset values.
    standard: (6.0, 4.0)
    notebooksize: (12.0, 8.0)

    Args:
        size (_type_): "standard" or "notebooksize"
    """
    if size == 'standard':
        plt.rcParams.update({"figure.figsize": [6.0, 4.0]})
    elif size == 'notebooksize':
        plt.rcParams.update({"figure.figsize": [12.0, 8.0]})
    else:
        raise ValueError


def rcdefaults():
    """Just a reminder function for the corresponding function in matplotlib.
    """
    plt.rcdefaults()


def set_schoolbookstyle():
    """Sets style to create schoolbooklike layout.
    """
    plt.rcParams.update({'axes.spines.right': False,
                         'axes.spines.top': False})


def sbplot(x, y, fig=None, ax=None, xlabel='x', ylabel='y', **plt_kwargs):
    """Wrapper around pyplot.plot with settings to produce
    schoolbook like axes.

    Args:
        x (numpy array): x values, forwarded to plot
        y (numpy array): y values, forwarded to plot
        fig (figure, optional): figure to use. Defaults to current figure.
        ax (axes, optional): axes to use. Defaults to current axes.
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    rc = {"xtick.direction": "inout", "ytick.direction": "inout",
          "xtick.major.size": 5, "ytick.major.size": 5, 'grid.alpha': .4}
    with plt.rc_context(rc):
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlabel(xlabel, ha='left')
        ax.set_ylabel(ylabel, rotation=0, ha='left')
        xlabel_offset_transform = ax.get_yaxis_transform(
        ) + ScaledTranslation(0, -2/72, fig.dpi_scale_trans)
        ylabel_offset_transform = ax.get_xaxis_transform(
        ) + ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)
        ax.xaxis.set_label_coords(1, 0, transform=xlabel_offset_transform)
        ax.yaxis.set_label_coords(0, 1, transform=ylabel_offset_transform)
        ax.plot((1), (0), ls="", marker=">", ms=5, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot((0), (1), ls="", marker="^", ms=5, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False)
        ax.plot(x, y, **plt_kwargs)


def export_png(name, fig=None, scalefactor=1, transparent=False):
    """Saves a figure in png format.

    Args:
        name (String): file name
        fig (figure, optional): figure to save. Defaults to current figure.
        sizefactor (int, optional): scale factor. Defaults to 1.
        transparent (bool, optional): use transparent background.
        Defaults to False.
    """
    if fig is None:
        fig = plt.gcf()
    fig.set_size_inches(scalefactor*6.38, scalefactor*3.94)
    if not name.endswith('.png'):
        name = name+'.png'
    fig.savefig(name, dpi=300, bbox_inches='tight', transparent=transparent)


def _pi_axis_formatter(val, pos, denomlim=10, pi=r'\pi'):
    """
    format label properly
    for example: 0.6666 pi --> 2π/3
               : 0      pi --> 0
               : 0.50   pi --> π/2
    """
    minus = "-" if val < 0 else ""
    val = abs(val)
    ratio = frac(val/np.pi).limit_denominator(denomlim)
    n, d = ratio.numerator, ratio.denominator

    fmt2 = "%s" % d
    if n == 0:
        fmt1 = "0"
    elif n == 1:
        fmt1 = pi
    else:
        fmt1 = r"%s%s" % (n, pi)

    fmtstring = "$" + minus + \
        (fmt1 if d == 1 else r"\frac{%s}{%s}" % (fmt1, fmt2)) + "$"

    return fmtstring


def set_pi_axis_formatter(axis=None):
    """Writes tick labes in multiple of pi. Ticks must be set seperatly.

    Args:
        axis (axis): xaxis or yaxis, defaults to xaxis of current axes
    """
    if axis is None:
        axis = plt.gca().xaxis
    axis.set_major_formatter(FuncFormatter(_pi_axis_formatter))


def set_xticks(base=1, ax=None):
    """Convenience function: assignes a multiple locator to xaxis

    Args:
        ax (axes, optional): axes to use. Defaults to current axes.
        base (int, optional): base for the MultipleLocator. Defaults to 1.
    """
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(base=base))


def set_yticks(base=1, ax=None):
    """Convenience function: assignes a multiple locator to xaxis

    Args:
        ax (axes, optional): axes to use. Defaults to current axes.
        base (float, optional): base for the MultipleLocator. Defaults to 1.
    """
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(base=base))


def set_minor_xticks(num_parts=2, ax=None):
    """Sets minor ticks so that each major interval is divided in num_parts.

    Args:
        num_parts (int, optional): number of subintervals between major ticks.
            Defaults to 2.
        ax (axes, optional): axes to use. Defaults to current axes.
    """
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_minor_locator(AutoMinorLocator(num_parts))


def set_minor_yticks(num_parts=2, ax=None):
    """Sets minor ticks so that each major interval is divided in num_parts.

    Args:
        num_parts (int, optional): number of subintervals between major ticks.
            Defaults to 2.
        ax (axes, optional): axes to use. Defaults to current axes.
    """
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_parts))
