#!/usr/bin/env python

"""Binary search for threshold finding.

"""

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"

import numpy as np

def find_zero(func, x1, x2, xtol=None, kwargs=None):
    """Find a zero crossing of a function using binary search.

    Parameters
    ----------
    func : function
        Function must be increasing and have one zero crossing.
    x1, x2 : float
        Lower and upper range.  Requirements (func(x1) < 0) and
        (func(x2) > 0).
    xtol : float, optional
        Range (x2-x1) at which the search stops.  If not specified,
        then 1e-3-th of the difference between x2 and x1 will be
        taken.
    kwargs : dict, optional
        Extra keyword arguments for the function.

    Returns
    -------
    float
        Function argument `x0` for which `func(x0) ~ 0`.

    """

    assert x1 < x2

    if kwargs is None:
        kwargs = {}

    if xtol is None:
        xtol = 1e-3 * np.abs(x2 - x1)

    if func(x1, **kwargs) > 0:
        return np.nan

    if func(x2, **kwargs) < 0:
        return np.nan

    while x2 - x1 > xtol:
        x = (x1 + x2) / 2
        y = func(x, **kwargs)

        if y > 0:
            x2 = x
        elif y < 0:
            x1 = x
        else:
            x1 = x2 = x

    x = (x1 + x2) / 2

    return x
