#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test binary search.

"""

from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import assert_almost_equal

import thorns as th



def test_find_zero():


    def func(x, foo):
        y = 2*x
        return y


    x0 = th.util.find_zero(
        func=func,
        x1=-10,
        x2=10,
        kwargs={'foo': np.pi},
        xtol=1e-9
    )


    assert_almost_equal(x0, 0)
