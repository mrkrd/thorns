#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal
)

import marlib.thorns as th

def test_calc_sac():

    trains = th.make_trains(
        [[1e-3, 2e-3, 3e-3],
        [1e-3, 2.01e-3, 2.5e-3]]
    )

    sac, t = th.calc_sac(
        trains,
        coincidence_window=1e-3,
        analysis_window=2e-3
    )


    print t
    print sac
    assert_array_almost_equal(
        t,
        [-1.2e-3, -0.4e-3,  0.4e-3,  1.2e-3,  2e-3 ]
    )

    assert_array_equal(
        sac,
        [ 0.11111111,  0.55555556,  0.44444444,  0.55555556,  0.11111111]
    )