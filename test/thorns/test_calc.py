#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_equal
)

import marlib.thorns as th



def test_calc_firing_rate():

    trains = th.make_trains(
        [[0.1, 0.4],
         [0.4, 0.5, 0.6]],
        duration=1
    )

    rate = th.calc_firing_rate(
        trains
    )

    assert_equal(rate, 2.5)



def test_calc_ci():

    trains = th.make_trains(
        [[1,3],
         [1,2,3]]
    )


    ci = th.calc_ci(
        trains,
        normalize=False
    )

    assert ci == 4




def test_calc_sac():

    trains = th.make_trains(
        [[1,3],
         [1,2,3]]
    )

    sac, bin_edges = th.calc_sac(
        trains,
        coincidence_window=1,
        analysis_window=3,
        normalize=False
    )


    assert_array_almost_equal(
        bin_edges,
        [-2., -1.,  0.,  1.,  2.,  3.]
    )
    assert_array_equal(
        sac,
        [2, 2, 4, 2, 2]
    )


