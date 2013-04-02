#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_equal,
    assert_almost_equal
)

import elmar.thorns as th



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


    assert_array_equal(
        bin_edges,
        [-2., -1.,  0.,  1.,  2.,  3.]
    )
    assert_array_equal(
        sac,
        [2, 2, 4, 2, 2]
    )




def test_calc_psth():

    trains = th.make_trains(
        [[0.5, 1.5, 2.5],
         [0.5, 2.5]]
    )

    psth, bin_edges = th.calc_psth(
        trains,
        bin_size=1,
        normalize=False
    )


    assert_array_equal(
        psth,
        [2, 1, 2]
    )
    assert_array_equal(
        bin_edges,
        [0, 1, 2, 3]
    )



def test_calc_psth_with_empty_trains():
    trains = th.make_trains(
        [[], []]
    )
    psth, bin_edges = th.calc_psth(
        trains,
        bin_size=1,
        normalize=False
    )


    assert psth is None
    assert bin_edges is None


    ### duration != 0
    trains = th.make_trains(
        [[], []],
        duration=2
    )
    psth, bin_edges = th.calc_psth(
        trains,
        bin_size=1,
        normalize=False
    )

    assert_array_equal(
        psth,
        [0,0]
    )
    assert_array_equal(
        bin_edges,
        [0,1,2]
    )



def test_firint_rate():

    trains = th.make_trains(
        [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
        duration=[2, 4]
    )


    rate = th.calc_rate(trains)


    assert_equal(rate, 1.)




def test_count_spikes():
    trains = th.make_trains(
        [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
        duration=[2, 4]
    )

    count = th.count_spikes(trains)

    assert_equal(count, 6)




def test_calc_isih():

    trains = th.make_trains(
        [[1,2,3], [2,5,8]]
    )

    isih, bin_edges = th.calc_isih(
        trains,
        bin_size=1,
        normalize=False
    )

    assert_array_equal(
        isih,
        [ 0, 2, 2]
    )
    assert_array_equal(
        bin_edges,
        [ 0, 1, 2, 3]
    )



def test_calc_entrainment():

    trains = th.make_trains(
        [[1, 2, 3], [0, 2, 4]]
    )

    ent = th.calc_entrainment(
        trains,
        freq=1,
        bin_size=0.1
    )

    assert_equal(ent, 0.5)




    ### NaN test
    trains = th.make_trains(
        [[1], []]
    )

    ent = th.calc_entrainment(
        trains,
        freq=1,
        bin_size=0.1
    )

    assert np.isnan(ent)



def test_calc_si():

    trains = th.make_trains(
        [np.linspace(0, 1, 1000)]
    )

    si = th.calc_si(
        trains,
        freq=10
    )
    assert_almost_equal(si, 0)



    ### Next test
    trains = th.make_trains(
        [np.zeros(100)]
    )
    si = th.calc_si(
        trains,
        freq=10
    )
    assert_equal(si, 1)
