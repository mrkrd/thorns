#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import pytest

import pandas as pd

import thorns as th



def test_get_duration():

    trains = pd.DataFrame([
        {'duration': 12, 'spikes': np.arange(10)},
        {'duration': 12, 'spikes': np.arange(15)},
    ])

    duration = th.get_duration(trains)

    assert_equal(duration, 12)



def test_get_duration_error():

    trains = pd.DataFrame([
        {'duration': 12, 'spikes': np.arange(10)},
        {'duration': 123, 'spikes': np.arange(15)},
    ])

    with pytest.raises(ValueError):
        duration = th.get_duration(trains)




def test_firing_rate():

    trains = th.make_trains(
        [[0.1, 0.4],
         [0.4, 0.5, 0.6]],
        duration=1
    )

    rate = th.firing_rate(
        trains
    )

    assert_equal(rate, 2.5)



def test_correlation_index():

    trains = th.make_trains(
        [[1,3],
         [1,2,3]]
    )


    ci = th.correlation_index(
        trains,
        normalize=False
    )

    assert_equal(ci, 4)




def test_shuffled_autocorrelogram():

    trains = th.make_trains(
        [[1,3],
         [1,2,3]]
    )

    sac, bin_edges = th.shuffled_autocorrelogram(
        trains,
        coincidence_window=1,
        analysis_window=3,
        normalize=False
    )


    assert_equal(
        bin_edges,
        [-2., -1.,  0.,  1.,  2.,  3.]
    )
    assert_equal(
        sac,
        [2, 2, 4, 2, 2]
    )




def test_psth():

    trains = th.make_trains(
        [[0.5, 1.5, 2.5],
         [0.5, 2.5]]
    )

    psth, bin_edges = th.psth(
        trains,
        bin_size=1,
        normalize=False
    )


    assert_equal(
        psth,
        [2, 1, 2]
    )
    assert_equal(
        bin_edges,
        [0, 1, 2, 3]
    )



def test_psth_with_empty_trains():
    trains = th.make_trains(
        [[], []]
    )
    psth, bin_edges = th.psth(
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
    psth, bin_edges = th.psth(
        trains,
        bin_size=1,
        normalize=False
    )

    assert_equal(
        psth,
        [0,0]
    )
    assert_equal(
        bin_edges,
        [0,1,2]
    )


def test_spike_count():
    trains = th.make_trains(
        [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
        duration=[2, 4]
    )

    count = th.spike_count(trains)

    assert_equal(count, 6)




def test_isih():

    trains = th.make_trains(
        [[1,2,3], [2,5,8]]
    )

    isih, bin_edges = th.isih(
        trains,
        bin_size=1,
    )

    assert_equal(
        isih,
        [ 0, 2, 2]
    )
    assert_equal(
        bin_edges,
        [ 0, 1, 2, 3]
    )



def test_entrainment():

    trains = th.make_trains(
        [[1, 2, 3], [0, 2, 4]]
    )

    ent = th.entrainment(
        trains,
        freq=1,
        bin_size=0.1
    )

    assert_equal(ent, 0.5)




    ### NaN test
    trains = th.make_trains(
        [[1], []]
    )

    ent = th.entrainment(
        trains,
        freq=1,
        bin_size=0.1
    )

    assert np.isnan(ent)



def test_vector_strength():

    ### Uniform spikes
    trains = th.make_trains(
        [np.arange(0, 1, 1/3600)]
    )

    si = th.vector_strength(
        trains,
        freq=10
    )
    assert_almost_equal(si, 0)



    ### Perfect synchrony
    trains = th.make_trains(
        [np.zeros(100)],
        duration=0.1
    )
    si = th.vector_strength(
        trains,
        freq=10
    )
    assert_equal(si, 1)



    ### Carefully chosen
    trains = th.make_trains(
        [np.tile([0, 0.25], 10)]
    )

    si = th.vector_strength(
        trains,
        freq=1
    )

    assert_equal(si, np.sqrt(2)/2)
