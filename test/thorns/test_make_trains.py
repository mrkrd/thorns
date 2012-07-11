#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

from numpy.testing import *

import numpy as np

import marlib.thorns as th



def test_from_array():

    fs = 1e3
    a = np.array(
        [[0,0,0,1,0,0],
         [0,2,1,0,0,0]]
    ).T

    result = th.make_trains(a, fs=fs)


    expected = th.make_trains(
        [np.array([3])/fs,
         np.array([1,1,2])/fs],
        duration = 6/fs
    )


    print expected.dtype
    print result.dtype
    assert_array_equal(
        expected['spikes'],
        result['spikes']
    )

    assert len(result) == 2

    assert_array_equal(
        result['duration'],
        [6/fs, 6/fs]
    )

    assert_array_equal(
        result['spikes'][0],
        [3/fs]
    )
    assert_array_equal(
        result['spikes'][1],
        [1/fs, 1/fs, 2/fs]
    )



def test_from_arrays():

    arrays = [
        [1,2,3],
        [1,3,5,7],
        [0,8]
    ]


    result = th.spikes._arrays_to_trains(
        arrays,
        duration=10.0
    )

    expected = zip(arrays, [10]*len(arrays))
    expected = np.array(
        expected,
        dtype=[
            ('spikes', np.ndarray),
            ('duration', float)
        ]
    )
    assert_array_equal(
        expected,
        result
    )




    result = th.make_trains(
        arrays,
        duration=10.0
    )
    assert_array_equal(
        expected,
        result
    )






def test_trains_to_array():

    fs = 1e3
    trains = np.array(
        [(np.array([1/fs]), 5/fs),
         (np.array([2/fs, 3/fs]), 5/fs)],
        dtype=[('spikes', np.ndarray),
               ('duration', float)]
    )


    result = th.spikes.trains_to_array(
        trains,
        fs
    )


    assert_array_equal(
        result,
        [[0,0],
         [1,0],
         [0,1],
         [0,1],
         [0,0],
         [0,0]]
    )






