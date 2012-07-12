#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

from numpy.testing import *

import numpy as np

import marlib.thorns as th


def assert_trains_equal(a,b):

    assert len(a) == len(b)

    ### Assert if spike trains are equal
    for ta,tb in zip(a['spikes'], b['spikes']):
        assert np.all(ta == tb)


    ### Assert if meta data is equal
    assert a.dtype == b.dtype
    meta = list(a.dtype.names)
    meta.remove('spikes')
    assert_array_equal(a[meta], b[meta])



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


    assert_trains_equal(
        expected,
        result
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


    expected = [(np.array(a),10) for a in arrays]
    expected = np.array(
        expected,
        dtype=[
            ('spikes', np.ndarray),
            ('duration', float)
        ]
    )

    e = [(np.array(a),10) for a in arrays]
    e = np.array(
        e,
        dtype=[
            ('spikes', np.ndarray),
            ('duration', float)
        ]
    )
    assert_trains_equal(
        result,
        expected
    )




    result = th.make_trains(
        arrays,
        duration=10.0
    )
    assert_trains_equal(
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






