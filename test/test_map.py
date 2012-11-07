#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"


import tempfile
import shutil

import numpy as np
from numpy.testing import (
    assert_array_equal
)

import marlib as mr


def square(x):
    return x**2


def test_serial_map():

    cachedir = tempfile.mkdtemp()

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = mr.map(
        square,
        dicts,
        backend='serial',
        cachedir=cachedir
    )
    results2 = mr.map(
        square,
        dicts,
        backend='serial',
        cachedir=cachedir
    )

    shutil.rmtree(cachedir)

    assert_array_equal(
        data**2,
        list(results1)
    )
    assert_array_equal(
        data**2,
        list(results2)
    )


def test_multiprocessing_map():

    cachedir = tempfile.mkdtemp()

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = mr.map(
        square,
        dicts,
        backend='multiprocessing',
        cachedir=cachedir
    )
    results2 = mr.map(
        square,
        dicts,
        backend='multiprocessing',
        cachedir=cachedir
    )

    shutil.rmtree(cachedir)

    assert_array_equal(
        data**2,
        list(results1)
    )
    assert_array_equal(
        data**2,
        list(results2)
    )



def test_playdoh_map():

    cachedir = tempfile.mkdtemp()

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = mr.map(
        square,
        dicts,
        backend='playdoh',
        cachedir=cachedir
    )
    results2 = mr.map(
        square,
        dicts,
        backend='playdoh',
        cachedir=cachedir
    )

    shutil.rmtree(cachedir)

    assert_array_equal(
        data**2,
        list(results1)
    )
    assert_array_equal(
        data**2,
        list(results2)
    )



if __name__ == '__main__':
    test_serial_map()
