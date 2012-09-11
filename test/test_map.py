#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
from numpy.testing import (
    assert_array_equal
)

import marlib as mar


def square(x):
    return x**2


def test_serial_map():

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results = mar.map(square, dicts, backend='serial')

    assert_array_equal(
        data**2,
        results
    )


def test_multiprocessing_map():

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results = mar.map(square, dicts, backend='multiprocessing')

    assert_array_equal(
        data**2,
        results
    )
