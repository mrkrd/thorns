#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"


import tempfile
import shutil
import unittest

import numpy as np
from numpy.testing import (
    assert_array_equal
)
from nose.tools import with_setup
import marlib as mr


def square(x):
    return x**2


def setup_dir():
    global cachedir
    cachedir = tempfile.mkdtemp()

def teardown_dir():
    global cachedir
    shutil.rmtree(cachedir, ignore_errors=True)



@with_setup(setup_dir, teardown_dir)
def test_serial_map():

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


    assert_array_equal(
        data**2,
        list(results1)
    )
    assert_array_equal(
        data**2,
        list(results2)
    )


@with_setup(setup_dir, teardown_dir)
def test_multiprocessing_map():


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


    assert_array_equal(
        data**2,
        list(results1)
    )
    assert_array_equal(
        data**2,
        list(results2)
    )



@with_setup(setup_dir, teardown_dir)
def test_playdoh_map():

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


    assert_array_equal(
        data**2,
        list(results1)
    )
    assert_array_equal(
        data**2,
        list(results2)
    )


@unittest.skip("not working with nosetest")
@with_setup(setup_dir, teardown_dir)
def test_ipython_map():

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = mr.map(
        square,
        dicts,
        backend='ipython',
        cachedir=cachedir
    )
    results2 = mr.map(
        square,
        dicts,
        backend='ipython',
        cachedir=cachedir
    )


    assert_array_equal(
        data**2,
        list(results1)
    )
    assert_array_equal(
        data**2,
        list(results2)
    )


@unittest.skip("nose does not like fork")
@with_setup(setup_dir, teardown_dir)
def test_serial_fork_map():


    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = mr.map(
        square,
        dicts,
        backend='serial_fork',
        cachedir=cachedir
    )

    results2 = mr.map(
        square,
        dicts,
        backend='serial_fork',
        cachedir=cachedir
    )


    try:
        results1 = list(results1)
        results2 = list(results2)
    except SystemExit:
        pass


    assert_array_equal(
        data**2,
        results1
    )
    assert_array_equal(
        data**2,
        results2
    )



if __name__ == '__main__':
    test_serial_fork_map()
    test_ipython_map()
