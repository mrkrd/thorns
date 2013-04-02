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
import elmar as mr


def square(x):
    return x**2


def setup_dir():
    global workdir
    workdir = tempfile.mkdtemp()

def teardown_dir():
    global workdir
    shutil.rmtree(workdir, ignore_errors=True)



@with_setup(setup_dir, teardown_dir)
def test_serial_map():

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = mr.map(
        square,
        dicts,
        backend='serial',
        workdir=workdir
    )
    results2 = mr.map(
        square,
        dicts,
        backend='serial',
        workdir=workdir
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
        workdir=workdir
    )
    results2 = mr.map(
        square,
        dicts,
        backend='multiprocessing',
        workdir=workdir
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
        workdir=workdir
    )
    results2 = mr.map(
        square,
        dicts,
        backend='playdoh',
        workdir=workdir
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
        workdir=workdir
    )
    results2 = mr.map(
        square,
        dicts,
        backend='ipython',
        workdir=workdir
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
def test_serial_proc_map():


    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = mr.map(
        square,
        dicts,
        backend='serial_proc',
        workdir=workdir
    )
    results1 = list(results1)

    results2 = mr.map(
        square,
        dicts,
        backend='serial_proc',
        workdir=workdir
    )
    results2 = list(results2)



    assert_array_equal(
        data**2,
        results1
    )
    assert_array_equal(
        data**2,
        results2
    )



if __name__ == '__main__':
    test_ipython_map()
