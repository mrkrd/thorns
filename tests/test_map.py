#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"


import tempfile
import shutil
import pytest

import numpy as np
from numpy.testing import assert_equal

import thorns as th


def square(x):
    return x**2

@pytest.fixture(scope="function")
def workdir(request):
    global workdir
    workdir = tempfile.mkdtemp()

    def fin():
        print("Removing temp dir: {}".format(workdir))

        shutil.rmtree(workdir, ignore_errors=True)

    return workdir




def test_serial_map(workdir):

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = th.util.map(
        square,
        dicts,
        backend='serial',
        workdir=workdir,
    )
    results2 = th.util.map(
        square,
        dicts,
        backend='serial',
        workdir=workdir
    )


    assert_equal(
        data**2,
        list(results1)
    )
    assert_equal(
        data**2,
        list(results2)
    )


def test_multiprocessing_map(workdir):


    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = th.util.map(
        square,
        dicts,
        backend='multiprocessing',
        workdir=workdir
    )
    results2 = th.util.map(
        square,
        dicts,
        backend='multiprocessing',
        workdir=workdir
    )


    assert_equal(
        data**2,
        list(results1)
    )
    assert_equal(
        data**2,
        list(results2)
    )



def test_playdoh_map(workdir):

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = th.util.map(
        square,
        dicts,
        backend='playdoh',
        workdir=workdir
    )
    results2 = th.util.map(
        square,
        dicts,
        backend='playdoh',
        workdir=workdir
    )


    assert_equal(
        data**2,
        list(results1)
    )
    assert_equal(
        data**2,
        list(results2)
    )


@pytest.mark.skipif('True')
def test_ipython_map(workdir):

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    results1 = th.util.map(
        square,
        dicts,
        backend='ipython',
        workdir=workdir
    )
    results2 = th.util.map(
        square,
        dicts,
        backend='ipython',
        workdir=workdir
    )

    assert_equal(
        data**2,
        list(results1)
    )
    assert_equal(
        data**2,
        list(results2)
    )


def test_isolated_serial_map(workdir):

    data = np.arange(3)
    dicts = [{'x':i} for i in data]

    results1 = th.util.map(
        square,
        dicts,
        backend='serial_isolated',
        workdir=workdir
    )
    results1 = list(results1)

    results2 = th.util.map(
        square,
        dicts,
        backend='serial_isolated',
        workdir=workdir
    )
    results2 = list(results2)



    assert_equal(
        data**2,
        results1
    )
    assert_equal(
        data**2,
        results2
    )



if __name__ == '__main__':
    test_ipython_map()
