#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"


import tempfile
import shutil
import pytest

import numpy as np
from numpy.testing import assert_equal

import pandas as pd
from pandas.util.testing import assert_frame_equal

import thorns as th


def square(x):
    return x**2

def multiply(x,y):
    return x*y


@pytest.fixture(scope="function")
def workdir(request):

    workdir = tempfile.mkdtemp()

    def fin():
        print("Removing temp dir: {}".format(workdir))

        shutil.rmtree(workdir, ignore_errors=True)

    request.addfinalizer(fin)

    return workdir



def test_map_serial(workdir):

    space = [{'x': i} for i in range(10)]

    results = th.util.map(
        square,
        space,
        backend='serial',
        cache='no',
        workdir=workdir,
    )


    expected = pd.DataFrame(
        {
            'x': range(10),
            0: np.arange(10)**2
        }
    ).set_index('x')

    assert_frame_equal(results, expected)






def test_map_cache(workdir):

    data = np.arange(10)
    dicts = [{'x':i} for i in data]

    df1 = th.util.map(
        square,
        dicts,
        backend='serial',
        cache='yes',
        workdir=workdir,
    )
    df2 = th.util.map(
        square,
        dicts,
        backend='serial',
        cache='yes',
        workdir=workdir,
    )


    assert_frame_equal(df1, df2)



def test_map_kwargs(workdir):

    space = [{'x': i} for i in range(10)]
    kwargs = {'y': 2}


    results = th.util.map(
        multiply,
        space,
        backend='serial',
        cache='no',
        workdir=workdir,
        kwargs=kwargs,
    )

    expected = pd.DataFrame(
        {
            'x': range(10),
            0: np.arange(10)*2
        }
    ).set_index('x')

    assert_frame_equal(results, expected)



def test_map_multiprocessing(workdir):

    space = [{'x': i} for i in range(10)]

    results = th.util.map(
        square,
        space,
        backend='multiprocessing',
        cache='no',
        workdir=workdir,
    )


    expected = pd.DataFrame(
        {
            'x': range(10),
            0: np.arange(10)**2
        }
    ).set_index('x')

    assert_frame_equal(results, expected)






def test_map_playdoh(workdir):

    space = [{'x': i} for i in range(10)]

    results = th.util.map(
        square,
        space,
        backend='playdoh',
        cache='no',
        workdir=workdir,
    )


    expected = pd.DataFrame(
        {
            'x': range(10),
            0: np.arange(10)**2
        }
    ).set_index('x')

    assert_frame_equal(results, expected)




def test_map_serial_isolated(workdir):

    space = [{'x': i} for i in range(10)]

    results = th.util.map(
        square,
        space,
        backend='serial_isolated',
        cache='no',
        workdir=workdir,
    )


    expected = pd.DataFrame(
        {
            'x': range(10),
            0: np.arange(10)**2
        }
    ).set_index('x')

    assert_frame_equal(results, expected)




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





if __name__ == '__main__':
    test_ipython_map()
