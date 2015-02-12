#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"

import shutil
import pytest
import tempfile

import numpy as np
from numpy.testing import assert_equal
import pandas as pd
from pandas.util.testing import assert_frame_equal

import thorns as th


@pytest.fixture(scope="function")
def workdir(request):

    workdir = tempfile.mkdtemp()

    def fin():
        print("Removing temp dir: {}".format(workdir))

        shutil.rmtree(workdir, ignore_errors=True)

    request.addfinalizer(fin)

    return workdir




def test_dump_and_load_single_df(workdir):

    data = pd.DataFrame([
        {'a': 1, 'b': 1.1, 'c': 1   },
        {'a': 2, 'b': 2.2, 'c': 4   },
        {'a': 3, 'b': 3.3, 'c': 9   },
        {'a': 1, 'b': 1.1, 'c': 1.11}, # duplicate of the 0th row
    ]).set_index(['a', 'b'])

    expected = pd.DataFrame([
        {'a': 2, 'b': 2.2, 'c': 4   },
        {'a': 3, 'b': 3.3, 'c': 9   },
        {'a': 1, 'b': 1.1, 'c': 1.11},
    ]).set_index(['a', 'b'])


    th.util.dumpdb(data, workdir=workdir)

    actual = th.util.loaddb(workdir=workdir)

    assert_frame_equal(actual, expected)


def test_dump_and_load(workdir):

    data1 = pd.DataFrame([
        {'x': 50, 'y': 400, 'f': np.array([1,2])},
        {'x': 55, 'y': 400, 'f': np.array([5,5])},
        {'x': 60, 'y': 400, 'f': np.array([2,3])},
    ]).set_index(['x','y'])

    data2 = pd.DataFrame([
        {'x': 50, 'y': 400, 'f': np.array([1,2])},
        {'x': 60, 'y': 400, 'f': np.array([20,30])},
    ]).set_index(['x','y'])



    th.util.dumpdb(data1, workdir=workdir)
    th.util.dumpdb(data2, workdir=workdir)



    db = th.util.loaddb(workdir=workdir)


    expected = data2


    assert_frame_equal(db, expected)




def test_dump_and_load_all(workdir):

    data1 = pd.DataFrame([
        {'x': 50, 'y': 400, 'f': np.array([1,2])},
        {'x': 55, 'y': 400, 'f': np.array([5,5])},
        {'x': 60, 'y': 400, 'f': np.array([2,3])},
    ]).set_index(['x','y'])

    data2 = pd.DataFrame([
        {'x': 50, 'y': 400, 'f': np.array([1,2])},
        {'x': 60, 'y': 400, 'f': np.array([20,30])},
    ]).set_index(['x','y'])



    th.util.dumpdb(data1, workdir=workdir)
    th.util.dumpdb(data2, workdir=workdir)



    db = th.util.loaddb(workdir=workdir, load_all=True)


    expected = pd.DataFrame([
        {'x': 55, 'y': 400, 'f': np.array([5,5])},
        {'x': 50, 'y': 400, 'f': np.array([1,2])},
        {'x': 60, 'y': 400, 'f': np.array([20,30])},
    ]).set_index(['x','y'])


    assert_frame_equal(db, expected)







def test_kwargs(workdir):

    data = pd.DataFrame([
        {'x': 50, 'y': 400, 'f': np.array([1,2])},
        {'x': 60, 'y': 400, 'f': np.array([2,3])},
    ]).set_index(['x','y'])



    th.util.dumpdb(
        data,
        kwargs={'type':'anf'},
        workdir=workdir,
    )


    db = th.util.loaddb(workdir=workdir)


    expected = pd.DataFrame([
        {'x': 50, 'y': 400, 'type': 'anf', 'f': np.array([1,2])},
        {'x': 60, 'y': 400, 'type': 'anf', 'f': np.array([2,3])},
    ]).set_index(['x','y','type'])


    assert_frame_equal(db, expected)
