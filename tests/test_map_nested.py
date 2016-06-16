# -*- coding: utf-8 -*-

"""Test nesting of thorns.unil.map

"""

from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals

import tempfile
import shutil
import pytest

import numpy as np
from numpy.testing import assert_equal

import pandas as pd
from pandas.util.testing import assert_frame_equal

import thorns as th

import time


@pytest.fixture(scope="function")
def workdir(request):

    workdir = tempfile.mkdtemp()

    def fin():
        print("Removing temp dir: {}".format(workdir))

        shutil.rmtree(workdir, ignore_errors=True)

    request.addfinalizer(fin)

    return workdir





def square(x):
    return x**2



def sum_of_squares(length, workdir):

    vec = {'x': np.arange(length)}

    squares = th.util.map(
        square,
        vec,
        backend='multiprocessing', # should be inhibited to 'serial'
        cache='no',
        workdir=workdir
    )

    result = np.sum(squares.values)

    return result





def test_map_nested(workdir):

    lengths = {'length': [2, 3]}

    actual = th.util.map(
        sum_of_squares,
        lengths,
        backend='multiprocessing',
        cache='no',
        workdir=workdir,
        kwargs={'workdir': workdir},
    )

    expected = pd.DataFrame({
            'length': [2,3],
            0: [1,5],
    }).set_index('length')


    assert_frame_equal(expected, actual)
