#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"

import shutil
import pytest
import tempfile

import numpy as np
from numpy.testing import assert_equal


import thorns as th



@pytest.fixture(scope="function")
def workdir(request):

    workdir = tempfile.mkdtemp()

    def fin():
        print("Removing temp dir: {}".format(workdir))

        shutil.rmtree(workdir, ignore_errors=True)

    request.addfinalizer(fin)

    return workdir




def test_dump_and_load(workdir):

    x1 = [
        {'dbspl': 50, 'cf': 400},
        {'dbspl': 60, 'cf': 400}
    ]
    y1 = [
        {'sac': np.array([1,2])},
        {'sac': np.array([2,3])}
    ]


    x2 = [
        {'dbspl': 50, 'cf': 400},
        {'dbspl': 60, 'cf': 400}
    ]
    y2 = [
        {'sac': np.array([1,2])},
        {'sac': np.array([20,30])}
    ]




    # store = tempfile.NamedTemporaryFile()
    # os.unlink(store.name)


    th.util.dumpdb(x1, y1, workdir=workdir)
    th.util.dumpdb(x2, y2, workdir=workdir)



    db = th.util.loaddb(workdir=workdir)


    assert len(db) == 2

    assert_equal(
        db.dbspl.values,
        [50, 60]
    )

    assert_equal(
        db.sac.values.tolist(),
        [[1,2], [20,30]]
    )





def test_kwargs(workdir):

    x1 = [
        {'dbspl': 50, 'cf': 400},
        {'dbspl': 60, 'cf': 400}
    ]
    y1 = [
        {'sac': np.array([1,2])},
        {'sac': np.array([2,3])}
    ]





    th.util.dumpdb(
        x1,
        y1,
        kwargs={'bla':'anf'},
        workdir=workdir
    )


    db = th.util.loaddb(workdir=workdir)


    assert len(db) == 2

    assert_equal(
        db.dbspl.values,
        [50, 60]
    )

    assert_equal(
        db.sac.values.tolist(),
        [[1,2], [2,3]]
    )

    assert_equal(
        db.bla.values.tolist(),
        ['anf', 'anf']
    )
