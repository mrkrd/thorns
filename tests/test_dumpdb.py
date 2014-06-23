#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"

import shutil

import numpy as np
from numpy.testing import assert_equal


import thorns as th



def test_dump_and_load():

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


    th.dumpdb(
        x1,y1
    )
    th.dumpdb(
        x2,y2
    )



    db = th.loaddb()


    assert len(db) == 2

    assert_equal(
        db.dbspl.values,
        [50, 60]
    )

    assert_equal(
        db.sac.values.tolist(),
        [[1,2], [20,30]]
    )


    ### TODO: proper setup/teardown
    shutil.rmtree('work')



def test_kwargs():

    x1 = [
        {'dbspl': 50, 'cf': 400},
        {'dbspl': 60, 'cf': 400}
    ]
    y1 = [
        {'sac': np.array([1,2])},
        {'sac': np.array([2,3])}
    ]





    th.dumpdb(
        x1,
        y1,
        kwargs={'bla':'anf'}
    )


    db = th.loaddb()


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

    shutil.rmtree('work')
