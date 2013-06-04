#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import tempfile
import shutil

import numpy as np
from numpy.testing import (
    assert_array_equal
)

import mrlib as mr



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




    dbdir = tempfile.mkdtemp()


    mr.dumpdb(
        x1,y1,
        dbdir=dbdir
    )
    mr.dumpdb(
        x2,y2,
        dbdir=dbdir
    )



    db = mr.loaddb(
        dbdir=dbdir
    )


    assert len(db) == 2

    assert_array_equal(
        db.dbspl.values,
        [50, 60]
    )

    assert_array_equal(
        db.sac.values.tolist(),
        [[1,2], [20,30]]
    )


    shutil.rmtree(dbdir)


def test_kwargs():

    x1 = [
        {'dbspl': 50, 'cf': 400},
        {'dbspl': 60, 'cf': 400}
    ]
    y1 = [
        {'sac': np.array([1,2])},
        {'sac': np.array([2,3])}
    ]




    dbdir = tempfile.mkdtemp()


    mr.dumpdb(
        x1,
        y1,
        bla='anf',
        dbdir=dbdir
    )


    db = mr.loaddb(
        dbdir=dbdir
    )


    assert len(db) == 2

    assert_array_equal(
        db.dbspl.values,
        [50, 60]
    )

    assert_array_equal(
        db.sac.values.tolist(),
        [[1,2], [2,3]]
    )

    assert_array_equal(
        db.bla.values.tolist(),
        ['anf', 'anf']
    )

    shutil.rmtree(dbdir)
