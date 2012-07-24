#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import tempfile
import shutil

import numpy as np
from numpy.testing import (
    assert_array_equal
)

from marlib import dumpdb



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


    dumpdb.dump(
        x1,y1,
        dbdir=dbdir
    )
    dumpdb.dump(
        x2,y2,
        dbdir=dbdir
    )



    db = dumpdb.DumpDB(
        dbdir=dbdir
    )


    assert len(db.data) == 2


    dbspls = db.get_col('dbspl')
    assert_array_equal(dbspls, [50, 60])


    sacs, = db.get_col('sac', dbspl=60)
    assert_array_equal(sacs, np.array([20,30]))


    shutil.rmtree(dbdir)

