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

    data = [
        {'dbspl': 50,
         'cf': 400,
         'sac': np.array([1,2])},

        {'dbspl': 60,
         'cf': 400,
         'sac': np.array([2,3])}
    ]


    dbdir = tempfile.mkdtemp()


    dumpdb.dump(
        data,
        dbdir=dbdir
    )



    db = dumpdb.DumpDB(
        x=['dbspl', 'cf'],
        y=['sac'],
        dbdir=dbdir
    )


    dbspl = db.get_col('dbspl')


    assert_array_equal(dbspl, [50, 60])


    shutil.rmtree(dbdir)

