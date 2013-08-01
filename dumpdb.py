#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import os
import datetime
import logging
from itertools import izip_longest
import shelve

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)




def dumpdb(xs=None, ys=None, name='dump', **kwargs):

    ## the case only data is given
    if ys is None:
        ys = xs
        xs = None

    if xs is None:
        xs = []
    elif isinstance(xs, dict):
        xs = [xs]

    if ys is None:
        ys = []
    elif (ys is None) or isinstance(ys, dict):
        ys = [ys]

    dbdir = 'work'

    fname = os.path.join(dbdir, name)

    if not os.path.exists(dbdir):
        os.makedirs(dbdir)

    logger.info("Dumping pars (xs) and data (ys) into {}.".format(fname))


    past = datetime.datetime.now()
    store = shelve.open(fname, protocol=-1)
    for x,y in izip_longest(xs, ys, fillvalue={}):
        now = datetime.datetime.now()
        assert past < now, "Keys are conflicting"

        x.update(kwargs)
        record = {
            'x': x,
            'y': y,
        }

        key = now.strftime("%Y%m%d-%H%M%S.%f")
        store[key] = record

        past = now






def loaddb(name='dump'):

    dbdir = 'work'

    fname = os.path.join(dbdir, name)

    logger.info("Loading dumpdb from {}".format(fname))

    store = shelve.open(fname, protocol=-1)

    xkeys = set()
    db = []
    index = []
    for key,record in sorted(store.iteritems()):
        xkeys.update(record['x'].keys())
        row = record['x']
        row.update(record['y'])

        db.append(row)
        index.append(key)


    db = pd.DataFrame(db, index=index)

    # Has to wrap arrays to be able to compare them
    _wrap_arrays(db)
    db.drop_duplicates(
        cols=list(xkeys),
        take_last=True,
        inplace=True
    )
    _unwrap_arrays(db)

    return db




class ArrayCompareWrapper(object):
    def __init__(self, arr):
        self.arr = arr
    def __eq__(self, other):
        return np.all(self.arr == other.arr)


def _wrap_arrays(db):
    for col in db.columns:
        for idx in db.index:
            val = db[col][idx]
            if isinstance(val, np.ndarray):
                db[col][idx] = ArrayCompareWrapper(val)


def _unwrap_arrays(db):
    for col in db.columns:
        for idx in db.index:
            val = db[col][idx]
            if isinstance(val, ArrayCompareWrapper):
                db[col][idx] = val.arr
