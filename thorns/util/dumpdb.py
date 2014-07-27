#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements permanent store for data.

"""
from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals

__author__ = "Marek Rudnicki"

import os
import datetime
import logging
from itertools import izip_longest
import shelve

import numpy as np
import pandas as pd


logger = logging.getLogger('thorns')



def get_store(workdir='work'):

    fname = os.path.join(workdir, 'store.db')

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    store = shelve.open(fname, protocol=-1)

    return store




def dumpdb(data, name='dump', workdir='work', kwargs=None):
    """Dump data in order to recall the most up-to-date records later.

    Parameters
    ----------
    data : pd.DataFrame
        Data that will be appended to the database.
    name : str, optional
        Base name of the pickle file.
    workdir : str, optional
        Directory for the data.
    kwargs : dict, optional
        Additional parameters common for all data (MultiIndex will be
        extended).

    """
    fname = os.path.join(workdir, name+'.db')

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    logger.info("Dumping data into {}.".format(fname))


    if kwargs is not None:
        raise NotImplementedError("MultiIndex of data should be updated by kwargs here.")


    now = datetime.datetime.now()
    key = now.strftime("%Y%m%d-%H%M%S.%f")

    store = shelve.open(fname, protocol=-1)

    store[key] = data






def loaddb(name='dump', workdir='work'):
    """Recall dumped parameters/data discarding duplicated records

    Parameters
    ----------
    name : str, optional
        Base of the data filename.
    workdir : str, optional
        Directory where the data is stored.



    Returns
    -------
    pd.DataFrame
        All data without duplicates.


    """

    fname = os.path.join(workdir, name+'.db')

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
