#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements permanent store for data.

"""
from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals

import os
import datetime
import logging

from sys import version
import shelve
import collections
import cPickle as pickle

import tables

import numpy as np
import pandas as pd

from math import isnan

logger = logging.getLogger('thorns')


class NDArrayHandler(object):
    @staticmethod
    def tag(a):
        return 'array'

    @staticmethod
    def rep(a):
        return a.tolist()

    @staticmethod
    def string_rep(a):
        return None


class NumpyIntHandler(object):
    @staticmethod
    def tag(i):
        return 'i'

    @staticmethod
    def rep(i):
        return int(i)

    @staticmethod
    def string_rep(i):
        return str(i)


class NumpyFloatHandler(object):
    @staticmethod
    def tag(f):
        return "z" if isnan(f) or f in (float('Inf'), float('-Inf')) else "d"

    @staticmethod
    def rep(f):
        if isnan(f):
            return "NaN"
        if f == float('Inf'):
            return "INF"
        if f == float("-Inf"):
            return "-INF"
        return float(f)

    @staticmethod
    def string_rep(f):
        return str(f)


def get_store(name='store', workdir='work'):
    """Return a quick and dirty shelve based persisten dict-like store.

    """
    fname = os.path.join(workdir, name + '.db')

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    store = shelve.open(fname, protocol=-1)

    return store


def dump(data, name, workdir='work', backend='pickle'):
    """Dump a data object."""
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    if backend == 'pickle':
        fname = os.path.join(workdir, name+'.pkl')

        with open(fname, 'w') as f:
            pickle.dump(data, f, -1)

    elif backend == 'transit':
        fname = os.path.join(workdir, name+'.json')

        from transit.writer import Writer

        with open(fname, 'w') as f:
            writer = Writer(f, 'json')

            writer.register(np.ndarray, NDArrayHandler)
            writer.register(np.int64, NumpyIntHandler)
            writer.register(np.float64, NumpyFloatHandler)

            writer.write(data)



def dumpdb(data, name='dump', workdir='work', backend='hdf', kwargs=None):
    """Dump data in order to recall the most up-to-date records later.

    Parameters
    ----------
    data : pd.DataFrame
        Data that will be appended to the database.
    name : str, optional
        Base name of the pickle file.
    workdir : str, optional
        Directory for the data.
    backend : str, optional
        Data storage format.
    kwargs : dict, optional
        Additional parameters common for all data (MultiIndex will be
        extended).

    """
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    if kwargs is not None:
        for k,v in kwargs.items():
            data[k] = v
        data = data.set_index(kwargs.keys(), append=True)

    if backend == 'hdf':
        fname = os.path.join(workdir, name+'.h5')

        logger.info("Dumping data into {}.".format(fname))

        now = datetime.datetime.now()
        key = now.strftime("T%Y%m%d_%H%M%S_%f")

        store = pd.io.pytables.HDFStore(fname, 'a')
        store[key] = data

        store.close()

    elif backend == 'shelve':
        fname = os.path.join(workdir, name+'.db')

        logger.info("Dumping data into {}.".format(fname))

        now = datetime.datetime.now()
        key = now.strftime("T%Y%m%d_%H%M%S_%f")

        store = shelve.open(fname, protocol=-1)

        store[key] = data

        store.close()

    elif backend == 'transit':
        from transit.writer import Writer

        logger.info("Dumping data into {}.".format(fname))

        d = data.reset_index().to_dict('records')

        dump(d, name=name, workdir=workdir, backend='transit')

    else:
        raise NotImplementedError()



def loaddb(name='dump', workdir='work', backend='hdf', timestamp=False, load_all=False):
    """Recall dumped data discarding duplicated records.

    Parameters
    ----------
    name : str, optional
        Base of the data filename.
    workdir : str, optional
        Directory where the data is stored.
    backend : str, optional
        Backend used for data storage.
    timestamp : bool, optional
        Add an extra column with timestamps to the index.
    load_all : bool, optional
        If True, data from all experiments will be loaded from the
        dumpdb file.  The default is to load only the most recent
        data.

    Returns
    -------
    pd.DataFrame
        Data without duplicates.

    """
    if timestamp:
        raise NotImplementedError("Should add extra columnt with timestamps to the index of the output.")


    if backend == 'hdf':
        db = _loaddb_hdf(name, workdir, load_all)
    elif backend == 'shelve':
        db = _loaddb_shelve(name, workdir, load_all)
    else:
        raise NotImplementedError()

    return db



def _loaddb_hdf(name, workdir, load_all):

    fname = os.path.join(workdir, name+'.h5')
    store = pd.io.pytables.HDFStore(fname, 'r')

    logger.info("Loading data from {}".format(fname))

    if load_all:
        xkeys = collections.OrderedDict() # poor-man's ordered set
        dbs = []

        ### Get all tables from the store
        for t in sorted(store.keys()):
            df = store[t]

            # Just want ordered unique values in xkeys (ordered set would
            # simplify it: orderedset.update(df.index.names))
            for name in df.index.names:
                xkeys.setdefault(name)

            df = df.reset_index()
            dbs.append(df)

        db = pd.concat(dbs)

    else:
        last_key = sorted(store.keys())[-1]
        df = store[last_key]

        xkeys = df.index.names

        db = df.reset_index()

    store.close()

    ### Drop duplicates and set index
    db = db.drop_duplicates(
        subset=list(xkeys),
        keep='last',
    )

    db = db.set_index(list(xkeys))

    return db




def _loaddb_shelve(name, workdir, load_all):

    if load_all:
        raise NotImplementedError()

    fname = os.path.join(workdir, name+'.db')
    store = shelve.open(fname, protocol=-1)

    logger.info("Loading data from {}".format(fname))

    last_key = sorted(store.keys())[-1]

    df = store[last_key]

    store.close()

    # Drop records with duplicated keys
    xkeys = df.index.names

    db = df.reset_index()

    db = db.drop_duplicates(
        subset=list(xkeys),
        keep='last',
    )

    db = db.set_index(list(xkeys))

    return db
