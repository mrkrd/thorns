#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import os
from glob import glob
import gzip
import cPickle
import string
import time

import numpy as np
import pandas as pd



def dumpdb(x, y=None, dbdir=None):

    if dbdir is None:
        dbdir = os.path.join('tmp', 'dumpdb')


    if not os.path.exists(dbdir):
        os.makedirs(dbdir)


    data = pd.DataFrame(x)
    xkeys = data.keys()

    if y is not None:
        data = data.join( pd.DataFrame(y) )


    keys_str = string.join(data.keys(), '-')
    time_str = "{!r}".format(time.time())
    fname = os.path.join(
        dbdir,
        time_str + "__" + keys_str + ".pkl.gz"
    )
    tmp_fname = fname + ".tmp"

    print "dumping:", fname
    with gzip.open(tmp_fname, 'wb', compresslevel=9) as f:
        cPickle.dump((data, xkeys), f, -1)

    assert not os.path.exists(fname)
    os.rename(tmp_fname, fname)



def loaddb(dbdir=None):

    if dbdir is None:
        dbdir = os.path.join('tmp', 'dumpdb')


    pathname = os.path.join(dbdir, '*.pkl.gz')

    xkeys = None
    data = []
    for fname in sorted(glob(pathname)):

        with gzip.open(fname, 'rb') as f:
            d, xk = cPickle.load(f)

        if xkeys is not None:
            assert np.all(xkeys == xk)
        xkeys = xk

        data.append(d)


    data = pd.concat(
        data,
        ignore_index=True
    )

    data.drop_duplicates(
        cols=list(xkeys),
        take_last=True,
        inplace=True
    )

    return data
