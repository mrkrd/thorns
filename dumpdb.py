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



def dumpdb(xx, yy=None, dbdir=None, **kwargs):

    if dbdir is None:
        dbdir = os.path.join('work', 'dumpdb')


    if not os.path.exists(dbdir):
        os.makedirs(dbdir)


    data = pd.DataFrame(xx)
    xxkeys = list(data.keys())

    if kwargs:
        pars = [kwargs for d in xx]
        data = data.join( pd.DataFrame(pars) )
        xxkeys.extend( kwargs.keys() )

    if yy is not None:
        data = data.join( pd.DataFrame(yy) )


    keys_str = string.join(data.keys(), '-')
    time_str = "{!r}".format(time.time())
    fname = os.path.join(
        dbdir,
        time_str + "__" + keys_str + ".pkl.gz"
    )
    tmp_fname = fname + ".tmp"

    print "DUMPDB: dumping", fname
    with gzip.open(tmp_fname, 'wb', compresslevel=9) as f:
        cPickle.dump((data, xxkeys), f, -1)

    assert not os.path.exists(fname)
    os.rename(tmp_fname, fname)



def loaddb(dbdir=None):

    if dbdir is None:
        dbdir = os.path.join('work', 'dumpdb')


    pathname = os.path.join(dbdir, '*.pkl.gz')

    xxkeys = None
    data = []
    for fname in sorted(glob(pathname)):

        with gzip.open(fname, 'rb') as f:
            d, xk = cPickle.load(f)

        if xxkeys is not None:
            assert np.all(xxkeys == xk)
        xxkeys = xk

        data.append(d)


    data = pd.concat(
        data,
        ignore_index=True
    )

    data.drop_duplicates(
        cols=list(xxkeys),
        take_last=True,
        inplace=True
    )

    return data
