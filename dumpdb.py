#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import os
from glob import glob
import cPickle as pickle
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
        time_str + "__" + keys_str + ".pkl"
    )
    tmp_fname = fname + ".tmp"

    print("DUMPDB: dumping", fname)
    with open(tmp_fname, 'wb') as f:
        pickle.dump((data, xxkeys), f, -1)

    assert not os.path.exists(fname)
    os.rename(tmp_fname, fname)



def loaddb(dbdir=None):

    if dbdir is None:
        dbdir = os.path.join('work', 'dumpdb')


    pathname = os.path.join(dbdir, '*.pkl')

    xxkeys = set()
    data = []
    for fname in sorted(glob(pathname)):
        print("LOADDB: loading", fname)

        with open(fname, 'rb') as f:
            d, xk = pickle.load(f)

        xxkeys.update(xk)
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
