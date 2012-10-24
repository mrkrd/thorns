#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import os
from glob import glob
import cPickle as pickle
import string
import datetime

import numpy as np
import pandas as pd




def _dump_records(pars, data, dbdir):

    timestamp = datetime.datetime.now()

    records = {
        'pars': pars,
        'data': data,
        'timestamp': timestamp
    }

    keys_str = string.join(pars.keys() + data.keys(), '_')
    time_str = timestamp.strftime("%Y%m%d-%H%M%S.%f")
    fname = os.path.join(
        dbdir,
        time_str + "__" + keys_str + ".pkl"
    )
    tmp_fname = fname + ".tmp"

    print("DUMPDB: dumping", fname)
    with open(tmp_fname, 'wb') as f:
        pickle.dump(records, f, -1)

    assert not os.path.exists(fname)
    os.rename(tmp_fname, fname)




def dumpdb(pars, data, dbdir=None, **kwargs):



    if isinstance(pars, dict):
        pars = [pars]

    if isinstance(data, dict):
        data = [data]



    if dbdir is None:
        dbdir = os.path.join('work', 'dumpdb')

    if not os.path.exists(dbdir):
        os.makedirs(dbdir)


    assert len(pars) == len(data)

    for p,d in zip(pars,data):
        p.update(kwargs)

        _dump_records(p, d, dbdir)









def loaddb(dbdir=None):

    if dbdir is None:
        dbdir = os.path.join('work', 'dumpdb')


    pathname = os.path.join(dbdir, '*.pkl')

    pars_keys = set()
    db = []
    index = []
    for fname in sorted(glob(pathname)):
        print("LOADDB: loading", fname)

        with open(fname, 'rb') as f:
            records = pickle.load(f)

        pars_keys.update(records['pars'].keys())
        row = records['pars']
        row.update(records['data'])

        db.append(row)
        index.append(records['timestamp'])


    db = pd.DataFrame(db, index=index)

    db.drop_duplicates(
        cols=list(pars_keys),
        take_last=True,
        inplace=True
    )

    return db
