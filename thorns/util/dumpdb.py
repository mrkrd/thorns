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


logger = logging.getLogger('thorns')



def get_store(workdir='work'):

    fname = os.path.join(workdir, 'store.db')

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    store = shelve.open(fname, protocol=-1)

    return store




def dumpdb(data, name='dump', workdir='work', backend='shelve' ,kwargs=None):
    """Dump data in order to recall the most up-to-date records later

    Parameters
    ----------
    data : pands.DataFrame
        The DataFrame which should be saved
    name : str, optional
        Base name of the pickle file.
    workdir : str, optional
        Directory for the data.
    backend : str, optional
        the backend that should be used to store the data
    
    kwargs : dict, optional
        If given, all `xy` (parameter) dicts will be updated using `kwargs`.

    """
 
    backend_dict = {'shelve':0, 'hdf':1}
    assert backend in backend_dict, "backend unknown"
    
    bend = backend_dict[backend]
    
    if bend == 0:
        fname = os.path.join(workdir, name+'.db')
    else:
        fname = os.path.join(workdir, name+'.hdb')
        
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    if bend == 0:
        logger.info("Dumping pars (xs) and data (ys) into {}.".format(fname))
    else:
        logger.info("Dumping pars (xs) and data (ys) into {} as HDF.".format(fname))
        
    past = datetime.datetime.now()
    
    #Pickle the data using shelve
    if bend == 0:
        store = shelve.open(fname, protocol=-1)
        now = datetime.datetime.now()
        assert past < now, "Keys are conflicting"
    
        key = now.strftime("%Y%m%d-%H%M%S.%f")
        store[key] = data
    
        past = now
    else:
        pass






def loaddb(name='dump', workdir='work', backend='shelve'):
    """Recall dumped parameters/data discarding duplicated records

    Parameters
    ----------
    name : str, optional
        Base of the data filename.
    workdir : str, optional
        Directory where the data is stored.
    backend : str, optional
        the backend that was used to store the data


    Returns
    -------
    pd.DataFrame
        All data without duplicates.


    """

    fname = os.path.join(workdir, name+'.db')

    logger.info("Loading dumpdb from {}".format(fname))

    store = shelve.open(fname, protocol=-1)

    xkeys = set()
    db = pd.DataFrame()
    #index = []
    for key,record in sorted(store.iteritems()):
        record['timestamp'] = len(record) * [key]
        print(len(record))
        db = db.append(record)
        #index.append(key)

    db = db.sort('timestamp')
    groups = db.groupby(level=db.index.names)  
    db = groups.last()

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
