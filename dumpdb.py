#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import os
import glob
import gzip
import cPickle
import string
import time

import numpy as np



def dump(data, dbdir='tmp/dumpdb'):
    if not os.path.exists(dbdir):
        os.makedirs(dbdir)

    if isinstance(data, dict):
        data = [data]


    names_str = string.join(data[0], '-')
    time_str = str(time.time())
    fname = os.path.join(
        dbdir,
        time_str + "__" + names_str + ".pkl.gz"
    )
    tmp_fname = fname + ".tmp"

    with gzip.open(tmp_fname, 'wb', compresslevel=9) as f:
        cPickle.dump(data, f, -1)

    os.rename(tmp_fname, fname)






class DumpDB(object):
    def __init__(self, x, y, data=None, dbdir='tmp/dumpdb'):

        self.dbdir = dbdir
        self.x = x
        self.y = y

        if data is None:
            self.data = []
            for fname in glob.glob( os.path.join(self.dbdir, '*.pkl.gz') ):
                with gzip.open(fname, 'rb') as f:
                    d = cPickle.load(f)
                self._update_data(d)
        else:
            self.data = data



    def _update_data(self, new_dicts):
        for cur in new_dicts:

            ### Select keys for the new dict
            new = {}
            for k in (self.x + self.y):
                new[k] = cur[k]


            ### Remove old dictionary if a new is compatible
            for old in self.data:
                if self._are_dicts_equal(new, old, self.x):
                    self.data.remove( old )
                    break


            ### Store the new dict
            self.data.append( new )



    def _are_dicts_equal(self, a, b, keys):

        if len(a) != len(b):
            return False

        for key in keys:
            va = a[key]
            vb = b[key]

            ### Compare nparrrays
            if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                if np.any(va != vb):
                    return False

            ### Compare python objects
            else:
                if va != vb:
                    return False

        return True


    def get_col(self, name, raw=False, **kwargs):
        data = []
        for d in self.data:
            sel = np.all( [d[k] == kwargs[k] for k in kwargs] )
            if sel:
                data.append( d[name] )

        if not raw:
            data = np.array( data )

        return data


    def print_table(self, keys=None):
        # TODO print a nice table

        if keys is None:
            keys = self.data[0].keys()
            # TODO test if all dicts have the same keys

        for k in keys:
            print "|", k,
        print
        print "|-"

        for d in self.data:
            for k in keys:
                print "|", d[k],
            print


    def __str__(self):
        s = ""
        for d in self.data:
            s += str(d)
            s += "\n"

        return s


def main():
    db = DumpDB()
    print db.data

    print db.get_col('rate', cf=800)
    print db.get_col('si')


if __name__ == "__main__":
    main()
