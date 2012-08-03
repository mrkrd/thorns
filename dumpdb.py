#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import os
import sys
from glob import glob
import gzip
import cPickle
import string
import time

import numpy as np




def dump(x, y=None, dbdir=None):

    if dbdir is None:
        d = os.path.splitext(
            os.path.basename(sys.argv[0])
        )[0]
        dbdir = os.path.join('tmp', d)


    if not os.path.exists(dbdir):
        os.makedirs(dbdir)


    if y is None:
        data = x
    else:
        data = []
        for a,b in zip(x,y):
            d = dict(a)
            d.update(b)
            data.append(d)

    xkeys = x[0].keys()


    names_str = string.join(data[0], '-')

    time_str = "{!r}".format(time.time())
    fname = os.path.join(
        dbdir,
        time_str + "__" + names_str + ".pkl.gz"
    )
    tmp_fname = fname + ".tmp"

    print "dumping:", fname
    with gzip.open(tmp_fname, 'wb', compresslevel=9) as f:
        cPickle.dump((data, xkeys), f, -1)

    assert not os.path.exists(fname)
    os.rename(tmp_fname, fname)





class DumpDB(object):
    def __init__(self, dbdir, data=None):

        assert os.path.exists(dbdir)

        self.dbdir = dbdir

        if data is None:
            self.data = []
            self.xkeys = None
            pathname = os.path.join(self.dbdir, '*.pkl.gz')

            for fname in sorted(glob(pathname)):

                with gzip.open(fname, 'rb') as f:
                    dicts, xkeys = cPickle.load(f)

                if self.xkeys is None:
                    self.xkeys = xkeys
                else:
                    assert self.xkeys == xkeys

                self._update_data(dicts)

        else:
            self.data = data
            self.xkeys = data.keys()




    def _update_data(self, new_dicts):
        for new in new_dicts:

            ### Remove old dictionary if a new is compatible
            for old in self.data:
                if self._are_dicts_equal(new, old, self.xkeys):
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


    def get_col(self, name, **kwargs):
        data = []
        for d in self.data:
            sel = np.all( [np.all(d[k] == kwargs[k]) for k in kwargs] )
            if sel:
                data.append( d[name] )

        return data


    def get_val(self, name, **kwargs):
        col = self.get_col(name, **kwargs)

        assert len(col) == 1, "Selected value not unique"

        return col[0]



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
