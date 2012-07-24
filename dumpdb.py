#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import os
import glob
import gzip
import cPickle
import numpy as np

class DumpDB(object):
    def __init__(self, data=None, db_dir='dumpdb'):
        self.db_dir = db_dir

        if data is None:
            self.data = []
            for fname in glob.glob( os.path.join(self.db_dir, '*') ):
                f = gzip.open(fname, 'rb')
                d = cPickle.load(f)
                self._update_data(d)
        else:
            self.data = data


    def _update_data(self, new_dicts):
        # print old_dicts
        for new in new_dicts:
            for old in self.data:
                if self._are_dicts_equal(new, old):
                    self.data.remove( old )
                    break
            self.data.append( new )


    def _are_dicts_equal(self, a, b):
        equal = True
        for v1,v2 in zip(a.values(), b.values()):
            if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                if np.any(v1 != v2):
                    equal = False
                    break
            else:
                if v1 != v2:
                    equal = False
                    break

        return equal


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
