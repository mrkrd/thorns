#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Demo that dumps and load data to a permanent storage.

"""

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"

import mrlib as mr

def main():

    ### Create data with duplicates (records)
    space = [
        {'a': 1, 'b': 1.1},
        {'a': 2, 'b': 2.2},
        {'a': 3, 'b': 3.3},
        {'a': 1, 'b': 1.1},
    ]
    results = [
        {'c': 1},
        {'c': 4},
        {'c': 9},
        {'c': 1.11},
    ]


    ### Dump the data
    mr.dumpdb(space, results)



    ### Load the data, note that duplicated resuts are dropped (only
    ### the most recent data is returned)
    db = mr.loaddb()



    ### Show the results
    print(db)




if __name__ == "__main__":
    main()
