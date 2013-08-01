#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import mrlib as mr

def main():

    space = [
        {
            'a': a,
            'b': b
        }
        for a in range(3)
        for b in range(3)
    ]

    results = [
        {
            'c': c,
        }
        for c in range(3)
    ]



    mr.dumpdb(space, results)


    db = mr.loaddb()

    print(db)




if __name__ == "__main__":
    main()
