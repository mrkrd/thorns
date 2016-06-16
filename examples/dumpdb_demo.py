#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Demo that dumps and load data to a permanent storage.

"""

from __future__ import division, print_function, absolute_import

import pandas as pd
import thorns as th

def main():

    ### Create data with duplicates (records)
    data = pd.DataFrame([
        {'a': 1, 'b': 1.1, 'c': 1   },
        {'a': 2, 'b': 2.2, 'c': 4   },
        {'a': 3, 'b': 3.3, 'c': 9   },
        {'a': 1, 'b': 1.1, 'c': 1.11}, # duplicate of the 0th row
    ])

    data = data.set_index(['a', 'b'])


    ### Dump the data
    th.util.dumpdb(data)



    ### Load the data, note that duplicated resuts are dropped (only
    ### the most recent data is returned)
    db = th.util.loaddb()



    ### Show the results
    print(db)




if __name__ == "__main__":
    main()
