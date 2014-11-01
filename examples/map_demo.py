# -*- coding: utf-8 -*-

"""Map demo.

"""

from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals

__author__ = "Marek Rudnicki"
__copyright__ = "Copyright 2014, Marek Rudnicki"
__license__ = "GPLv3+"


import numpy as np
import pandas as pd

import thorns as th


def square(x):
    y = {'y': x**2}
    return y


def main():

    xs = {'x': np.arange(10)}

    ys = th.util.map(
        square,
        xs
    )

    print(ys)



if __name__ == "__main__":
    main()
