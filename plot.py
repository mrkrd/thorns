#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import argparse

import numpy as np
import matplotlib.pyplot as plt



_parser = argparse.ArgumentParser()

_parser.add_argument(
    '--plot',
    dest='plot',
    nargs=1
)



def plot(data, kind=None):

    if kind is None:

        if isinstance(data, np.ndarray) and data.ndim == 1:
            kind = 'vector'

        elif isinstance(data, np.ndarray) and data.ndim == 2:
            kind = 'imshow'




    fig, ax = plt.subplots(1,1)

    if kind in ('vector', 'plot'):
        ax.plot(data)

    elif kind in ('imshow', 'matrix'):
        ax.imshow(
            data,
            aspect='auto'
        )

    else:
        raise RuntimeError("Plot kind {} not implemented".format(kind))


    plt.show()



def main():
    pass

if __name__ == "__main__":
    main()
