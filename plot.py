#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import numpy as np
import datetime
import os

import mrlib as mr




def show():
    import matplotlib.pyplot as plt
    plt.show()


def plot(y=None, x=None, fs=None, kind=None):

    backend = mr.args.plot

    if backend is None:
        return


    if fs is None:
        fs = 1

    if x is None:
        x = np.arange(len(y)) / fs




    if kind is None:
        if isinstance(y, np.ndarray) and y.ndim == 1:
            kind = 'vector'
        elif isinstance(y, list):
            kind = 'vector'
        elif isinstance(y, np.ndarray) and y.ndim == 2:
            kind = 'imshow'




    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)

    if kind in ('vector', 'plot'):
        ax.plot(x, y)

    elif kind in ('imshow', 'matrix'):
        img = ax.imshow(y, aspect='auto')
        plt.colorbar(img)

    else:
        raise RuntimeError("Plot kind {} not implemented".format(kind))





    dirname = "work/plots"
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
    fname = os.path.join(dirname, time_str + "__" + kind + "." + backend)

    if backend == 'png':
        mr.mkdir(dirname)
        fig.savefig(fname)

    elif backend == 'pdf':
        mr.mkdir(dirname)
        fig.savefig(fname)

    elif backend == 'show':
        plt.show()

    else:
        raise RuntimeError("Unknown plotting backend: {}".format(backend))




def main():
    pass

if __name__ == "__main__":
    main()
