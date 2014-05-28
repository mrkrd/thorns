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


def plot(y=None, x=None, fs=None, kind=None, style=''):


    if 'MRplot' not in os.environ:
        return
    else:
        backend = os.environ['MRplot']

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
        ax.plot(x, y, style)

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





def set_rc_style(style='default'):
    """Set the style for plots by modifying mpl.rcParams (call before
    plotting) and axes (call after plotting).

    """

    import matplotlib as mpl

    mpl.rcdefaults()

    if style == 'jaro':
        style_dict = {
            'figure.figsize': (3.27, 2.5),
            'lines.linewidth': 1.5,
            'lines.markersize': 3,
            'axes.grid': False,
            'axes.color_cycle': ["#348ABD", "#A60628", "#7A68A6", "#467821", "#D55E00", "#CC79A7", "#56B4E9", "#009E73", "#F0E442", "#0072B2"],
            'font.family': 'sans',
            'font.size': 9,
            'legend.fancybox': True,
            'legend.frameon': False,
            'legend.fontsize': 'small',
            'legend.numpoints': 2,
        }

        mpl.rcParams.update(style_dict)

    elif style == 'default':
        pass

    else:
        raise NotImplementedError("Style not implemented: {}".format(style))



def set_fig_style(style='default', fig=None):

    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.gcf()

    if fig is not None:
        axes = fig.axes

    if style == 'jaro':

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()


        fig.set_figheight(2.5 * len(axes))
        fig.tight_layout()


    elif style == 'default':
        pass

    else:
        raise NotImplementedError("Style not implemented: {}".format(style))
