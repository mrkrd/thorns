"""
Spike analysis software.

"""

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"


import os
import sys


if 'THpdb' in os.environ:
    import pdb, sys, traceback
    def info(type, value, tb):
        traceback.print_exception(type, value, tb)
        pdb.pm()
    sys.excepthook = info


if 'THlog' in os.environ:
    import logging

    if os.environ['THlog'] in ('d', 'debug'):
        level = 'DEBUG'

    else:
        level = os.environ['THlog'].upper()

    logger = logging.getLogger()
    logger.setLevel(level)



import thorns.util


from thorns.spikes import (
    make_trains,

    trains_to_array,

    accumulate,
    accumulate_spikes,
    accumulate_trains,
    accumulate_spike_trains,

    select_trains,
    select,
    sel,

    trim_spike_trains,
    trim_trains,
    trim,

    fold_spike_trains,
    fold_trains,
    fold,
)


from thorns.stats import (
    get_duration,

    shuffled_autocorrelogram,
    sac,

    correlation_index,
    ci,

    firing_rate,
    rate,

    psth,

    isih,

    entrainment,

    synchronization_index,
    si,

    count_spikes,
    count,

    period_histogram
)

from thorns.plotting import (
    plot_raster,
    plot_psth,
    plot_neurogram,
    plot_sac,
    plot_period_histogram,
    plot_signal,
    show
)


def gcf():
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    return fig
