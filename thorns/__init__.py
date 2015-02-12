"""Spike analysis software.

"""

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"

__version__ = "0.7.1"

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

    logger = logging.getLogger('thorns')
    logger.setLevel(level)



import thorns.util
import thorns.waves
import thorns.io


from thorns.spikes import (
    make_trains,
    trains_to_array,
    accumulate,
    select_trains,
    trim,
    fold,
)


from thorns.stats import (
    get_duration,
    shuffled_autocorrelogram,
    correlation_index,
    firing_rate,
    psth,
    isih,
    entrainment,
    vector_strength,
    spike_count,
    period_histogram
)

from thorns.plotting import (
    plot_raster,
    plot_psth,
    plot_isih,
    plot_neurogram,
    plot_sac,
    plot_period_histogram,
    plot_signal,
    show,
    gcf,
)
