#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"

import numpy as np

from mrlib.thorns import spikes
from mrlib.thorns import calc

GOLDEN = 1.6180339887


def plot_neurogram(spike_trains, fs, axis=None, **kwargs):

    neurogram = spikes.trains_to_array(
        spike_trains,
        fs
    )

    if axis is None:
        import matplotlib.pyplot as plt
        axis = plt.gca()

    extent = (
        0,                       # left
        neurogram.shape[0] / fs, # right
        0,                       # bottom
        neurogram.shape[1]       # top
    )

    axis.imshow(
        neurogram.T,
        aspect='auto',
        extent=extent,
        **kwargs
    )

    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Channel number")

    return axis




def plot_raster(spike_trains, axis=None, style='k.', **kwargs):
    """Plot raster plot."""

    trains = spike_trains['spikes']
    duration = np.max( spike_trains['duration'] )

    # Compute trial number
    L = [ len(train) for train in trains ]
    r = np.arange(len(trains))
    n = np.repeat(r, L)

    # Spike timings
    s = np.concatenate(tuple(trains))


    if axis is None:
        import matplotlib.pyplot as plt
        axis = plt.gca()

    axis.plot(s, n, style, **kwargs)
    axis.set_xlabel("Time [s]")
    axis.set_xlim( (0, duration) )
    axis.set_ylabel("Trial Number")
    axis.set_ylim( (-0.5, len(trains)-0.5) )


    return axis





def plot_psth(spike_trains, bin_size, axis=None, **kwargs):
    """Plots PSTH of spike_trains."""


    psth, bin_edges = calc.psth(
        spike_trains,
        bin_size
    )


    if axis is None:
        import matplotlib.pyplot as plt
        axis = plt.gca()


    axis.plot(
        bin_edges[:-1],
        psth,
        drawstyle='steps-post',
        **kwargs
    )


    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Spikes per Second")


    return axis


# def isih(spike_trains, bin_size=1e-3, plot=None, **style):
#     """Plot inter-spike interval histogram."""

#     hist = stats.isih(spike_trains, bin_size)

#     c = biggles.Histogram(hist, x0=0, binsize=bin_size)
#     c.style(**style)

#     if plot is None:
#         plot = biggles.FramedPlot()
#     plot.xlabel = "Inter-Spike Interval [ms]"
#     plot.ylabel = "Probability Density Function"
#     plot.add(c)
#     plot.xrange = (0, None)
#     plot.yrange = (0, None)

#     return plot


def plot_period_histogram(
        spike_trains,
        freq,
        nbins=64,
        axis=None,
        style='',
        density=False,
        **kwargs
):
    """Plot period histogram of the given spike trains.

    Parameters
    ----------
    spike_trains : spike_trains
        Spike trains for plotting.
    freq : float
        Stimulus frequency.
    nbins : int
        Number of bins for the histogram.
    axis : plt.Axis, optional
        Matplotlib Axis to plot on.
    style : str, optional
        Plotting style (See matplotlib plotting styles).
    density : bool, optional
        If False, the result will contain the number of samples in
        each bin. If True, the result is the value of the probability
        density function at the bin, normalized such that the integral
        over the range is 1. (See `np.histogram()` for reference)


    Returns
    -------
    plt.Axis
        Matplotlib axis containing the plot.

    """

    hist, bin_edges = calc.period_histogram(
        spike_trains,
        freq=freq,
        nbins=nbins,
        density=density
    )



    if axis is None:
        import matplotlib.pyplot as plt
        axis = plt.gca(polar=True)


    axis.fill(
        bin_edges[:-1],
        hist,
        style,
        **kwargs
    )

    # axis.plot(
    #     bin_edges[:-1],
    #     hist,
    #     style,
    #     **kwargs
    # )


    axis.set_xlabel("Stimulus Phase")
    # axis.set_ylabel("Probability Density Function")

    return axis



def plot_sac(
        spike_trains,
        coincidence_window=50e-6,
        analysis_window=5e-3,
        normalize=True,
        axis=None,
        style='k-',
        **kwargs
):
    """Plot shuffled autocorrelogram (SAC) (Joris 2006)"""

    sac, bin_edges = calc.sac(
        spike_trains,
        coincidence_window=coincidence_window,
        analysis_window=analysis_window,
        normalize=normalize
    )


    if axis is None:
        import matplotlib.pyplot as plt
        axis = plt.gca()


    axis.plot(
        bin_edges[:-1],
        sac,
        style,
        drawstyle='steps-post',
        **kwargs
    )


    axis.set_xlabel("Delay [s]")
    axis.set_ylabel("Normalized Number of Coincidences")

    return axis
