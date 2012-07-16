#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import matplotlib.pyplot as plt

from . import spikes
from . import calc

golden = 1.6180339887

def plot_raster(spike_trains, axis=None, style='k,', **kwargs):
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
        axis = plt.gca()

    axis.plot(s, n, style, **kwargs)
    axis.set_xlabel("Time [ms]")
    axis.set_xlim( (0, duration) )
    axis.set_ylabel("Trial #")
    axis.set_ylim( (-0.5, len(trains)-0.5) )


    return axis


def spike_signal(spike_trains, bin_size=1, plot=None, **style):
    assert False, "not implemented"
    fs = 1 / bin_size

    spikes = spikes_to_signal(fs, spike_trains)
    spikes = 1 - spikes/spikes.max()
    d = biggles.Density(spikes, [[0,0],[1,1]])

    if plot is None:
        plot = biggles.FramedPlot()
    plot.add(d)

    return plot


def plot_psth(spike_trains, bin_size, axis=None, **kwargs):
    """Plots PSTH of spike_trains."""


    psth, bin_edges = calc.calc_psth(
        spike_trains,
        bin_size
    )


    if axis is None:
        plot = plt.gca()


    axis.plot(
        bin_edges[:-1],
        psth,
        drawstyle='steps-post',
        **kwargs
    )


    axis.set_xlabel("Time [ms]")
    axis.set_ylabel("Spikes per second")


    return axis


def isih(spike_trains, bin_size=1e-3, plot=None, **style):
    """Plot inter-spike interval histogram."""

    hist = stats.isih(spike_trains, bin_size)

    c = biggles.Histogram(hist, x0=0, binsize=bin_size)
    c.style(**style)

    if plot is None:
        plot = biggles.FramedPlot()
    plot.xlabel = "Inter-Spike Interval [ms]"
    plot.ylabel = "Probability Density Function"
    plot.add(c)
    plot.xrange = (0, None)
    plot.yrange = (0, None)

    return plot


def period_histogram(spike_trains,
                     stimulus_freq,
                     nbins=64,
                     spike_fs=None,
                     center=False,
                     plot=None,
                     label=None,
                     **style):
    """Plots period histogram."""

    trains = spike_trains['spikes']

    # Align bins to sampling frequency, if given
    if spike_fs is not None:
        nbins = int(spike_fs / stimulus_freq)

    all_spikes = np.concatenate(tuple(trains))
    folded = np.fmod(all_spikes, 1/stimulus_freq)
    normalized = folded * stimulus_freq

    print "Make sure that np.histogram get the right number of bins"

    hist, edges = np.histogram(
        normalized,
        bins=nbins,
        range=(0, 1),
        normed=True
    )

    ### TODO: find the direction instead of max value
    if center:
        center_idx = hist.argmax()
        hist = np.roll(hist, nbins//2 - center_idx)

    c = biggles.Histogram(hist, x0=0, binsize=1/len(hist))

    c.style(**style)
    if label is not None:
        c.label = label


    if plot is None:
        plot = biggles.FramedPlot()
    plot.xlabel = "Normalized Phase"
    plot.ylabel = "Probability Density Function"
    plot.add(c)

    return plot


def sac(spike_trains,
        coincidence_window=0.05,
        analysis_window=5,
        stimulus_duration=None,
        plot=None,
        **style):
    """Plot shuffled autocorrelogram (SAC) (Joris 2006)"""
    import biggles

    t, sac = stats.shuffled_autocorrelation(spike_trains,
                                            coincidence_window,
                                            analysis_window,
                                            stimulus_duration)

    c = biggles.Curve(t, sac)
    c.style(**style)

    if plot is None:
        plot = biggles.FramedPlot()
    plot.xlabel = "Delay [ms]"
    plot.ylabel = "Normalized Coincidences Count"
    plot.add(c)

    return plot


def main():
    pass

if __name__ == "__main__":
    main()