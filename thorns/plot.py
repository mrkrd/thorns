#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import biggles
from . import spikes
from . import stats

golden = 1.6180339887

def raster(spike_trains, plot=None, backend='biggles', **style):
    """Plot raster plot."""

    trains = spike_trains['spikes']
    duration = spike_trains['duration'].max()

    # Compute trial number
    L = [ len(train) for train in trains ]
    r = np.arange(len(trains))
    n = np.repeat(r, L)

    # Spike timings
    s = np.concatenate(tuple(trains))



    if backend == 'biggles':
        c = biggles.Points(s, n, type='dot')
        c.style(**style)

        if plot is None:
            plot = biggles.FramedPlot()
        plot.xlabel = "Time [ms]"
        plot.xrange = (0, duration)
        plot.ylabel = "Trial Number"
        plot.yrange = (-0.5, len(trains)-0.5)
        plot.add(c)

    elif backend == 'matplotlib':
        import matplotlib.pyplot as plt
        if plot is None:
            plot = plt.gca()
        plot.plot(s, n, 'k,')
        plot.set_xlabel("Time [ms]")
        plot.set_xlim( (0, duration) )
        plot.set_ylabel("Trial #")
        plot.set_ylim( (-0.5, len(trains)-0.5) )

    return plot


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


def psth(spike_trains, bin_size=1e-3, plot=None, **style):
    """ Plots PSTH of spike_trains.

    spike_trains: list of spike trains
    bin_size: bin size in ms
    trial_num: total number of trials
    plot: biggles container
    **style: biggles curve style (e.g., color='red')
    """
    trains = spike_trains['spikes']
    duration = spike_trains['duration'].max()

    all_spikes = np.concatenate(tuple(trains))

    nbins = np.floor(duration / bin_size) + 1

    hist, bins = np.histogram(all_spikes,
                              bins=nbins,
                              range=(0, nbins*bin_size))


    # Normalize hist for spikes per second
    if 'trial_num' in spike_trains.dtype.names:
        trial_num = sum(spike_trains['trial_num'])
    else:
        trial_num = len(trains)

    hist =  hist / bin_size / trial_num


    c = biggles.Histogram(hist, x0=0, binsize=bin_size)
    c.style(**style)

    if plot is None:
        plot = biggles.FramedPlot()
    plot.xlabel = "Time [ms]"
    plot.ylabel = "Spikes per second"
    plot.add(c)
    plot.xrange = (0, None)
    plot.yrange = (0, None)

    return plot


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
