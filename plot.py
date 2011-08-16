#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import biggles

golden = 1.6180339887

def plot_raster(spike_trains, plot=None, backend='biggles', **style):
    """ Plot raster plot. """

    # Compute trial number
    L = [ len(train) for train in spike_trains ]
    r = np.arange(len(spike_trains))
    n = np.repeat(r, L)

    # Spike timings
    s = np.concatenate(tuple(spike_trains))



    if backend == 'biggles':
        import biggles
        c = biggles.Points(s, n, type='dot')
        c.style(**style)

        if plot is None:
            plot = biggles.FramedPlot()
        plot.xlabel = "Time (ms)"
        plot.ylabel = "Trial Number"
        plot.yrange = (-0.5, len(spike_trains)-0.5)
        plot.add(c)

    elif backend == 'matplotlib':
        import matplotlib.pyplot as plt
        if plot is None:
            plot = plt.gca()
        plot.plot(s, n, 'k,')
        plot.set_xlabel("Time (ms)")
        plot.set_ylabel("Trial #")
        plot.set_ylim( (-0.5, len(spike_trains)-0.5) )

    return plot


def plot_spikegram(spike_trains, bin_size=1, plot=None, **style):
    import biggles

    fs = 1000 / bin_size

    spikes = spikes_to_signal(fs, spike_trains)
    spikes = 1 - spikes/spikes.max()
    d = biggles.Density(spikes, [[0,0],[1,1]])

    if plot is None:
        plot = biggles.FramedPlot()
    plot.add(d)

    return plot


def plot_psth(spike_trains, bin_size=1, trial_num=None, plot=None, **style):
    """ Plots PSTH of spike_trains.

    spike_trains: list of spike trains
    bin_size: bin size in ms
    trial_num: total number of trials
    plot: biggles container
    **style: biggles curve style (e.g., color='red')
    """
    import biggles

    all_spikes = np.concatenate(tuple(spike_trains))

    assert len(all_spikes)>0, "No spikes!"

    nbins = np.ceil(all_spikes.max() / bin_size)

    values, bins = np.histogram(all_spikes, nbins,
                                range=(0, all_spikes.max()))


    # Normalize values for spikes per second
    if trial_num == None:
        trial_num = len(spike_trains)
    values = 1000 * values / bin_size / trial_num


    c = biggles.Histogram(values, x0=0, binsize=bin_size)
    c.style(**style)

    if plot is None:
        plot = biggles.FramedPlot()
    plot.xlabel = "Time (ms)"
    plot.ylabel = "Spikes per second"
    plot.add(c)
    plot.xrange = (0, None)
    plot.yrange = (0, None)

    return plot


def plot_isih(spike_trains, bin_size=1, trial_num=None, plot=None, **style):
    """ Plot inter-spike interval histogram. """
    import biggles

    values, bins = calc_isih(spike_trains, bin_size, trial_num)

    c = biggles.Histogram(values, x0=bins[0], binsize=bin_size)
    c.style(**style)

    if plot is None:
        plot = biggles.FramedPlot()
    plot.xlabel = "Inter-Spike Interval (ms)"
    plot.ylabel = "Interval Count"
    plot.add(c)
    plot.xrange = (0, None)
    plot.yrange = (0, None)

    return plot


def plot_period_histogram(spike_trains, fstim,
                          nbins=64, spike_fs=None,
                          center=False,
                          plot=None,
                          label=None,
                          **style):
    """ Plots period histogram. """
    import biggles

    if spike_fs is not None:
        nbins = int(spike_fs / fstim)


    fstim = fstim / 1000        # Hz -> kHz; s -> ms

    if len(spike_trains) == 0:
        return 0

    all_spikes = np.concatenate(tuple(spike_trains))

    if len(all_spikes) == 0:
        return 0

    folded = np.fmod(all_spikes, 1/fstim)
    ph,edges = np.histogram(folded, bins=nbins, range=(0,1/fstim))

    ### Normalize
    ph = ph / np.sum(ph)

    ### TODO: find the direction instead of max value
    if center:
        center_idx = ph.argmax()
        ph = np.roll(ph, nbins//2 - center_idx)

    c = biggles.Histogram(ph, x0=0, binsize=1/len(ph))
    c.style(**style)
    if label is not None:
            c.label = label


    if plot is None:
        plot = biggles.FramedPlot()
    plot.xlabel = "Normalized Phase"
    plot.ylabel = "Normalized Spike Count"
    plot.add(c)

    return plot


def plot_sac(spike_trains, coincidence_window=0.05, analysis_window=5,
             stimulus_duration=None, plot=None, **style):
    """ Plot shuffled autocorrelogram (SAC) (Joris 2006) """
    import biggles

    t, sac = calc_shuffled_autocorrelation(spike_trains, coincidence_window,
                                           analysis_window, stimulus_duration)

    c = biggles.Curve(t, sac)
    c.style(**style)

    if plot is None:
        plot = biggles.FramedPlot()
    plot.xlabel = "Delay (ms)"
    plot.ylabel = "Normalized Coincidences Count"
    plot.add(c)

    return plot


def main():
    pass

if __name__ == "__main__":
    main()
