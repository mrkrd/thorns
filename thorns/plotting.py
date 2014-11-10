#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"


import numpy as np

from . import spikes
from . import stats


def plot_neurogram(spike_trains, fs, ax=None, **kwargs):
    """Visualize `spike_trains` by converting them to bit map and plot
    using `plt.imshow()`.  Set `fs` reasonably in order to avoid
    aliasing effects.

    For smaller number of spike trains, it's usually better to use
    `plot_raster`.

    """

    neurogram = spikes.trains_to_array(
        spike_trains,
        fs
    )

    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    extent = (
        0,                       # left
        neurogram.shape[0] / fs, # right
        0,                       # bottom
        neurogram.shape[1]       # top
    )

    ax.imshow(
        neurogram.T,
        aspect='auto',
        extent=extent,
        **kwargs
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel number")

    return ax




def plot_raster(spike_trains, ax=None, style='k.', **kwargs):
    """Plot raster plot."""

    trains = spike_trains['spikes']
    duration = np.max( spike_trains['duration'] )

    # Compute trial number
    L = [ len(train) for train in trains ]
    r = np.arange(len(trains))
    n = np.repeat(r, L)

    # Spike timings
    s = np.concatenate(tuple(trains))


    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    ax.plot(s, n, style, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_xlim( (0, duration) )
    ax.set_ylabel("Train Number")
    ax.set_ylim( (-0.5, len(trains)-0.5) )


    return ax





def plot_psth(
        spike_trains,
        bin_size,
        ax=None,
        drawstyle='steps-post',
        **kwargs
):
    """Plots PSTH of spike_trains."""


    psth, bin_edges = stats.psth(
        spike_trains,
        bin_size
    )


    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()


    ax.plot(
        bin_edges[:-1],
        psth,
        drawstyle=drawstyle,
        **kwargs
    )


    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spikes per Second")


    return ax





def plot_isih(
        spike_trains,
        bin_size,
        ax=None,
        drawstyle='steps-post',
        density=True,
        **kwargs
):
    """Plot inter-spike interval histogram.

    """
    hist, bin_edges = stats.isih(
        spike_trains=spike_trains,
        bin_size=bin_size,
        density=density,
    )

    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()


    ax.plot(
        bin_edges[:-1],
        hist,
        drawstyle=drawstyle,
        **kwargs
    )


    ax.set_xlabel("Inter-Spike Interval (s)")

    if density:
        ax.set_ylabel("Probability Density Function")
    else:
        ax.set_ylabel("Interval Count per Bin")


    return ax






def plot_period_histogram(
        spike_trains,
        freq,
        nbins=64,
        shift=0,
        ax=None,
        style='',
        density=False,
        drawstyle='steps-post',
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
    shift : float
        Defines how much should the phase be shifted in the plot.
    ax : plt.Ax, optional
        Matplotlib Ax to plot on.
    style : str, optional
        Plotting style (See matplotlib plotting styles).
    density : bool, optional
        If False, the result will contain the number of samples in
        each bin. If True, the result is the value of the probability
        density function at the bin, normalized such that the integral
        over the range is 1. (See `np.histogram()` for reference)
    drawstyle : {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}
        Set the drawstyle of the plot.

    Returns
    -------
    plt.Axis
        Matplotlib axis containing the plot.

    """

    hist, bin_edges = stats.period_histogram(
        spike_trains,
        freq=freq,
        nbins=nbins,
        density=density
    )


    shift_samp = int(np.round( (shift*nbins) / (2*np.pi) ))
    hist = np.roll(hist, shift_samp)


    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca(polar=True)


    ax.plot(
        bin_edges[:-1],
        hist,
        style,
        drawstyle=drawstyle,
        **kwargs
    )

    # ax.plot(
    #     bin_edges[:-1],
    #     hist,
    #     style,
    #     **kwargs
    # )

    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels([0, r"$\pi$", r"2$\pi$"])

    ax.set_xlabel("Stimulus phase")

    if density:
        ax.set_ylabel("Probability density function")
    else:
        ax.set_ylabel("Spike count")

    return ax



def plot_sac(
        spike_trains,
        coincidence_window=50e-6,
        analysis_window=5e-3,
        normalize=True,
        ax=None,
        style='k-',
        **kwargs
):
    """Plot shuffled autocorrelogram (SAC) (Joris 2006)"""

    sac, bin_edges = stats.sac(
        spike_trains,
        coincidence_window=coincidence_window,
        analysis_window=analysis_window,
        normalize=normalize
    )


    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()


    ax.plot(
        bin_edges[:-1],
        sac,
        style,
        drawstyle='steps-post',
        **kwargs
    )


    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Normalized Number of Coincidences")

    return ax



def plot_signal(signal, fs=None, ax=None, style='', **kwargs):
    """Plot time signal.

    Parameters
    ----------
    signal : array_like
        Time signal.
    fs : float, optional
        Sampling freuency of the signal.
    ax : plt.Axis, optional
        Axis to plot onto.
    style : str, optional
        Plotting style string.

    Returns
    -------
    plt.Axis
       Matplotlib Axis with the plot.

    """

    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()


    if fs is None:
        fs = 1
        xlabel = "Sample Number"
    else:
        xlabel = "Time (s)"


    t = np.arange(len(signal)) / fs


    ax.set_xlim((t[0],t[-1]))

    ax.plot(t, signal, style, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")

    return ax




def show():
    """Equivalent of plt.show()"""
    import matplotlib.pyplot as plt
    plt.show()


def gcf():
    """Equivalent of plt.gcf()"""
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    return fig
