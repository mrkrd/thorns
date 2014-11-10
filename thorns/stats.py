#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

__author__ = "Marek Rudnicki"

import numpy as np
import pandas as pd
import warnings


def get_duration(spike_trains):
    """Return the common duration of `spike_trains`."""

    duration = spike_trains['duration'].unique()

    if len(duration) != 1:
        raise ValueError("The duration values are not all the same: {}".format(duration))

    return duration[0]




def psth(spike_trains, bin_size, normalize=True, **kwargs):
    """Calculate peristimulus time histogram (PSTH).


    Returns
    -------
    array_like
        Histogram values.
    array_like
        Bin edges.

    """
    all_spikes = np.concatenate(tuple(spike_trains['spikes']))

    duration = get_duration(spike_trains)
    nbins = np.ceil(duration / bin_size)

    if nbins == 0:
        return None, None

    hist, bin_edges = np.histogram(
        all_spikes,
        bins=nbins,
        range=(0, nbins*bin_size),
        **kwargs
    )

    if normalize:
        psth = hist / bin_size / len(spike_trains)
    else:
        psth = hist


    return psth, bin_edges




def isih(spike_trains, bin_size, **kwargs):
    """Calculate inter-spike interval histogram (ISIH).

    Returns
    -------
    array_like
        Histogram values.
    array_like
        Bin edges.

    """
    isis = np.concatenate(
        tuple( np.diff(train) for train in spike_trains['spikes'] )
    )

    if len(isis) == 0:
        return None, None

    nbins = np.ceil(np.max(isis) / bin_size)

    hist, bin_edges = np.histogram(
        isis,
        bins=nbins,
        range=(0, nbins*bin_size),
        **kwargs
    )

    return hist, bin_edges




def entrainment(spike_trains, freq, bin_size=1e-3):
    """Calculate entrainment of `spike_trains` in response to periodic
    stimulus (`freq` Hz).

    """

    hist, bin_edges = isih(
        spike_trains,
        bin_size=bin_size
    )

    if hist is None:
        return np.nan


    stim_period = 1 / freq

    ent_win = (
        (bin_edges[:-1] > 0.5*stim_period)
        &
        (bin_edges[:-1] < 1.5*stim_period)
    )

    entrainment = np.sum(hist[ent_win]) / np.sum(hist)

    return entrainment



def vector_strength(spike_trains, freq):
    """Calculate vector strength of `spike_trains` in response to periodic
    stimulus (`freq` Hz).

    """
    if isinstance(spike_trains, pd.Series):
        spike_trains = pd.DataFrame(spike_trains).T

    all_spikes = np.concatenate( tuple(spike_trains['spikes']) )

    rate = firing_rate(spike_trains)

    if (rate < 10) or (len(all_spikes) < 8):
        warnings.warn("Spike count ({count}) or firing rate ({rate}) too small to reliably calculate vector strength.".format(rate=rate, count=len(all_spikes)))
        return np.nan


    folded = np.fmod(all_spikes, 1/freq)

    angles = folded * freq * 2 * np.pi

    vectors = np.exp(1j*angles)

    r = np.abs(np.mean(vectors))

    return r




# def shuffle_spikes(spike_trains):
#     """ Get input spikes.  Randomly permute inter spikes intervals.
#     Return new spike trains.

#     """
#     new_trains = []
#     for train in spike_trains:
#         isi = np.diff(np.append(0, train)) # Append 0 in order to vary
#                                            # the onset
#         shuffle(isi)
#         shuffled_train = np.cumsum(isi)
#         new_trains.append(shuffled_train)

#     return new_trains


# def test_shuffle_spikes():
#     print "test_shuffle_spikes():"
#     spikes = [np.array([2, 3, 4]),
#               np.array([1, 3, 6])]

#     print spikes
#     print shuffle_spikes(spikes)



def firing_rate(spike_trains):
    """Calculates average firing rate of neurons."""

    if isinstance(spike_trains, pd.Series):
        spike_trains = pd.DataFrame(spike_trains).T

    if len(spike_trains) == 0:
        return np.nan

    duration = np.sum( spike_trains['duration'] )

    trains = spike_trains['spikes']
    spike_num = np.concatenate(tuple(trains)).size

    rate = spike_num / duration

    return rate





def spike_count(spike_trains):
    """Count all spikes in `spike_trains`."""
    all_spikes = np.concatenate(tuple(spike_trains['spikes']))
    return len(all_spikes)



def correlation_index(
        spike_trains,
        coincidence_window=50e-6,
        normalize=True):
    """Compute correlation index from `spike_trains` as described in
    [Joris2006]_.


    References
    ----------

    .. [Joris2006] Joris, P. X., Louage, D. H., Cardoen, L., & van der
       Heijden, M. (2006). Correlation index: a new metric to quantify
       temporal coding. Hearing research, 216, 19-30.

    """

    if len(spike_trains) == 0:
        return 0

    all_spikes = np.concatenate(tuple(spike_trains['spikes']))

    if len(all_spikes) == 0:
        return 0


    Nc = 0                      # Total number of coincidences

    for spike in all_spikes:
        hits = all_spikes[
            (all_spikes >= spike) &
            (all_spikes <= spike+coincidence_window)
        ]
        Nc += len(hits) - 1



    if normalize:
        trial_num = len(spike_trains)
        rate = firing_rate(spike_trains)
        duration = get_duration(spike_trains)

        norm = trial_num*(trial_num-1) * rate**2 * coincidence_window * duration

        ci = Nc / norm

    else:
        ci = Nc


    return ci





def shuffled_autocorrelogram(
        spike_trains,
        coincidence_window=50e-6,
        analysis_window=5e-3,
        normalize=True):
    """Calculate shuffled autocorrelogram (SAC) of `spike_trains` as
    described in [Joris2006]_.



    Returns
    -------
    ndarray
        Histogram values.
    ndarray
        Bin edges.


    References
    ----------

    .. [Joris2006] Joris, P. X., Louage, D. H., Cardoen, L., & van der
       Heijden, M. (2006). Correlation index: a new metric to quantify
       temporal coding. Hearing research, 216, 19-30.

    """

    duration = get_duration(spike_trains)
    trains = spike_trains['spikes']
    trial_num = len(trains)

    nbins = np.ceil(analysis_window / coincidence_window)

    cum = []
    for i,train in enumerate(trains):
        other_trains = list(trains)
        other_trains.pop(i)
        almost_all_spikes = np.concatenate(other_trains)

        for spike in train:
            centered = almost_all_spikes - spike
            trimmed = centered[
                (centered >= 0) & (centered < nbins*coincidence_window)
            ]
            cum.append(trimmed)

    cum = np.concatenate(cum)

    hist, bin_edges = np.histogram(
        cum,
        bins=nbins,
        range=(0, nbins*coincidence_window)
    )

    if normalize:
        firing_rate = firing_rate(spike_trains)
        norm = trial_num*(trial_num-1) * firing_rate**2 * coincidence_window * duration
        sac_half = hist / norm
    else:
        sac_half = hist

    sac = np.concatenate( (sac_half[::-1][0:-1], sac_half) )
    bin_edges = np.concatenate( (-bin_edges[::-1][1:-1], bin_edges) )


    return sac, bin_edges






def period_histogram(
        spike_trains,
        freq,
        nbins=64,               # int(spike_fs / freq)
        **kwargs
):
    """Calculate period histogram of `spike_trains` to the periodic
    stimulus (`ferq` Hz).


    Returns
    -------
    ndarray
        Histogram values.
    ndarray
        Bin edges.

    """

    all_spikes = np.concatenate( tuple(spike_trains['spikes']) )
    folded = np.fmod(all_spikes, 1/freq)
    normalized = folded * freq * 2 * np.pi

    hist, bin_edges = np.histogram(
        normalized,
        bins=nbins,
        range=(0, 2*np.pi),
        **kwargs
    )

    return hist, bin_edges
