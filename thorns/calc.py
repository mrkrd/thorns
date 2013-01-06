#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np


def get_duration(spike_trains):
    duration = spike_trains['duration'].values[0]
    assert np.all(spike_trains['duration'] == duration)

    return duration




def calc_psth(spike_trains, bin_size, normalize=True):

    duration = get_duration(spike_trains)
    trial_num = len(spike_trains)

    trains = spike_trains['spikes']
    all_spikes = np.concatenate(tuple(trains))

    nbins = np.ceil(duration / bin_size)

    if nbins == 0:
        return None, None

    hist, bin_edges = np.histogram(
        all_spikes,
        bins=nbins,
        range=(0, nbins*bin_size)
    )


    if normalize:
        psth = hist / bin_size / trial_num
    else:
        psth = hist


    return psth, bin_edges




def calc_isih(spike_trains, bin_size, normalize=True):
    """Calculate inter-spike interval histogram."""

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
        normed=normalize
    )

    return hist, bin_edges




def calc_entrainment(spike_trains, freq, bin_size=1e-3):
    """Calculate entrainment."""

    hist, bin_edges = calc_isih(
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



def calc_synchronization_index(spike_trains, freq):
    """Calculate synchronization index aka vector strength."""

    all_spikes = np.concatenate( tuple(spike_trains['spikes']) )

    if len(all_spikes) == 0:
        return np.nan


    folded = np.fmod(all_spikes, 1/freq)
    ph, edges = np.histogram(
        folded,
        bins=1000,
        range=(0, 1/freq)
    )

    # indexing trick is necessary, because the sample at 2*pi belongs
    # to the next cycle
    x = np.cos(np.linspace(0, 2*np.pi, len(ph)+1))[0:-1]
    y = np.sin(np.linspace(0, 2*np.pi, len(ph)+1))[0:-1]

    xsum2 = (np.sum(x*ph))**2
    ysum2 = (np.sum(y*ph))**2

    r = np.sqrt(xsum2 + ysum2) / np.sum(ph)

    return r


calc_si = calc_synchronization_index



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


def calc_firing_rate(spike_trains):
    """Calculates average firing rate."""

    duration = np.sum( spike_trains['duration'] )

    trains = spike_trains['spikes']
    spike_num = np.concatenate(tuple(trains)).size

    rate = spike_num / duration

    return rate


calc_rate = calc_firing_rate



def count_spikes(spike_trains):
    all_spikes = np.concatenate(tuple(spike_trains['spikes']))
    return len(all_spikes)

count = count_spikes



def calc_correlation_index(
        spike_trains,
        coincidence_window=50e-6,
        normalize=True):
    """Compute correlation index (Joris 2006)"""

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
        firing_rate = calc_firing_rate(spike_trains)
        duration = get_duration(spike_trains)

        norm = trial_num*(trial_num-1) * firing_rate**2 * coincidence_window * duration

        ci = Nc / norm

    else:
        ci = Nc


    return ci


calc_ci = calc_correlation_index


def calc_shuffled_autocorrelogram(
        spike_trains,
        coincidence_window=50e-6,
        analysis_window=5e-3,
        normalize=True):
    """Calculate Shuffled Autocorrelogram (Joris 2006)"""

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
        firing_rate = calc_firing_rate(spike_trains)
        norm = trial_num*(trial_num-1) * firing_rate**2 * coincidence_window * duration
        sac_half = hist / norm
    else:
        sac_half = hist

    sac = np.concatenate( (sac_half[::-1][0:-1], sac_half) )
    bin_edges = np.concatenate( (-bin_edges[::-1][1:-1], bin_edges) )


    return sac, bin_edges


calc_sac = calc_shuffled_autocorrelogram




def calc_period_histogram(
        spike_trains,
        freq,
        nbins=64,               # int(spike_fs / freq)
        normalize=True):


    all_spikes = np.concatenate( tuple(spike_trains['spikes']) )
    folded = np.fmod(all_spikes, 1/freq)
    normalized = folded * freq

    hist, bin_edges = np.histogram(
        normalized,
        bins=nbins,
        range=(0, 1),
        normed=normalize
    )


    return hist, bin_edges





def main():
    import doctest

    print "Doctest start:"
    doctest.testmod()
    print "done."


if __name__ == "__main__":
    main()
