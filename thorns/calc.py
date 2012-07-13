#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np


def get_duration(spike_trains):
    assert np.all(spike_trains['duration'] == spike_trains['duration'][0])
    duration = spike_trains['duration'][0]

    return duration



def calc_isih(spike_trains, bin_size=1e-3):
    """ Calculate inter-spike interval histogram.

    >>> from thorns import arrays_to_trains
    >>> spikes = [np.array([1,2,3]), np.array([2,5,8])]
    >>> spike_trains = arrays_to_trains(spikes)
    >>> isih(spike_trains)
    array([ 0. ,  0.5,  0. ,  0.5])

    """
    trains = spike_trains['spikes']

    if 'trial_num' in spike_trains.dtype.names:
        trial_num = sum(spike_trains['trial_num'])
    else:
        trial_num = len(trains)

    isi_trains = [ np.diff(train) for train in trains ]

    all_isi = np.concatenate(isi_trains)

    if len(all_isi) == 0:
        return np.array([])

    nbins = np.floor(all_isi.max() / bin_size) + 1

    hist, bins = np.histogram(all_isi,
                              bins=nbins,
                              range=(0, nbins*bin_size),
                              normed=True)

    return hist


def entrainment(spike_trains, fstim, bin_size=1e-3):
    """ Calculate entrainment of spike_trains.

    >>> from thorns import arrays_to_trains
    >>> spikes = [np.array([2, 4, 6]), np.array([0, 5, 10])]
    >>> spike_trains = arrays_to_trains(spikes)
    >>> entrainment(spike_trains, fstim=500)
    0.5
    """
    hist = isih(spike_trains, bin_size=bin_size)

    if len(hist) == 0:
        return 0

    bins = np.arange( len(hist) ) * bin_size

    stim_period = 1/fstim    # ms

    entrainment_window = (bins > stim_period/2) & (bins < stim_period*3/2)

    entrainment =  np.sum(hist[entrainment_window]) / np.sum(hist)

    return entrainment



def synchronization_index(spike_trains, fstim):
    """ Calculate Synchronization Index.

    spike_trains: list of arrays of spiking times
    fstim: stimulus frequency in Hz

    return: synchronization index

    >>> fs = 36000.0
    >>> fstim = 100.0

    >>> test0 = [np.arange(0, 0.1, 1/fs)*1000, np.arange(0, 0.1, 1/fs)*1000]
    >>> si0 = calc_synchronization_index(test0, fstim)
    >>> si0 < 1e-4   # numerical errors
    True

    >>> test1 = [np.zeros(fs)]
    >>> si1 = calc_synchronization_index(test1, fstim)
    >>> si1 == 1
    True
    """

    if len(spike_trains) == 0:
        return 0

    all_spikes = np.concatenate( tuple(spike_trains['spikes']) )

    if len(all_spikes) == 0:
        return 0

    all_spikes = all_spikes - all_spikes.min()

    folded = np.fmod(all_spikes, 1/fstim)
    ph,edges = np.histogram(folded, bins=1000, range=(0, 1/fstim))


    # indexing trick is necessary, because the sample at 2*pi belongs
    # to the next cycle
    x = np.cos(np.linspace(0, 2*np.pi, len(ph)+1))[0:-1]
    y = np.sin(np.linspace(0, 2*np.pi, len(ph)+1))[0:-1]

    xsum2 = (np.sum(x*ph))**2
    ysum2 = (np.sum(y*ph))**2

    r = np.sqrt(xsum2 + ysum2) / np.sum(ph)

    return r


si = synchronization_index
vector_strength = synchronization_index
vs = synchronization_index



def _raw_correlation_index(spike_trains, window_len=0.05):
    """ Computes unnormalized correlation index. (Joris et al. 2006)


    >>> trains = [np.array([1, 2]), np.array([1.03, 2, 3])]
    >>> _raw_correlation_index(trains)
    3

    """
    all_spikes = np.concatenate(tuple(spike_trains))

    Nc = 0                      # Total number of coincidences

    for spike in all_spikes:
        hits = all_spikes[(all_spikes >= spike) &
                          (all_spikes <= spike+window_len)]
        Nc += len(hits) - 1

    return Nc


def shuffle_spikes(spike_trains):
    """ Get input spikes.  Randomly permute inter spikes intervals.
    Return new spike trains.

    """
    new_trains = []
    for train in spike_trains:
        isi = np.diff(np.append(0, train)) # Append 0 in order to vary
                                           # the onset
        shuffle(isi)
        shuffled_train = np.cumsum(isi)
        new_trains.append(shuffled_train)

    return new_trains


def test_shuffle_spikes():
    print "test_shuffle_spikes():"
    spikes = [np.array([2, 3, 4]),
              np.array([1, 3, 6])]

    print spikes
    print shuffle_spikes(spikes)


def calc_firing_rate(spike_trains):
    """ Calculates average firing rate.

    spike_trains: trains of spikes
    stimulus_duration: in ms, if None, then calculated from spike timeings

    return: average firing rate in spikes per second (Hz)

    >>> from thorns import arrays_to_trains
    >>> a = [np.arange(20), np.arange(10)]
    >>> spike_trains = arrays_to_trains(a, duration=1000)
    >>> average_firing_rate(spike_trains)
    15.0

    """
    duration = np.sum( spike_trains['duration'] )

    trains = spike_trains['spikes']
    spike_num = np.concatenate(tuple(trains)).size

    r = spike_num / duration

    return r


calc_rate = calc_firing_rate


def count_spikes(spike_trains):
    all_spikes = np.concatenate(tuple(spike_trains))
    return len(all_spikes)

count = count_spikes
calc_spike_count = count_spikes


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

    cum = []
    for i in range(len(trains)):
        other_trains = list(trains)
        train = other_trains.pop(i)
        almost_all_spikes = np.concatenate(other_trains)

        for spike in train:
            centered = almost_all_spikes - spike
            trimmed = centered[
                (centered > -analysis_window) & (centered < analysis_window)
            ]
            cum.append(trimmed)

    cum = np.concatenate(cum)

    hist, bin_edges = np.histogram(
        cum,
        bins=np.floor(2*analysis_window/coincidence_window)+1,
        range=(-analysis_window, analysis_window)
    )

    if normalize:
        firing_rate = calc_firing_rate(spike_trains)
        norm = trial_num*(trial_num-1) * firing_rate**2 * coincidence_window * duration
        sac = hist / norm
    else:
        sac = hist

    t = bin_edges[0:-1] + (bin_edges[1] - bin_edges[0])

    print hist
    print bin_edges

    return sac, t


calc_sac = calc_shuffled_autocorrelogram









def main():
    import doctest

    print "Doctest start:"
    doctest.testmod()
    print "done."


if __name__ == "__main__":
    main()
