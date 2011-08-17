#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import random
import numpy as np


def arrays_to_trains(arrays, duration=None):
    """Convert list of arrays with spike timings to spike trains
    rec.array

    """

    if duration is None:
        duration = np.concatenate(arrays).max()

    t = []
    for a in arrays:
        t.append( (a, duration) )

    trains = np.rec.array(t, dtype=[('spikes', np.ndarray),
                                    ('duration', float)])

    return trains


def select_trains(spike_trains, **kwargs):

    trains = spike_trains
    for key,val in kwargs.items():
        trains = trains[ spike_trains[key] == val ]

    return trains

select_spike_trains = select_trains
sel = select_trains


def dicts_to_trains(dicts):
    """Convert list of dictionaries to np.rec.array

    >>> dicts = [{'a':1, 'b':2}, {'a':5, 'b':6}]
    >>> dicts_to_trains(dicts)  #doctest: +NORMALIZE_WHITESPACE
    rec.array([(1, 2), (5, 6)],
          dtype=[('a', '<i8'), ('b', '<i8')])

    """
    keys = dicts[0].keys()
    types = dicts[0].values()
    arr = []
    for d in dicts:
        assert set(d.keys()) == set(keys)
        arr.append( [d[k] for k in keys] )

    rec_arr = np.rec.array(arr, names=keys)
    return rec_arr


def _signal_to_spikes_1d(fs, signal):
    """ Convert 1D time function array into array of spike timings.

    fs: sampling frequency in Hz
    signal: input signal

    return: spike timings in ms

    >>> fs = 10
    >>> signal = np.array([0,2,0,0,1,0])
    >>> _signal_to_spikes_1d(fs, signal)
    array([ 100.,  100.,  400.])

    """
    assert signal.ndim == 1

    signal = signal.astype(int)

    t = np.arange(len(signal))
    spikes = np.repeat(t, signal) * 1000 / fs

    return spikes


def signal_to_spikes(fs, signals):
    """ Convert time functions to a list of spike trains.

    fs: samping frequency in Hz
    signals: input signals

    return: spike trains with spike timings

    >>> fs = 10
    >>> s = np.array([[0,0,0,1,0,0], [0,2,1,0,0,0]]).T
    >>> signal_to_spikes(fs, s)
    [array([ 300.]), array([ 100.,  100.,  200.])]

    """
    spike_trains = []

    if signals.ndim == 1:
        spike_trains = [ _signal_to_spikes_1d(fs, signals) ]
    elif signals.ndim == 2:
        spike_trains = [ _signal_to_spikes_1d(fs, signal)
                         for signal in signals.T ]
    else:
        assert False, "Input signal must be 1 or 2 dimensional"

    return spike_trains


def _spikes_to_signal_1d(fs, spikes, tmax=None):
    """ Convert spike train to its time function. 1D version.

    >>> fs = 10
    >>> spikes = np.array([100, 500, 1000, 1000])
    >>> _spikes_to_signal_1d(fs, spikes)
    array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2])

    """
    if tmax == None:
        tmax = np.max(spikes)

    bins = np.floor(tmax*fs/1000) + 1
    real_tmax = bins * 1000/fs
    signal, bin_edges = np.histogram(spikes, bins=bins, range=(0,real_tmax))

    return signal


def spikes_to_signal(fs, spike_trains, tmax=None):
    """ Convert spike trains to theirs time functions.

    fs: sampling frequency (Hz)
    spike_trains: trains of spikes to be converted (ms)
    tmax: length of the output signal (ms)

    return: time signal

    >>> spikes_to_signal(10, [np.array([100]), np.array([200, 300])])
    array([[0, 0],
           [1, 0],
           [0, 1],
           [0, 1]])

    """
    if tmax == None:
        tmax = max( [max(train) for train in spike_trains if len(train)>0] )

    max_len = np.ceil( tmax * fs / 1000 ) + 1
    signals = np.zeros( (max_len, len(spike_trains)) )

    signals = [_spikes_to_signal_1d(fs, train, tmax) for train in spike_trains]
    signals = np.array(signals).T

    # import matplotlib.pyplot as plt
    # plt.imshow(signals, aspect='auto')
    # plt.show()
    return signals


def accumulate_spikes(spike_trains, cfs):
    """ Concatenate spike trains of the same CF and sort by increasing CF

    >>> spikes = [np.array([1]), np.array([2]), np.array([3]), np.array([])]
    >>> cfs = np.array([2,1,2,3])
    >>> accumulate_spikes(spikes, cfs)
    ([array([2]), array([1, 3]), array([], dtype=float64)], array([1, 2, 3]))

    """
    accumulated_trains = []
    accumulated_cfs = np.unique(cfs)
    for cf in accumulated_cfs:
        selected_trains = [spike_trains[i] for i in np.where(cfs==cf)[0]]
        t = np.concatenate( selected_trains )
        accumulated_trains.append(t)

    return accumulated_trains, accumulated_cfs




def trim_spike_trains(spike_trains, start, stop=None):
    """ Return spike trains with that are between `start' and `stop'.

    >>> spikes = [np.array([1,2,3,4]), np.array([3,4,5,6])]
    >>> print trim_spikes(spikes, 2, 4)
    [array([0, 1, 2]), array([1, 2])]

    """
    all_spikes = np.concatenate(spike_trains)

    if len(all_spikes) == 0:
        return spike_trains

    if stop is None:
        stop = all_spikes.max()

    trimmed = []
    for train in spike_trains:
        t = train[(train >= start) & (train <= stop)]
        trimmed.append(t)

    shifted = shift_spikes(trimmed, -start)

    return shifted


trim = trim_spike_trains
trim_trains = trim_spike_trains


# def remove_empty(spike_trains):
#     new_trains = []
#     for train in spike_trains:
#         if len(train) != 0:
#             new_trains.append(train)
#     return new_trains



def fold_spikes(spike_trains, period):
    """ Fold each of the spike trains.

    >>> spike_trains = [np.array([1,2,3,4]), np.array([2,3,4,5])]
    >>> fold_spikes(spike_trains, 3)
    [array([1, 2]), array([0, 1]), array([2]), array([0, 1, 2])]

    >>> spike_trains = [np.array([2.]), np.array([])]
    >>> fold_spikes(spike_trains, 2)
    [array([], dtype=float64), array([ 0.]), array([], dtype=float64), array([], dtype=float64)]

    """
    all_spikes = np.concatenate(tuple(spike_trains))
    if len(all_spikes) == 0:
        return spike_trains

    max_spike = all_spikes.max()
    period_num = int(np.ceil((max_spike+1) / period))

    folded = []
    for train in spike_trains:
        for idx in range(period_num):
            lo = idx * period
            hi = (idx+1) * period
            sec = train[(train>=lo) & (train<hi)]
            sec = np.fmod(sec, period)
            folded.append(sec)

    return folded

fold = fold_spikes



def concatenate_spikes(spike_trains):
    return [np.concatenate(tuple(spike_trains))]

concatenate = concatenate_spikes
concat = concatenate_spikes


def shift_spikes(spike_trains, shift):
    shifted = [train+shift for train in spike_trains]

    return shifted

shift = shift_spikes


def split_and_fold_trains(long_train,
                          silence_duration,
                          tone_duration,
                          pad_duration,
                          remove_pads=False):
    silence = trim(long_train, 0, silence_duration)

    tones_and_pads = trim(long_train, silence_duration)
    tones_and_pads = fold(tones_and_pads, tone_duration+pad_duration)

    if remove_pads:
        tones_and_pads = trim(tones_and_pads, 0, tone_duration)

    return silence, tones_and_pads

split_and_fold = split_and_fold_trains


if __name__ == "__main__":
    import doctest

    print "Doctest start:"
    doctest.testmod()
    print "done."

    # test_shuffle_spikes()

