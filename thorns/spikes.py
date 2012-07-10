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
    ### Derive types from the first dictionary
    types = {}
    for key,val in dicts[0].items():
        if isinstance(val, str):
            cnt = len(val)
        else:
            cnt = 1
        types[key] = [type(val), cnt]


    arr = []
    for d in dicts:
        assert set(d.keys()) == set(types.keys())

        rec = []
        for k,t in types.items():
            if isinstance(d[k], str) and (len(d[k]) > t[1]):
                t[1] = len(d[k])

            rec.append(d[k])

        arr.append( tuple(rec) )


    dt = np.dtype({'names':types.keys(), 'formats':[tuple(each) for each in types.values()]})
    arr = np.array(arr, dtype=dt)
    return arr


def _signal_to_train(signal, fs):
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
    spikes = np.repeat(t, signal) * 1 / fs

    return spikes


def signal_to_trains(signal, fs):
    """ Convert time functions to a list of spike trains.

    fs: samping frequency in Hz
    signals: input signals

    return: spike trains with spike timings

    >>> fs = 10
    >>> s = np.array([[0,0,0,1,0,0], [0,2,1,0,0,0]]).T
    >>> signal_to_spikes(fs, s)
    [array([ 300.]), array([ 100.,  100.,  200.])]

    """
    duration = len(signal) / fs

    trains = []

    if signal.ndim == 1:
        trains = [ _signal_to_train(signal, fs) ]
    elif signal.ndim == 2:
        trains = [ _signal_to_train(s, fs) for s in signal.T ]
    else:
        assert False, "Input signal must be 1 or 2 dimensional"

    spike_trains = arrays_to_trains(trains, duration=duration)

    return spike_trains


def _train_to_signal(train, fs):
    """ Convert spike train to its time function. 1D version.

    >>> fs = 10
    >>> spikes = np.array([100, 500, 1000, 1000])
    >>> _train_to_signal(fs, spikes)
    array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2])

    """
    tmax = train['duration']
    spikes = train['spikes']

    bins = np.floor(tmax*fs) + 1
    real_tmax = bins * 1/fs
    signal, bin_edges = np.histogram(spikes,
                                     bins=bins,
                                     range=(0, real_tmax))

    return signal


def trains_to_signal(spike_trains, fs):
    """ Convert spike trains to theirs time functions.

    fs: sampling frequency (Hz)
    spike_trains: trains of spikes to be converted (ms)

    return: time signal

    >>> spikes_to_signal(10, [np.array([100]), np.array([200, 300])])
    array([[0, 0],
           [1, 0],
           [0, 1],
           [0, 1]])

    """
    durations = spike_trains['duration']
    assert np.all(durations == durations[0])

    signals = [_train_to_signal(train, fs) for train in spike_trains]
    signal = np.array(signals).T

    return signal


def accumulate_spike_trains(spike_trains, ignore=[]):
    """ Concatenate spike trains of the same CF and sort by increasing CF

    >>> spikes = [np.array([1]), np.array([2]), np.array([3]), np.array([])]
    >>> cfs = np.array([2,1,2,3])
    >>> accumulate_spikes(spikes, cfs)
    ([array([2]), array([1, 3]), array([], dtype=float64)], array([1, 2, 3]))

    """
    keys = list(spike_trains.dtype.names)
    keys.remove('spikes')
    for i in ignore:
        keys.remove(i)

    meta = spike_trains[keys]

    unique_meta, indices = np.unique(meta, return_inverse=True)

    trains = []
    for idx in np.unique(indices):
        selector = np.where( indices == idx )[0]

        acc_spikes = np.concatenate(tuple( spike_trains['spikes'][selector] ))
        acc_spikes.sort()

        trains.append( (acc_spikes,) + tuple(unique_meta[idx]) )

    dt = [('spikes', np.ndarray)] + [(name,unique_meta.dtype[name])
                                     for name in unique_meta.dtype.names]

    trains = np.rec.array(trains, dtype=dt)

    return trains



accumulate_spikes = accumulate_spike_trains
accumulate_trains = accumulate_spike_trains
accumulate = accumulate_spike_trains




def trim_spike_trains(spike_trains, *args):
    """ Return spike trains with that are between `start' and `stop'.

    >>> spikes = [np.array([1,2,3,4]), np.array([3,4,5,6])]
    >>> print trim_spikes(spikes, 2, 4)
    [array([0, 1, 2]), array([1, 2])]

    """
    if len(args) == 1 and isinstance(args[0], tuple):
        start, stop = args[0]
    elif len(args) == 1:
        start = args[0]
        stop = None
    elif len(args) == 2:
        start, stop = args
    else:
        assert False, "(start, stop)"

    arrays = []
    for key in spike_trains.dtype.names:
        if key == 'spikes':
            arrays.append(
                _trim_arrays(spike_trains['spikes'],
                             spike_trains['duration'],
                             start,
                             stop)
                )
        elif key == 'duration':
            tmax = np.array(spike_trains['duration'])

            if stop is not None:
                tmax[ tmax>stop ] = stop

            new_durs = tmax - start
            new_durs[ new_durs<0 ] = 0

            arrays.append(new_durs)
        else:
            arrays.append(spike_trains[key])

    trimmed = np.rec.array(zip(*arrays), dtype=spike_trains.dtype)

    return trimmed


def _trim_arrays(arrays, durations, start, stop):
    """Trim spike trains to (start, stop)"""

    tmin = start

    if stop is None:
        tmaxs = durations
    else:
        tmaxs = [stop for i in range(len(arrays))]

    trimmed = []
    for arr,tmax in zip(arrays,tmaxs):
        a = arr[ (arr >= tmin) & (arr <= tmax) ]
        a = a - tmin
        trimmed.append(a)

    return trimmed


trim = trim_spike_trains
trim_trains = trim_spike_trains


# def remove_empty(spike_trains):
#     new_trains = []
#     for train in spike_trains:
#         if len(train) != 0:
#             new_trains.append(train)
#     return new_trains



def fold_spike_trains(spike_trains, period):
    """ Fold each of the spike trains.

    >>> from thorns import arrays_to_trains
    >>> a = [np.array([1,2,3,4]), np.array([2,3,4,5])]
    >>> spike_trains = arrays_to_trains(a, duration=9)
    >>> fold_spike_trains(spike_trains, 3)
    [array([1, 2]), array([0, 1]), array([2]), array([0, 1, 2])]

    # >>> spike_trains = [np.array([2.]), np.array([])]
    # >>> fold_spike_trains(spike_trains, 2)
    # [array([], dtype=float64), array([ 0.]), array([], dtype=float64), array([], dtype=float64)]

    """
    assert np.all(spike_trains['duration'] == spike_trains['duration'][0])
    duration = spike_trains['duration'][0]

    print "fold_spike_trains() need update: does not copy metadata!"
    period_num = int( np.ceil(duration / period) )

    last_period = np.fmod(duration, period)

    folded = []
    for train in spike_trains['spikes']:
        for idx in range(period_num):
            lo = idx * period
            hi = (idx+1) * period
            sec = train[(train>=lo) & (train<hi)]
            sec = np.fmod(sec, period)
            folded.append(sec)

    folded_trains = arrays_to_trains(folded, duration=period)

    if last_period != 0:
        folded_trains[-1]['duration'] = last_period

    return folded_trains

fold = fold_spike_trains
fold_trains = fold_spike_trains



def concatenate_spikes(spike_trains):
    return [np.concatenate(tuple(spike_trains))]

concatenate = concatenate_spikes
concat = concatenate_spikes


def shift_spikes(spike_trains, shift):
    shifted = [train+shift for train in spike_trains]

    return shifted

shift = shift_spikes


def split_and_fold_trains(spike_trains,
                          silence_duration,
                          tone_duration,
                          pad_duration,
                          remove_pads):
    silence = trim(spike_trains, 0, silence_duration)


    tones_and_pads = trim(spike_trains, silence_duration)
    tones_and_pads = fold(tones_and_pads, tone_duration+pad_duration)

    # import plot
    # plot.raster(tones_and_pads).show()

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

    # arr = [
    #     {'spikes': np.arange(10),
    #      'cf': 2,
    #      'bla': 'a'},
    #     {'spikes': np.arange(7),
    #      'cf': 4,
    #      'bla': 'bb'}
    # ]
    # print dicts_to_trains(arr)

