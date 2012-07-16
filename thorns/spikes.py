#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import random
import numpy as np

from collections import Iterable

from . import calc




def select_trains(spike_trains, **kwargs):

    mask = np.ones(len(spike_trains), dtype=bool)
    for key,val in kwargs.items():
        mask = mask & (spike_trains[key] == val)

    selected = spike_trains[mask]

    return selected


select_spike_trains = select_trains
select = select_trains
sel = select_trains





def make_trains(data, **kwargs):
    assert isinstance(data, Iterable)

    if 'fs' in kwargs:
        assert 'duration' not in kwargs


    meta = {}
    for k,v in kwargs.items():
        if k == 'fs':
            continue

        if isinstance(v, Iterable) and not isinstance(v, basestring):
            assert len(v) == len(data)
            meta[k] = v
        else:
            meta[k] = [v] * len(data)



    if isinstance(data, np.ndarray) and (data.ndim == 2) and ('fs' in kwargs):
        trains = _array_to_trains(data, kwargs['fs'], **meta)

    elif isinstance(data[0], Iterable):
        trains = _arrays_to_trains(data, **meta)


    return trains





def _arrays_to_trains(arrays, **kwargs):


    ### Meta data for each train
    meta = {}
    for k,v in kwargs.items():
        if isinstance(v, Iterable):
            assert len(v) == len(arrays)
            meta[k] = v
        else:
            meta[k] = [v] * len(arrays)


    ### Types of the metadata
    types = [('spikes', np.ndarray)]
    for key,val in meta.items():
        field_name = key

        if key == 'duration':
            field_dtype = float
        else:
            field_dtype = type(val[0])

        if field_dtype is str:
            field_shape = max( [len(s) for s in val] )
        else:
            field_shape = 1

        types.append(
            (field_name, field_dtype, field_shape)
        )



    ### Make sure we have duration
    if 'duration' not in meta:
        duration = max([np.max(a) for a in arrays if len(a)>0])
        meta['duration'] = [duration] * len(arrays)
        types.append(('duration', float))


    arrays = (np.array(a) for a in arrays)
    trains = zip(arrays, *meta.values())


    trains = np.array(
        trains,
        dtype=types
    )

    return trains



# def _dicts_to_trains(dicts):
#     """Convert list of dictionaries to np.rec.array

#     >>> dicts = [{'a':1, 'b':2}, {'a':5, 'b':6}]
#     >>> dicts_to_trains(dicts)  #doctest: +NORMALIZE_WHITESPACE
#     rec.array([(1, 2), (5, 6)],
#           dtype=[('a', '<i8'), ('b', '<i8')])

#     """
#     ### Derive types from the first dictionary
#     types = {}
#     for key,val in dicts[0].items():
#         if isinstance(val, str):
#             cnt = len(val)
#         else:
#             cnt = 1
#         types[key] = [type(val), cnt]


#     arr = []
#     for d in dicts:
#         assert set(d.keys()) == set(types.keys())

#         rec = []
#         for k,t in types.items():
#             if isinstance(d[k], str) and (len(d[k]) > t[1]):
#                 t[1] = len(d[k])

#             rec.append(d[k])

#         arr.append( tuple(rec) )


#     dt = np.dtype({'names':types.keys(), 'formats':[tuple(each) for each in types.values()]})
#     arr = np.array(arr, dtype=dt)
#     return arr





def _array_to_trains(array, fs, **kwargs):
    """ Convert time functions to a list of spike trains.

    fs: samping frequency in Hz
    a: input array

    return: spike trains with spike timings

    """
    assert array.ndim == 2

    trains = []
    for a in array.T:
        a = a.astype(int)
        t = np.arange(len(a))
        spikes = np.repeat(t, a) / fs

        trains.append( spikes )


    assert 'duration' not in kwargs

    kwargs['duration'] = len(array) / fs

    spike_trains = _arrays_to_trains(
        trains,
        **kwargs
    )

    return spike_trains








def trains_to_array(spike_trains, fs):
    """Convert spike trains to signals."""

    duration = calc.get_duration(spike_trains)

    nbins = np.ceil(duration * fs)
    tmax = nbins / fs

    signals = []
    for spikes in spike_trains['spikes']:
        signal, bin_edges = np.histogram(
            spikes,
            bins=nbins,
            range=(0, tmax)
        )
        signals.append(
            signal
        )

    signals = np.array(signals).T

    return signals






def accumulate_spike_trains(spike_trains, ignore=[]):

    """Concatenate spike trains with the same meta data. Trains will
    be sorted by the metadata.

    """

    keys = list(spike_trains.dtype.names)
    keys.remove('spikes')
    for key in ignore:
        keys.remove(key)

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

    trains = np.array(trains, dtype=dt)

    return trains



accumulate_spikes = accumulate_spike_trains
accumulate_trains = accumulate_spike_trains
accumulate = accumulate_spike_trains




def trim_spike_trains(spike_trains, start, stop):
    """Return spike trains with that are between `start' and
    `stop'.

    """

    assert start < stop

    if start is None:
        tmin = 0
    else:
        tmin = start

    if stop is None:
        tmaxs = spike_trains['duration']
    else:
        tmaxs = np.ones(len(spike_trains)) * stop


    trimmed_trains = []
    for key in spike_trains.dtype.names:
        if key == 'spikes':
            trimmed_spikes = []
            for spikes,tmax in zip(spike_trains['spikes'], tmaxs):

                spikes = spikes[ (spikes >= tmin) & (spikes <= tmax)]
                spikes -= tmin

                trimmed_spikes.append(spikes)


            trimmed_trains.append(
                trimmed_spikes
            )


        elif key == 'duration':
            durations = np.array(spike_trains['duration'])

            durations[ durations>tmaxs ] = tmaxs[ durations>tmaxs ]
            durations -= tmin

            trimmed_trains.append(durations)


        else:
            trimmed_trains.append(spike_trains[key])

    trimmed_trains = np.array(
        zip(*trimmed_trains),
        dtype=spike_trains.dtype
    )

    return trimmed_trains




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

