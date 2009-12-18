# Author: Marek Rudnicki
# Time-stamp: <2009-12-18 00:32:12 marek>
#
# Description: pyThorns -- spike analysis software for Python

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import shuffle

import waves

def signal_to_spikes_1D(fs, signal):
    """
    Convert 1D time function array into array of spike timings.

    fs: sampling frequency in Hz
    signal: input signal

    return: spike timings in ms

    >>> fs = 10
    >>> signal = np.array([0,2,0,0,1,0])
    >>> signal_to_spikes_1D(fs, signal)
    array([ 100.,  100.,  400.])
    """
    assert signal.ndim == 1
    assert (np.mod(signal, 1) == 0).all()

    signal = signal.astype(int)

    t = np.arange(len(signal))
    spikes = np.repeat(t, signal) * 1000 / fs

    return spikes


def signal_to_spikes(fs, signals):
    """
    Convert time functions to a list of spike trains.

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
        spike_trains = [ signal_to_spikes_1D(fs, signals) ]
    elif signals.ndim == 2:
        spike_trains = [ signal_to_spikes_1D(fs, signal)
                         for signal in signals.T ]
    else:
        assert False

    return spike_trains





def spikes_to_signal_1D(fs, spikes, tmax=None):
    """
    Convert spike train to its time function. 1D version.

    >>> fs = 10
    >>> spikes = np.array([100, 500, 1000, 1000])
    >>> spikes_to_signal_1D(fs, spikes)
    array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2])
    """
    if tmax == None:
        tmax = max(spikes)

    bins = np.ceil(tmax*fs/1000) + 1
    signal, bin_edges = np.histogram(spikes, bins=bins, range=(0,tmax))

    return signal



def spikes_to_signal(fs, spike_trains, tmax=None):
    """
    Convert spike trains to theirs time functions.

    fs: sampling frequency (Hz)
    spike_trains: trains of spikes to be converted (ms)
    tmax: length of the output signal (ms)

    return: time signal

    >>> spikes_to_signal(10, [np.array([100]), np.array([200, 300])])
    array([[ 0.,  0.],
           [ 1.,  0.],
           [ 0.,  1.],
           [ 0.,  1.]])
    """
    if tmax == None:
        tmax = max( [max(train) for train in spike_trains] )

    max_len = np.ceil( tmax * fs / 1000 ) + 1
    signals = np.zeros( (max_len, len(spike_trains)) )

    for i,train in enumerate(spike_trains):
        s = spikes_to_signal_1D(fs, train, tmax)
        signals[:,i] = s

    return signals


# TODO: def is_spike_train()

def plot_raster(spike_trains, axis=None, **kwargs):
    """
    Plot raster plot.
    """

    # Compute trial number
    L = [ len(train) for train in spike_trains ]
    r = np.arange(len(spike_trains))
    n = np.repeat(r, L)

    # Spike timings
    s = np.concatenate(tuple(spike_trains))


    if axis == None:
        axis = plt.gca()
        axis.plot(s, n, 'k,', **kwargs)
        axis.set_xlabel("Time [ms]")
        axis.set_ylabel("Trial #")
        plt.show()
    else:
        axis.plot(s, n, 'k,', **kwargs)
        axis.set_xlabel("Time [ms]")
        axis.set_ylabel("Trial #")


def plot_psth(spike_trains, bin_size=1, trial_num=None, axis=None, **kwargs):
    """ Plots PSTH of spike_trains.

    spike_trains: list of spike trains
    bin_size: bin size in ms
    trial_num: total number of trials
    axis: axis to draw on
    **kwargs: plt.plot arguments
    """
    all_spikes = np.concatenate(tuple(spike_trains))

    nbins = np.ceil((max(all_spikes) - min(all_spikes)) / bin_size)

    values, bins = np.histogram(all_spikes, nbins)

    # Normalize values for spikes per second
    if trial_num == None:
        trial_num = len(spike_trains)
    values = 1000 * values / bin_size / trial_num

    if axis == None:
        axis = plt.gca()
        axis.plot(bins[:-1], values, **kwargs)
        axis.set_xlabel("Time [ms]")
        axis.set_ylabel("Spikes per second")
        plt.show()
    else:
        axis.plot(bins[:-1], values, **kwargs)
        axis.set_xlabel("Time [ms]")
        axis.set_ylabel("Spikes per second")



def plot_isih(spike_trains, bin_size=1, trial_num=None, axis=None, **kwargs):
    """
    Plot inter-spike interval histogram.
    """
    isi_trains = [ np.diff(train) for train in spike_trains ]

    all_isi = np.concatenate( isi_trains )

    nbins = np.ceil((max(all_isi) - min(all_isi)) / bin_size)

    values, bins = np.histogram(all_isi, nbins)

    # Normalize values
    if trial_num == None:
        trial_num = len(spike_trains)
    # values = values / bin_size / trial_num
    values = values / trial_num

    if axis == None:
        axis = plt.gca()
        axis.plot(bins[:-1], values, **kwargs)
        axis.set_xlabel("Inter-Spike Interval [ms]")
        axis.set_ylabel("Interval #")
        plt.show()
    else:
        axis.plot(bins[:-1], values, **kwargs)
        axis.set_xlabel("Inter-Spike Interval [ms]")
        axis.set_ylabel("Interval #")





def synchronization_index(Fstim, spike_trains):
    """
    Calculate Synchronization Index.

    Fstim: stimulus frequency in Hz
    spike_trains: list of arrays of spiking times
    min_max: range for SI calculation

    return: synchronization index

    >>> fs = 36000.0
    >>> Fstim = 100.0

    >>> test0 = [np.arange(0, 0.1, 1/fs)*1000, np.arange(0, 0.1, 1/fs)*1000]
    >>> si0 = synchronization_index(Fstim, test0)
    >>> si0 < 1e-4   # numerical errors
    True

    >>> test1 = [np.zeros(fs)]
    >>> si1 = synchronization_index(Fstim, test1)
    >>> si1 == 1
    True
    """
    Fstim = Fstim / 1000        # Hz -> kHz; s -> ms

    all_spikes = np.concatenate(tuple(spike_trains))

    if len(all_spikes) == 0:
        return 0

    all_spikes = all_spikes - all_spikes.min()

    folded = np.fmod(all_spikes, 1/Fstim)
    ph = np.histogram(folded, bins=180)[0]

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
    """
    Computes unnormalized correlation index. (Joris et al. 2006)


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
    """
    Get input spikes.  Randomly permute inter spikes intervals.
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



def average_firing_rate(spike_trains, stimulus_duration=None):
    """
    Calculates average firing rate.

    spike_trains: trains of spikes
    stimulus_duration: in ms, if None, then calculated from spike timeings

    return: average firing rate in spikes per second (Hz)

    >>> spike_trains = [range(20), range(10)]
    >>> average_firing_rate(spike_trains, 1000)
    15.0
    """
    all_spikes = np.concatenate(tuple(spike_trains))
    if stimulus_duration == None:
        stimulus_duration = all_spikes.max() - all_spikes.min()
    trial_num = len(spike_trains)
    r = all_spikes.size / (stimulus_duration * trial_num)
    r = r * 1000                # kHz -> Hz
    return r


firing_rate = average_firing_rate



def count_spikes(spike_trains):
    all_spikes = np.concatenate(tuple(spike_trains))
    return len(all_spikes)

count = count_spikes


def correlation_index(spike_trains, coincidence_window=0.05, stimulus_duration=None):
    """
    Comput correlation index (Joris 2006)
    """
    if len(spike_trains) == 0:
        return 0

    if stimulus_duration == None:
        all_spikes = np.concatenate(tuple(spike_trains))
        stimulus_duration = all_spikes.max() - all_spikes.min()

    firing_rate = average_firing_rate(spike_trains, stimulus_duration)
    firing_rate = firing_rate / 1000
    # average_firing_rate() takes input in ms and output in sp/s, threfore:
    # Hz -> kHz

    trial_num = len(spike_trains)

    # Compute raw CI and normalize it
    ci = (_raw_correlation_index(spike_trains) /
          ( trial_num*(trial_num-1) * firing_rate**2 * coincidence_window * stimulus_duration))

    return ci


ci = correlation_index


def shuffled_autocorrelation(spike_trains, coincidence_window=0.05, analysis_window=5,
                             stimulus_duration=None):
    """
    Calculate Shuffled Autocorrelogram (Joris 2006)

    >>> a = [np.array([1, 2, 3]), np.array([1, 2.01, 2.5])]
    >>> shuffled_autocorrelation(a, coincidence_window=1, analysis_window=2)
    (array([-1.2, -0.4,  0.4,  1.2,  2. ]), array([ 0.11111111,  0.55555556,  0.44444444,  0.55555556,  0.11111111]))
    """
    if stimulus_duration == None:
        all_spikes = np.concatenate(tuple(spike_trains))
        stimulus_duration = all_spikes.max() - all_spikes.min()
    firing_rate = average_firing_rate(spike_trains, stimulus_duration)
    firing_rate = firing_rate / 1000
    # average_firing_rate() takes input in ms and output in sp/s, threfore:
    # Hz -> kHz

    trial_num = len(spike_trains)

    cum = []
    for i in range(len(spike_trains)):
        other_trains = list(spike_trains)
        train = other_trains.pop(i)
        almost_all_spikes = np.concatenate(tuple(other_trains))

        for spike in train:
            centered = almost_all_spikes - spike
            trimmed = centered[(centered > -analysis_window) & (centered < analysis_window)]
            cum.append(trimmed)

    cum = np.concatenate(tuple(cum))

    hist, bin_edges = np.histogram(cum,
                                   bins=np.floor(2*analysis_window/coincidence_window)+1,
                                   range=(-analysis_window, analysis_window))
    sac = (hist /
           ( trial_num*(trial_num-1) * firing_rate**2 * coincidence_window * stimulus_duration))

    t = bin_edges[0:-1] + (bin_edges[1] - bin_edges[0])

    return t, sac



sac = shuffled_autocorrelation


def plot_sac(spike_trains, coincidence_window=0.05, analysis_window=5,
             stimulus_duration=None, axis=None, **kwargs):
    """
    Plot shuffled autocorrelogram (SAC) (Joris 2006)
    """

    t, sac = shuffled_autocorrelation(spike_trains, coincidence_window,
                                      analysis_window, stimulus_duration)

    if axis == None:
        axis = plt.gca()
        axis.plot(t, sac, **kwargs)
        axis.set_xlabel("Delay [ms]")
        axis.set_ylabel("Normalized # coincidences")
        plt.show()
    else:
        axis.plot(t, sac, **kwargs)
        axis.set_xlabel("Delay [ms]")
        axis.set_ylabel("Normalized # coincidences")



def split_trains(spike_trains, idx):
    """
    Returns two spike trains created by spliting the input spike_train
    at index `idx'.
    """
    left = spike_trains[0:idx]
    right = spike_trains[idx:]

    return left, right


def pop_trains(spike_trains, num):
    """
    Pop `num' of trains from `spike_trains'.
    """
    popped = [ spike_trains.pop() for each in range(num) ]

    popped.reverse()

    return popped



def trim_spikes(spike_trains, start, stop):
    """
    Return spike trains with that are between `start' and `stop'.

    >>> spikes = [np.array([1,2,3,4]), np.array([3,4,5,6])]
    >>> trim_spikes(spikes, 2, 4)
    [array([2, 3, 4]), array([3, 4])]
    """
    trimmed = [ train[(train >= start) & (train <= stop)]
                for train in spike_trains ]

    return trimmed

trim = trim_spikes



def concat_and_fold(spike_trains, period):
    all_spikes = np.concatenate( tuple(spike_trains) )
    return [ np.fmod(all_spikes, period) ]



def fold_spikes(spike_trains, period):
    """
    Fold each of the spike trains.

    >>> spike_trains = [np.array([1,2,3,4]), np.array([3,4,5,6])]
    >>> fold_spikes(spike_trains, 2)
    [array([1]), array([0, 1]), array([1]), array([0, 1])]
    """
    folded = []
    for train in spike_trains:
        period_num = int( np.ceil(train.max() / period) )
        for idx in range( period_num ):
            lo = idx * period
            hi = (idx+1) * period
            sec = train[(train>=lo) & (train<hi)]
            if len(sec) > 0:
                sec = np.fmod(sec, period)
                folded.append( sec )
    return folded
fold = fold_spikes



def concatenate_spikes(spike_trains):
    return [np.concatenate( tuple(spike_trains) )]

concatenate = concatenate_spikes
concat = concatenate_spikes

if __name__ == "__main__":
    import doctest

    print "Doctest start:"
    doctest.testmod()
    print "done."

    # test_shuffle_spikes()
