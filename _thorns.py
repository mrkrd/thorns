from __future__ import division

import numpy as np
from numpy.random import shuffle

golden = 1.6180339887


def _signal_to_spikes(fs, signal):
    """ Convert 1D time function array into array of spike timings.

    fs: sampling frequency in Hz
    signal: input signal

    return: spike timings in ms

    >>> fs = 10
    >>> signal = np.array([0,2,0,0,1,0])
    >>> _signal_to_spikes(fs, signal)
    array([ 100.,  100.,  400.])

    """
    assert signal.ndim == 1
    assert (np.mod(signal, 1) == 0).all()

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
        spike_trains = [ _signal_to_spikes(fs, signals) ]
    elif signals.ndim == 2:
        spike_trains = [ _signal_to_spikes(fs, signal)
                         for signal in signals.T ]
    else:
        assert False, "Input signal must be 1 or 2 dimensional"

    return spike_trains


def _spikes_to_signal(fs, spikes, tmax=None):
    """ Convert spike train to its time function. 1D version.

    >>> fs = 10
    >>> spikes = np.array([100, 500, 1000, 1000])
    >>> _spikes_to_signal(fs, spikes)
    array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2])

    """
    if tmax == None:
        tmax = max(spikes)

    bins = np.ceil(tmax*fs/1000) + 1
    signal, bin_edges = np.histogram(spikes, bins=bins, range=(0,tmax))

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

    signals = [_spikes_to_signal(fs, train, tmax) for train in spike_trains]
    signals = np.array(signals).T

    # import matplotlib.pyplot as plt
    # plt.imshow(signals, aspect='auto')
    # plt.show()
    return signals


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
    axis: axis to draw on
    **kwargs: plt.plot arguments
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

    return plot


def calc_isih(spike_trains, bin_size=1, trial_num=None):
    """ Calculate inter-spike interval histogram.

    >>> spikes = [np.array([1,2,3]), np.array([2,5,8])]
    >>> calc_isih(spikes)
    (array([ 0.,  1.,  1.]), array([ 0.,  1.,  2.,  3.]))

    """
    isi_trains = [ np.diff(train) for train in spike_trains ]

    all_isi = np.concatenate(isi_trains)

    if len(all_isi) < 2:
        return np.array([]), np.array([])

    nbins = np.ceil(all_isi.max() / bin_size)

    if nbins == 0:              # just in case of two equal spikes
        nbins = 1

    values, bins = np.histogram(all_isi, nbins, range=(0,all_isi.max()))

    # Normalize values
    if trial_num == None:
        trial_num = len(spike_trains)
    # values = values / bin_size / trial_num
    values = values / trial_num

    return values, bins


def calc_entrainment(spike_trains, fstim, bin_size=1):
    """ Calculate entrainment of spike_trains.

    >>> spike_trains = [np.array([2, 4, 6]), np.array([0, 5, 10])]
    >>> calc_entrainment(spike_trains, fstim=500)
    0.5
    """
    isih, bins = calc_isih(spike_trains, bin_size=bin_size)

    if len(isih) == 0:
        return 0

    stim_period = 1000/fstim    # ms

    entrainment_window = (bins[:-1] > stim_period/2) & (bins[:-1] < stim_period*3/2)

    entrainment =  np.sum(isih[entrainment_window]) / np.sum(isih)

    return entrainment


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

    return plot


def plot_period_histogram(spike_trains, fstim, nbins=64, plot=None, **style):
    """ Plots period histogram. """
    import biggles

    fstim = fstim / 1000        # Hz -> kHz; s -> ms

    if len(spike_trains) == 0:
        return 0

    all_spikes = np.concatenate(tuple(spike_trains))

    if len(all_spikes) == 0:
        return 0

    folded = np.fmod(all_spikes, 1/fstim)
    ph,edges = np.histogram(folded, bins=nbins, range=(0,1/fstim))

    # Normalize
    ph = ph / np.sum(ph)

    # TODO: find the direction instead of max value
    center_idx = ph.argmax()
    ph = np.roll(ph, nbins//2 - center_idx)

    c = biggles.Histogram(ph, x0=0, binsize=1/len(ph))
    c.style(**style)

    if plot is None:
        plot = biggles.FramedPlot()
    plot.xlabel = "Normalized Phase"
    plot.ylabel = "Normalized Spike Count"
    plot.add(c)

    return plot


def calc_synchronization_index(spike_trains, fstim):
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

    fstim = fstim / 1000        # Hz -> kHz; s -> ms

    if len(spike_trains) == 0:
        return 0

    all_spikes = np.concatenate(tuple(spike_trains))

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


calc_si = calc_synchronization_index
calc_vector_strength = calc_synchronization_index
calc_vs = calc_synchronization_index



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


def calc_average_firing_rate(spike_trains, stimulus_duration=None, trial_num=None):
    """ Calculates average firing rate.

    spike_trains: trains of spikes
    stimulus_duration: in ms, if None, then calculated from spike timeings

    return: average firing rate in spikes per second (Hz)

    >>> spike_trains = [range(20), range(10)]
    >>> calc_average_firing_rate(spike_trains, 1000)
    15.0

    """
    if len(spike_trains) == 0:
        return 0
    all_spikes = np.concatenate(tuple(spike_trains))
    if stimulus_duration == None:
        stimulus_duration = all_spikes.max() - all_spikes.min()
    if trial_num == None:
        trial_num = len(spike_trains)
    r = all_spikes.size / (stimulus_duration * trial_num)
    r = r * 1000                # kHz -> Hz
    return r


calc_firing_rate = calc_average_firing_rate
calc_rate = calc_average_firing_rate


def count_spikes(spike_trains):
    all_spikes = np.concatenate(tuple(spike_trains))
    return len(all_spikes)

count = count_spikes


def calc_correlation_index(spike_trains, coincidence_window=0.05, stimulus_duration=None):
    """ Compute correlation index (Joris 2006) """
    if len(spike_trains) == 0:
        return 0

    all_spikes = np.concatenate(tuple(spike_trains))
    if len(all_spikes) == 0:
        return 0

    if stimulus_duration == None:
        stimulus_duration = all_spikes.max() - all_spikes.min()

    firing_rate = calc_average_firing_rate(spike_trains, stimulus_duration)
    firing_rate = firing_rate / 1000
    # calc_average_firing_rate() takes input in ms and output in sp/s, threfore:
    # Hz -> kHz

    trial_num = len(spike_trains)

    # Compute raw CI and normalize it
    ci = (_raw_correlation_index(spike_trains) /
          ( trial_num*(trial_num-1) * firing_rate**2 * coincidence_window * stimulus_duration))

    return ci


calc_ci = calc_correlation_index


def calc_shuffled_autocorrelation(spike_trains, coincidence_window=0.05, analysis_window=5,
                                  stimulus_duration=None):
    """ Calculate Shuffled Autocorrelogram (Joris 2006)

    >>> a = [np.array([1, 2, 3]), np.array([1, 2.01, 2.5])]
    >>> calc_shuffled_autocorrelation(a, coincidence_window=1, analysis_window=2)
    (array([-1.2, -0.4,  0.4,  1.2,  2. ]), array([ 0.11111111,  0.55555556,  0.44444444,  0.55555556,  0.11111111]))

    """
    if stimulus_duration == None:
        all_spikes = np.concatenate(tuple(spike_trains))
        stimulus_duration = all_spikes.max() - all_spikes.min()
    firing_rate = calc_average_firing_rate(spike_trains, stimulus_duration)
    firing_rate = firing_rate / 1000
    # calc_average_firing_rate() takes input in ms and output in sp/s, threfore:
    # Hz -> kHz

    trial_num = len(spike_trains)

    cum = []
    for i in range(len(spike_trains)):
        other_trains = list(spike_trains)
        train = other_trains.pop(i)
        almost_all_spikes = np.concatenate(other_trains)

        for spike in train:
            centered = almost_all_spikes - spike
            trimmed = centered[(centered > -analysis_window) & (centered < analysis_window)]
            cum.append(trimmed)

    cum = np.concatenate(cum)

    hist, bin_edges = np.histogram(cum,
                                   bins=np.floor(2*analysis_window/coincidence_window)+1,
                                   range=(-analysis_window, analysis_window))
    sac = (hist /
           ( trial_num*(trial_num-1) * firing_rate**2 * coincidence_window * stimulus_duration))

    t = bin_edges[0:-1] + (bin_edges[1] - bin_edges[0])

    return t, sac


calc_sac = calc_shuffled_autocorrelation


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


def split_trains(spike_trains, idx):
    """ Returns two spike trains created by spliting the input spike_train
    at index `idx'.

    """
    left = spike_trains[0:idx]
    right = spike_trains[idx:]

    return left, right


def pop_trains(spike_trains, num):
    """  Pop `num' of trains from `spike_trains'. """
    popped = [ spike_trains.pop() for each in range(num) ]

    popped.reverse()

    return popped


def trim_spikes(spike_trains, start, stop=None):
    """ Return spike trains with that are between `start' and `stop'.

    >>> spikes = [np.array([1,2,3,4]), np.array([3,4,5,6])]
    >>> trim_spikes(spikes, 2, 4)
    [array([0, 1, 2]), array([1, 2])]

    """
    all_spikes = np.concatenate(tuple(spike_trains))
    if len(all_spikes) == 0:
        return spike_trains

    if stop is None:
        stop = np.concatenate(tuple(spike_trains)).max()

    trimmed = [ train[(train >= start) & (train <= stop)]
                for train in spike_trains ]

    shifted = shift_spikes(trimmed, -start)

    return shifted


trim = trim_spikes


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


def split_and_fold_trains(long_train, silence_duration, tone_duration, pad_duration):
    silence = trim(long_train, 0, silence_duration)

    tones = trim(long_train, silence_duration)
    tones = fold(tones, tone_duration+pad_duration)

    return silence, tones

split_and_fold = split_and_fold_trains


if __name__ == "__main__":
    import doctest

    print "Doctest start:"
    doctest.testmod()
    print "done."

    # test_shuffle_spikes()
