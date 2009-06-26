# thorns: spike analysis toolbox

import numpy as np

def timef_to_spikes_1D(fs, f):
    """
    Convert time function 1D array into spike events array.
    """
    f = np.asarray(f).astype(int)

    fs = float(fs)

    # non_empty = np.where(f > 0)[0]

    # for time_idx in non_empty:
    #     time_samp = f[time_idx]
    #     spikes.extend([1000 * time_idx/fs for each in range(time_samp)]) # ms

    spikes = [ [i*fs]*n for i,n in enumerate(f) ]

    return np.concatenate(spikes)


def timef_to_spikes(fs, f):
    """
    Convert time functions to a list of spike events.
    """
    fs = float(fs)

    spike_trains = []

    if f.ndim == 1:
        spike_trains.append(timef_to_spikes_1D(fs, f))
    elif f.ndim == 2:
        for f_1D in f:
            spike_trains.append(timef_to_spikes_1D(fs, f_1D))
    else:
        assert False

    return spike_trains



def spikes_to_timef_1D(fs, spikes, tmax=None):
    """
    Convert spike train to its time function. 1D version.
    """
    # Test if all elements are floats (1D case)
    #assert np.all([isinstance(each, float) for each in spikes])

    if tmax == None:
        tmax = max(spikes)

    f = np.zeros(np.floor(tmax * fs) + 1)

    # TODO: vecotorize
    # f[np.floor(spikes * fs).astype(int)] += 1
    for spike in spikes:
        f[int(np.floor(spike*fs))] += 1

    return f


def spikes_to_timef(fs, spikes):
    """
    Convert spike trains to theirs time functions.
    """
    # All elements are list (2D case)
    if np.all([isinstance(each, np.ndarray) for each in spikes]):

        f_list = []
        for train in spikes:
            f1d = spikes_to_timef_1D(fs, train)
            f_list.append(f1d)

        # Adjust length of each time function to the maximum
        max_len = max([len(f1d) for f1d in f_list])

        # Construct output
        f = np.zeros( (max_len, len(f_list)) )

        for i,f1d in enumerate(f_list):
            f[0:len(f1d), i] = f1d



    else:
        f = spikes_to_timef_1D(fs, spikes)

    return f


def plot_raster(spikes):
    import matplotlib.pyplot as plt

    # Assert list of arrays as input
    assert np.all([isinstance(each, np.ndarray) for each in spikes])

    n = []
    s = []
    for i,train in enumerate(spikes):
        s.extend(train)                 # flatten spikes
        n.extend([i for each in train]) # trial number

    plt.plot(s, n, 'o')
    plt.show()


def plot_psth(spikes, bin_size=1, ax=None, tmax=None):
    """
    Plots PSTH from spikes (list of arrays of spike timings)

    bin_size: bin size in ms
    ax: axis to draw on
    """
    import matplotlib.pyplot as plt

    # Assert list of arrays as input
    assert np.all([isinstance(each, np.ndarray) for each in spikes])

    all_spikes = []
    for train in spikes:
        all_spikes.extend(train)

    # Remove entries greater than tmax
    if tmax != None:
        all_spikes = np.delete(all_spikes, np.where(np.asarray(all_spikes) > tmax)[0])

    if len(all_spikes) != 0:

        # bin size = 1ms
        nbins = np.floor((max(all_spikes) - min(all_spikes)) / float(bin_size))


        if ax == None:
        #plt.hist(all_spikes, nbins)
            plt.hist(all_spikes, nbins)
            plt.show()
        else:
        #ax.hist(all_spikes, nbins)
            n, bins = np.histogram(all_spikes, nbins)
            ax.plot(bins[0:-1], n)



def calc_SI(fs, fstim, psth):
    """
    Computes Syncronization Index.

    fs: sampling frequency
    fstim: stimulus frequency (sine wave)
    spieks: clean (without padding) PST histogram

    return: syncronization index
    """
    fs = float(fs)
    fstim = float(fstim)

    signal_samp = len(psth)
    signal_sec = len(psth) / fs   # s

    period_samp = np.floor(fs / fstim)
    period_sec = 1 / fstim         # s

    # number of periods
    period_num = np.floor(signal_sec / period_sec)

    start_sec = np.arange(0, period_num * period_sec, period_sec)
    start_samp = np.floor(start_sec * fs)

    # Generate index
    # [[ t1   t2   t3 ]
    #  [t1+1 t2+1 t3+1]
    #  [t1+2 t2+2 t3+2]
    #  [t1+3 t2+3 t3+3]
    #  [t1+4 t2+4 t3+4]]
    # where t1, t2, t3 are initial indexes of each stimulus period
    idx = np.repeat([start_samp], period_samp, axis=0)
    range_sqr = np.repeat([np.arange(period_samp)], len(start_samp), axis=0).T
    idx = idx + range_sqr

    psth_fold = psth[idx.astype(int)]

    ph = np.sum(psth_fold, axis=1)


    # indexing trick is necessary, because the sample at 2*pi belongs
    # to the next cycle
    x = np.cos(np.linspace(0, 2*np.pi, len(ph)+1))[0:-1]
    y = np.sin(np.linspace(0, 2*np.pi, len(ph)+1))[0:-1]

    xsum2 = (np.sum(x*ph))**2
    ysum2 = (np.sum(y*ph))**2

    r = np.sqrt(xsum2 + ysum2) / np.sum(ph)

    return r


def test_calc_SI():
    fs = 100000.0
    fstim = 800.0

    test0 = np.ones(fs)        # 1 s
    print "test0 SI: ", calc_SI(fs, fstim, test0)

    test1 = np.zeros(fs)
    test1[np.round(fs/2)] = 30
    print "test1 SI: ", calc_SI(fs, fstim, test1)




def set_dB_SPL(dB, signal):
    p0 = 2e-5                   # Pa
    squared = signal**2
    rms = np.sqrt( np.sum(squared) / len(signal) )

    if rms == 0:
        r = 0
    else:
        r = 10**(dB / 20.0) * p0 / rms;


    return signal * r * 1e6     # uPa



if __name__ == "__main__":
    test_calc_SI()
