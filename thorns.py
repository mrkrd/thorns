# Thorns:  spike analysis software

import numpy as np

def signal_to_spikes_1D(fs, f):
    """
    Convert time function 1D array into spike events array.
    """
    f = np.asarray(f).astype(int)

    fs = float(fs)

    non_empty = np.where(f > 0)[0]
    spikes = []
    for time_idx in non_empty:
        time_samp = f[time_idx]
        spikes.extend([1000 * time_idx/fs for each in range(time_samp)]) # ms

    return np.array(spikes)


def signal_to_spikes(fs, f):
    """
    Convert time functions to a list of spike trains.
    """
    fs = float(fs)

    spike_trains = []

    if f.ndim == 1:
        spike_trains.append(signal_to_spikes_1D(fs, f))
    elif f.ndim == 2:
        for f_1D in f.T:
            spike_trains.append(signal_to_spikes_1D(fs, f_1D))
    else:
        assert False

    return spike_trains



def spikes_to_signal_1D(fs, spikes, tmax=None):
    """
    Convert spike train to its time function. 1D version.
    """
    # Test if all elements are floats (1D case)
    #assert np.all([isinstance(each, float) for each in spikes])

    # TODO: implement with np.histogram

    if tmax == None:
        tmax = max(spikes)

    f = np.zeros(np.floor(tmax * fs) + 1)

    # TODO: vecotorize
    # f[np.floor(spikes * fs).astype(int)] += 1
    for spike in spikes:
        f[int(np.floor(spike*fs))] += 1

    return f


def spikes_to_signal(fs, spikes):
    """
    Convert spike trains to theirs time functions.
    """
    # All elements are list (2D case)
    if np.all([isinstance(each, np.ndarray) for each in spikes]):

        f_list = []
        for train in spikes:
            f1d = spikes_to_signal_1D(fs, train)
            f_list.append(f1d)

        # Adjust length of each time function to the maximum
        max_len = max([len(f1d) for f1d in f_list])

        # Construct output
        f = np.zeros( (max_len, len(f_list)) )

        for i,f1d in enumerate(f_list):
            f[0:len(f1d), i] = f1d



    else:
        f = spikes_to_signal_1D(fs, spikes)

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



def synchronization_index(Fstim, spikes):
    """
    Calculate Synchronization Index.

    Fstim: stimulus frequency in Hz
    spikes: list of arrays of spiking times

    return: synchronization index
    """
    Fstim = float(Fstim)
    Fstim = Fstim / 1000        # Hz -> kHz

    all_spikes = np.concatenate(tuple(spikes))

    # TODO: remove here spikes from the beginning and the end

    all_spikes = all_spikes - all_spikes.min()

    #bins = 360                  # number of bins per Fstim period
    # hist_range = (0, 1/Fstim)
    # ph = np.zeros(bins)
    # period_num = np.floor(all_spikes.max() * Fstim) + 1

    # for i in range(period_num):
    #     lo = i / Fstim
    #     hi = lo + 1/Fstim
    #     current_spikes = all_spikes[(all_spikes >= lo) & (all_spikes < hi)]
    #     ph += np.histogram(current_spikes-lo, bins=bins, range=hist_range)[0]

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



def test_synchronization_index():
    fs = 36000.0
    Fstim = 100.0

    test0 = [np.arange(0, 0.1, 1/fs)*1000, np.arange(0, 0.1, 1/fs)*1000]
    si0 = synchronization_index(Fstim, test0)
    print "test0 SI: ", si0,
    assert si0 < 1e-4
    print "OK"

    test1 = [np.zeros(fs)]
    si1 = synchronization_index(Fstim, test1)
    print "test1 SI: ", si1,
    assert si1 == 1
    print "OK"



if __name__ == "__main__":
    test_synchronization_index()
