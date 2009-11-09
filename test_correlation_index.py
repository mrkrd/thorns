import numpy as np
import matplotlib.pyplot as plt

import thorns as th
import cochlea


def test_ci():
    a = [np.array([1, 2, 4, 6]),
         np.array([1, 2.01, 5])]

    print th.correlation_index(a)

    fs = 100000.0               # Hz
    bf = 1000.0                 # Hz
    ear = cochlea.Sumner2002(hsr=1, msr=0, lsr=0, freq=bf)
    t = np.arange(0, 0.1, 1/fs)
    s = np.sin(2 * np.pi * t * bf)
    s = cochlea.set_dB_SPL(60, s)
    hsr, msr, lsr = ear.run(fs, s, times=200)

    spikes = hsr['spikes']
    # th.plot_psth(spikes)
    # th.plot_raster(spikes)

    print th.correlation_index(spikes)

def test_sac():
    # a = [np.array([1, 2, 3]), np.array([1, 2.01, 4])]
    # t, sac = th.shuffled_autocorrelation(a)


    b = np.arange(1,2,0.05)
    c = np.arange(1,2,0.05)
    spikes = [b, np.array([1])]
    t, sac = th.shuffled_autocorrelation(spikes, coincidence_window=0.05)
    plt.plot(t, sac)
    plt.show()


    fs = 100000.0               # Hz
    bf = 500.0                  # Hz
    ear = cochlea.Sumner2002(hsr=0, msr=0, lsr=1, freq=bf)
    t = np.arange(0, 0.1, 1/fs)
    s = np.sin(2 * np.pi * t * bf)
    s = cochlea.set_dB_SPL(60, s)
    hsr, msr, lsr = ear.run(fs, s, times=200)
    spikes = lsr['spikes']

    th.plot_psth(spikes, bin_size=0.2, trial_num=200)
    th.plot_isih(spikes, bin_size=0.1, trial_num=200)

    print "Computing SAC:"
    t, sac = th.shuffled_autocorrelation(spikes, coincidence_window=0.05)

    plt.plot(t, sac)
    plt.show()

    import scipy.io
    scipy.io.savemat("lsr.mat", {"spikes": spikes, "t": t, "sac": sac})

    print sac, np.sum(sac)


if __name__ == "__main__":
    # test_ci()
    test_sac()
