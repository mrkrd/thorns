import numpy as np

import thorns as th


def test_spikes_and_signals():
    fs = 33

    signal = np.array([0, 1, 0, 2, 0, 1])
    print "signal:", signal

    spikes = th.signal_to_spikes_1D(fs, signal)
    print "spikes:", spikes

    signal = th.spikes_to_signal_1D(fs, spikes)
    print "signal:", signal


if __name__ == "__main__":
    test_spikes_and_signals()
