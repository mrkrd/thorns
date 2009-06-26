
import numpy as np

import thorns as th

def test_timef_to_spikes_1D():
    print "Testing timef_to_spikes_1D...",
    f = [0, 0, 3, 2, 0, 1, 0]
    fs = 1

    expectation = np.array([2, 2, 2, 3, 3, 5])

    out = th.timef_to_spikes_1D(fs, f)

    print f, out

    assert np.all(expectation == out)
    print "OK"


def test_timef_to_spikes():
    print "Testing timef_to_spikes...",
    f = np.array([[0, 0, 3, 2, 0, 1, 0],
                  [0, 0, 2, 2, 0, 1, 0]])
    fs = 1

    expectation = [np.array([2, 2, 2, 3, 3, 5]),
                   np.array([2, 2, 3, 3, 5])]

    out = th.timef_to_spikes(fs, f)

    for i,spikes in enumerate(out):
        assert np.all(out[i] == expectation[i])

    print "OK"

def test_spikes_to_timef_1D():
    print "Testing test_spikes_to_timef_1D... (*)",

    fs = 1
    spikes = [1., 1., 2., 3., 3., 3., 10.]

    expected = np.array([ 0.,  2.,  1.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  1.])

    out = th.spikes_to_timef_1D(fs, spikes)
    assert np.all(expected == out)
    print "OK"


    print "Testing test_spikes_to_timef_1D... (**)",

    fs = 2
    spikes = [1., 1., 0.3, 0.5]

    expected = np.array([ 1.,  1.,  2.])

    out = th.spikes_to_timef_1D(fs, spikes)

    assert np.all(expected == out)
    print "OK"


def test_spikes_to_timef():
    print "Testing test_spikes_to_timef... (*)",

    fs = 1
    spikes = [[1., 1., 2., 3., 3., 3., 10.],
              [1.]]

    expected = np.array([[0., 2., 1., 3., 0., 0., 0., 0., 0., 0., 1.],
                         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T

    out = th.spikes_to_timef(fs, spikes)
    assert np.all(expected == out)
    print "OK"


def test_plot_raster():
    print "Testing plot_raster...",

    spikes = [[1, 3, 5, 7],
              [2, 4, 6, 8],
              [1, 3, 5, 7]]

    th.plot_raster(spikes)
    print "OK?"


if __name__ == "__main__":
    test_timef_to_spikes_1D()
    test_timef_to_spikes()
    test_spikes_to_timef_1D()
    test_spikes_to_timef()
    test_plot_raster()
