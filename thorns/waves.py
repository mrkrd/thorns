#!/usr/bin/env python

# Author: Marek Rudnicki
# Time-stamp: <2009-12-14 15:19:49 marek>

# Description:

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# TODO: help, testdoc
def detect_peaks(fs, signal):
    d = np.diff(signal)
    s = np.sign(d)

    sd = np.diff(s)

    peak_idx = np.where(sd==-2)[0] + 1
    peak_val = signal[peak_idx]

    assert ((signal[peak_idx-1] < signal[peak_idx]).all() and
            (signal[peak_idx+1] < signal[peak_idx]).all())

    return peak_idx, peak_val



def root_mean_square(s):
    return np.sqrt( np.sum(s**2) / len(s) )

rms = root_mean_square



def set_dB_SPL(dB, signal):
    p0 = 2e-5                   # Pa
    squared = signal**2
    rms = np.sqrt( np.sum(squared) / len(signal) )

    if rms == 0:
        r = 0
    else:
        r = 10**(dB / 20.0) * p0 / rms;

    return signal * r * 1e6     # uPa

set_dbspl = set_dB_SPL



def make_time(fs, s):
    """
    Based on signal `s' with sampling frequancy `fs' produce time
    vector and return.  Useful for plots.

    >>> make_time(10, [2,3,4,4,5])
    array([ 0.,  1.,  2.,  3.,  4.])
    """
    tmax = (len(s)-1) / fs
    return np.linspace(0, tmax, len(s))

t = make_time



# TODO: generation of standard ramped tones
def generate_ramped_tone(fs, freq,
                         tone_duration=50,
                         ramp_duration=2.5,
                         pad_duration=55,
                         dbspl=None):
    """
    fs: Hz
    durations: ms
    """
    tsin = np.arange(0, tone_duration, 1000/fs)
    s = np.sin(2 * np.pi * tsin * freq/1000)
    if dbspl != None:
        s = set_dbspl(dbspl, s)

    ramp = np.linspace(0, 1, np.ceil(ramp_duration * fs/1000))
    s[0:len(ramp)] = s[0:len(ramp)] * ramp
    s[-len(ramp):] = s[-len(ramp):] * ramp[::-1]

    pad = np.zeros(pad_duration * fs/1000)
    s = np.concatenate( (s, pad) )

    return s




if __name__ == "__main__":
    import doctest

    print "Doctest start:"
    doctest.testmod()
    print "done."
