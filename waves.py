#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np



def set_dbspl(signal, dbspl):

    if np.issubdtype(signal.dtype, int):
        signal = signal.astype(float)

    p0 = 2e-5
    rms = np.sqrt( np.sum(signal**2) / len(signal) )

    scalled = signal * 10**(dbspl / 20.0) * p0 / rms

    return scalled





def make_ramped_tone(
        fs,
        freq,
        tone_duration=50e-3,
        ramp_duration=2.5e-3,
        pad_duration=55e-3,
        dbspl=None):
    """ Generate ramped tone singal.

    fs: sampling frequency [Hz]
    freq: frequency of the tone [Hz]
    tone_durations: [s]
    ramp_duration: [s]
    pad_duration: [s]
    dbspl: dB SPL

    """
    t = np.arange(0, tone_duration, 1/fs)
    s = np.sin(2 * np.pi * t * freq)
    if dbspl != None:
        s = set_dbspl(s, dbspl)

    ramp = np.linspace(0, 1, np.ceil(ramp_duration * fs))
    s[0:len(ramp)] = s[0:len(ramp)] * ramp
    s[-len(ramp):] = s[-len(ramp):] * ramp[::-1]

    pad = np.zeros(pad_duration * fs)
    s = np.concatenate( (s, pad) )

    return s


