#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import scipy.signal as dsp


def root_mean_square(s):
    return np.sqrt( np.sum(s**2) / s.size )

rms = root_mean_square



def filter(signal, fs, band):

    lo, hi = band

    freqs = np.linspace(0, fs/2, len(signal)/2+1)


    signal_fft = np.fft.rfft(signal)
    signal_fft[ (freqs < lo) | (freqs > hi) ] = 0

    filtered = np.fft.irfft(signal_fft)

    return filtered




def set_dbspl(signal, dbspl):
    p0 = 20e-6
    rms = np.sqrt( np.sum(signal**2) / signal.size )

    scalled = signal * 10**(dbspl / 20.0) * p0 / rms

    return scalled


def resample(signal, fs, new_fs):
    new_signal = dsp.resample(signal, len(signal)*new_fs/fs)
    return new_signal



def cut_add(a,b):
    assert a.ndim == b.ndim == 1

    length = min([len(a), len(b)])
    s = a[0:length] + b[0:length]

    return s




def make_ramped_tone(
        fs,
        freq,
        duration=50e-3,
        ramp=2.5e-3,
        pad=0,
        dbspl=None):
    """ Generate ramped tone singal.

    fs: sampling frequency [Hz]
    freq: frequency of the tone [Hz]
    tone_durations: [s]
    ramp: [s]
    pad: [s]
    dbspl: dB SPL

    """
    t = np.arange(0, duration, 1/fs)
    s = np.sin(2 * np.pi * t * freq)
    if dbspl != None:
        s = set_dbspl(s, dbspl)

    ramp_signal = np.linspace(0, 1, np.ceil(ramp * fs))
    s[0:len(ramp_signal)] = s[0:len(ramp_signal)] * ramp_signal
    s[-len(ramp_signal):] = s[-len(ramp_signal):] * ramp_signal[::-1]

    pad_signal = np.zeros(pad * fs)
    s = np.concatenate( (s, pad_signal) )

    return s



def make_white_noise(
        fs,
        duration,
        band,
        seed,
        ramp=2.5e-3,
        pad=0,
        dbspl=None
    ):

    np.random.seed(seed)


    lo, hi = band

    n = int( np.round(duration * fs) )
    freqs_abs = np.abs( np.fft.fftfreq(n, 1/fs) )

    passband_mask = (freqs_abs > lo) & (freqs_abs < hi)

    angles = np.random.rand(n) * 2 * np.pi

    reals = np.cos(angles)
    imags = np.sin(angles)

    ss = reals + 1j*imags
    ss[ np.invert(passband_mask) ] = 0

    s = np.fft.ifft(ss).real

    s = set_dbspl(s, dbspl)


    ### Ramping
    ramp_signal = np.linspace(0, 1, np.round(ramp*fs))
    s[0:len(ramp_signal)] = s[0:len(ramp_signal)] * ramp_signal
    s[-len(ramp_signal):] = s[-len(ramp_signal):] * ramp_signal[::-1]


    ### Padding
    pad_signal = np.zeros(pad * fs)
    s = np.concatenate( (s, pad_signal) )


    return s





def make_electrical_pulse(
        fs,
        amplitudes,
        durations,
        gap=0,
        pad=0):


    assert len(amplitudes) == len(durations)

    gap_signal = np.zeros(gap * fs)
    pad_signal = np.zeros(pad * fs)

    signals = []
    for amp,dur in zip(amplitudes, durations):

        signals.append( amp * np.ones(dur * fs) )
        signals.append( gap_signal )

    # Remove the last gap
    signals.pop(-1)


    signals.append( pad_signal )

    signal = np.concatenate( signals )

    return signal
